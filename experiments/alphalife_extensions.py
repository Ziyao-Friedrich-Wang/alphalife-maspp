#!/usr/bin/env python3
"""Lifecycle extensions: FF6 residual health, liquidity-aware repair, and regime gates."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alphalife_full import (
    add_future_horizons,
    build_states,
    factor_allocation_backtest,
    load_ff6,
    repair_experiment_full,
    stock_level_ic_multi,
    summarize_portfolios_any,
    tstat_mean,
)
from alphalife_mvp import DEFAULT_DATA_ROOT, OOS_DATE, RANK_FEATURES, annualized_stats, factor_return_path, load_factor_returns, pivot_factor_returns


STATE_MULT = {"Healthy": 1.0, "Warning": 0.35, "Decayed": 0.15}
RETURN_COLS = [
    "static_equal",
    "rolling_sharpe",
    "health_only",
    "healthy_only",
    "no_redundancy_penalty",
    "alphalife_full",
    "alphalife_repaired",
    "residual_lifecycle",
    "blend_lifecycle",
    "regime_gated",
    "blend_regime_gated",
    "liquidity_repaired",
    "all_actions_repaired",
]


def normalized(w: pd.Series) -> pd.Series:
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(w), index=w.index)


def rolling_ff6_residuals(wide: pd.DataFrame, ff6: pd.DataFrame, lookback: int = 120, min_obs: int = 60) -> pd.DataFrame:
    ff = ff6.set_index("date")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].copy()
    common = wide.index.intersection(ff.index)
    ymat = wide.loc[common]
    xmat = ff.loc[common]
    out = pd.DataFrame(index=common, columns=wide.columns, dtype=float)
    x_all = np.column_stack([np.ones(len(xmat)), xmat.to_numpy()])
    for col in ymat.columns:
        y = ymat[col].to_numpy()
        res = np.full(len(y), np.nan)
        for i in range(len(y)):
            lo = max(0, i - lookback + 1)
            valid = np.isfinite(y[lo : i + 1]) & np.isfinite(x_all[lo : i + 1]).all(axis=1)
            if valid.sum() < min_obs:
                continue
            xv = x_all[lo : i + 1][valid]
            yv = y[lo : i + 1][valid]
            beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
            res[i] = y[i] - x_all[i] @ beta if np.isfinite(y[i]) and np.isfinite(x_all[i]).all() else np.nan
        out[col] = res
    return out


def build_regime_labels(ff6: pd.DataFrame) -> pd.Series:
    ff = ff6.set_index("date").sort_index()
    mkt = ff["Mkt-RF"]
    vol12 = mkt.rolling(12, min_periods=8).std()
    ret12 = mkt.rolling(12, min_periods=8).sum()
    high_vol = vol12 >= vol12.expanding(min_periods=60).quantile(0.70)
    bear = ret12 <= ret12.expanding(min_periods=60).quantile(0.30)
    label = pd.Series("normal", index=ff.index)
    label[high_vol] = "high_vol"
    label[bear] = "bear"
    label[bear & high_vol] = "bear_high_vol"
    return label


def rolling_regime_gate(wide: pd.DataFrame, ff6: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    labels = build_regime_labels(ff6).reindex(wide.index)
    gates = pd.DataFrame(1.0, index=wide.index, columns=wide.columns)
    for i, date in enumerate(wide.index):
        if i < max(36, lookback // 2):
            continue
        lo = max(0, i - lookback + 1)
        current_regime = labels.iloc[i]
        hist_regimes = labels.iloc[lo : i + 1]
        same = hist_regimes == current_regime
        if same.sum() < 12:
            continue
        hist = wide.iloc[lo : i + 1]
        regime_mean = hist.loc[same.to_numpy()].mean()
        global_mean = hist.mean()
        # Deliberately coarse gates to reduce overfitting.
        gate = pd.Series(0.75, index=wide.columns)
        gate[(regime_mean > global_mean) & (regime_mean > 0)] = 1.20
        gate[regime_mean < 0] = 0.35
        gates.loc[date] = gate
    return gates


def trailing_best_weighting(
    returns_by_weighting: dict[str, pd.DataFrame],
    alpha: str,
    idx: int,
    lookback: int,
    choices: list[str],
) -> str:
    best, best_score = choices[0], -np.inf
    for w in choices:
        if alpha not in returns_by_weighting[w].columns:
            continue
        hist = returns_by_weighting[w][alpha].iloc[max(0, idx - lookback) : idx].dropna()
        if len(hist) < max(24, lookback // 2):
            continue
        sd = hist.std(ddof=1)
        score = hist.mean() / sd if sd > 0 else -np.inf
        if score > best_score:
            best, best_score = w, score
    return best


def make_extended_portfolios(
    raw_states: pd.DataFrame,
    residual_states: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    gates: pd.DataFrame,
    lookback: int,
    oos_date: str,
) -> pd.DataFrame:
    base = returns_by_weighting["vw"]
    raw_idx = raw_states.set_index(["date", "alpha"]).sort_index()
    res_idx = residual_states.set_index(["date", "alpha"]).sort_index()
    rows = []
    for idx, date in enumerate(base.index[:-1]):
        if date < pd.Timestamp(oos_date):
            continue
        if date not in raw_idx.index.get_level_values(0) or date not in res_idx.index.get_level_values(0):
            continue
        next_date = base.index[idx + 1]
        ret_next = base.loc[next_date].dropna()
        names = ret_next.index.intersection(raw_idx.loc[date].index).intersection(res_idx.loc[date].index)
        if len(names) == 0:
            continue
        ret_next = ret_next.loc[names]
        raw = raw_idx.loc[date].loc[names]
        res = res_idx.loc[date].loc[names]
        gate = gates.loc[date, names].fillna(1.0)

        raw_mult = raw["state"].map(STATE_MULT).fillna(0.0)
        res_mult = res["state"].map(STATE_MULT).fillna(0.0)
        raw_score = (raw["health_score"] - raw["health_score"].quantile(0.30)).clip(lower=0) * raw_mult
        res_score = (res["health_score"] - res["health_score"].quantile(0.30)).clip(lower=0) * res_mult
        blend_score = 0.65 * raw_score + 0.35 * res_score

        w_residual = normalized(res_score)
        w_blend = normalized(blend_score)
        w_regime = normalized(raw_score * gate)
        w_blend_regime = normalized(blend_score * gate)

        repaired_all = {}
        repaired_liquid = {}
        for alpha in names:
            all_w = trailing_best_weighting(returns_by_weighting, alpha, idx, lookback, [w for w in ["vw", "ew", "vw_cap"] if w in returns_by_weighting])
            liquid_w = trailing_best_weighting(returns_by_weighting, alpha, idx, lookback, [w for w in ["vw", "vw_cap"] if w in returns_by_weighting])
            repaired_all[alpha] = returns_by_weighting[all_w].loc[next_date, alpha]
            repaired_liquid[alpha] = returns_by_weighting[liquid_w].loc[next_date, alpha]
        repaired_all = pd.Series(repaired_all).dropna()
        repaired_liquid = pd.Series(repaired_liquid).dropna()

        rows.append(
            {
                "date": next_date,
                "residual_lifecycle": float((ret_next * w_residual).sum()),
                "blend_lifecycle": float((ret_next * w_blend).sum()),
                "regime_gated": float((ret_next * w_regime).sum()),
                "blend_regime_gated": float((ret_next * w_blend_regime).sum()),
                "liquidity_repaired": float((repaired_liquid * normalized((blend_score * gate).loc[repaired_liquid.index])).sum()),
                "all_actions_repaired": float((repaired_all * normalized((blend_score * gate).loc[repaired_all.index])).sum()),
                "avg_gate": float(gate.mean()),
                "n_alphas": int(len(names)),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(port: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows = []
    strategies = [c for c in RETURN_COLS if c in port.columns]
    for base in baselines:
        for s in strategies:
            if s == base or base not in port:
                continue
            diff = port[s] - port[base]
            rows.append(
                {
                    "strategy": s,
                    "baseline": base,
                    "mean_monthly_diff": float(diff.mean()),
                    "annualized_diff_approx": float(diff.mean() * 12),
                    "tstat": float(tstat_mean(diff)),
                    "win_rate": float((diff > 0).mean()),
                    "n_months": int(diff.dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_extension_portfolios(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in RETURN_COLS if c in port.columns]:
        stats = annualized_stats(port[col])
        stats["strategy"] = col
        stats["monthly_mean"] = float(port[col].mean())
        stats["monthly_tstat"] = float(tstat_mean(port[col]))
        rows.append(stats)
    return pd.DataFrame(rows)[
        ["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "monthly_mean", "monthly_tstat", "n_months"]
    ]


def run(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_by_weighting = {}
    long_by_weighting = {}
    for w in ["vw", "ew", "vw_cap"]:
        if factor_return_path(data_root, w).exists():
            long = load_factor_returns(data_root, w)
            long_by_weighting[w] = long
            returns_by_weighting[w] = pivot_factor_returns(long)
    base = returns_by_weighting["vw"]
    ff6 = load_ff6(out_dir)

    raw_states = build_states(base, long_by_weighting["vw"], args.lookback)
    residual_wide = rolling_ff6_residuals(base, ff6, args.residual_lookback)
    residual_states = build_states(residual_wide, long_by_weighting["vw"], args.lookback)
    gates = rolling_regime_gate(base, ff6, args.regime_lookback)

    # Recreate comparable baseline variants from full script.
    from alphalife_full import factor_allocation_backtest

    base_port = factor_allocation_backtest(raw_states, returns_by_weighting, "vw", args.lookback, OOS_DATE)
    ext_port = make_extended_portfolios(raw_states, residual_states, returns_by_weighting, gates, args.lookback, OOS_DATE)
    port = base_port.merge(ext_port, on="date", how="inner")
    port.to_csv(out_dir / "extended_portfolio_returns.csv", index=False)
    summary = summarize_extension_portfolios(port)
    summary.to_csv(out_dir / "extended_portfolio_summary.csv", index=False)
    tests = paired_tests(port, ["static_equal", "rolling_sharpe", "alphalife_full", "alphalife_repaired"])
    tests.to_csv(out_dir / "extended_pairwise_tests.csv", index=False)

    # Repair variants: liquidity-aware choices only.
    liquid_trials = []
    target = raw_states[(raw_states["date"] >= OOS_DATE) & raw_states["state"].isin(["Warning", "Decayed"])]
    for _, row in target.iterrows():
        date, alpha = row["date"], row["alpha"]
        if date not in base.index or alpha not in base.columns:
            continue
        idx = base.index.get_loc(date)
        if isinstance(idx, slice) or idx < args.lookback or idx + 12 >= len(base.index):
            continue
        bw = trailing_best_weighting(returns_by_weighting, alpha, idx, args.lookback, [w for w in ["vw", "vw_cap"] if w in returns_by_weighting])
        fdates = base.index[idx + 1 : idx + 13]
        base_future = base.loc[fdates, alpha].sum()
        repair_future = returns_by_weighting[bw].loc[fdates, alpha].sum()
        liquid_trials.append(
            {
                "date": date,
                "alpha": alpha,
                "state": row["state"],
                "selected_repair": bw,
                "base_future12_ret": float(base_future),
                "repair_future12_ret": float(repair_future),
                "repair_improvement": float(repair_future - base_future),
                "repair_success": bool(repair_future > base_future and repair_future > 0),
            }
        )
    liquid_trials = pd.DataFrame(liquid_trials)
    liquid_trials.to_csv(out_dir / "liquidity_repair_trials.csv", index=False)
    repair_summary = {
        "n": int(len(liquid_trials)),
        "success_rate": float(liquid_trials["repair_success"].mean()) if len(liquid_trials) else np.nan,
        "mean_improvement": float(liquid_trials["repair_improvement"].mean()) if len(liquid_trials) else np.nan,
        "median_improvement": float(liquid_trials["repair_improvement"].median()) if len(liquid_trials) else np.nan,
        "repair_counts": liquid_trials["selected_repair"].value_counts().to_dict() if len(liquid_trials) else {},
    }
    (out_dir / "liquidity_repair_summary.json").write_text(json.dumps(repair_summary, indent=2))

    # Stock-level liquidity filter already tests market-equity filters.
    stock_ic = stock_level_ic_multi(data_root, out_dir, RANK_FEATURES[: args.stock_features], [0.0, 0.2, 0.5, 0.7])

    try:
        import matplotlib.pyplot as plt

        cols = [
            "static_equal",
            "rolling_sharpe",
            "alphalife_full",
            "alphalife_repaired",
            "residual_lifecycle",
            "blend_regime_gated",
            "liquidity_repaired",
            "all_actions_repaired",
        ]
        wealth = (1 + port.set_index("date")[cols]).cumprod()
        ax = wealth.plot(figsize=(13, 7), title="AlphaLife Extensions: Residualization, Liquidity Repair, Regime Gates")
        ax.set_ylabel("Growth of $1")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_extension_wealth.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(str(exc))

    notes = f"""# AlphaLife Extension Results

This run adds three interpretable lifecycle actions:

1. FF6 residual health scoring.
2. Market-regime gates based on trailing market volatility and 12-month market return.
3. Liquidity-aware repair that restricts repair choices to VW and VW-cap factor returns.

Main output files:

- `extended_portfolio_summary.csv`
- `extended_pairwise_tests.csv`
- `liquidity_repair_summary.json`
- `stock_level_ic_summary_by_filter.csv`
- `fig_extension_wealth.png`

Best Sharpe strategy:

{summary.sort_values('sharpe', ascending=False).head(5).to_markdown(index=False)}

Liquidity-aware repair summary:

```json
{json.dumps(repair_summary, indent=2)}
```
"""
    (out_dir / "extension_results_summary.md").write_text(notes)

    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--out-dir", default="outputs/alphalife_extensions")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--residual-lookback", type=int, default=120)
    p.add_argument("--regime-lookback", type=int, default=120)
    p.add_argument("--stock-features", type=int, default=len(RANK_FEATURES))
    return p.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
