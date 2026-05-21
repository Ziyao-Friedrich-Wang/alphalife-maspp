#!/usr/bin/env python3
"""AlphaLife-MAS MVP experiments for monthly U.S. factor data.

This script is intentionally conservative:
- It treats JKP US monthly factor returns as tradable Alpha return series.
- It treats an optional U.S. stock-month panel as the stock-level validation panel.
- It uses only local files and writes no credentials.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(os.getenv("ALPHALIFE_DATA_ROOT", "data"))
MIN_DATE = "1963-01-31"
OOS_DATE = "1990-01-31"
MAX_DATE = "2024-12-31"

RANK_FEATURES = [
    "rank_ret_1_0",
    "rank_ret_3_1",
    "rank_ret_6_1",
    "rank_ret_9_1",
    "rank_ret_12_1",
    "rank_ret_12_7",
    "rank_ret_60_12",
    "rank_be_me",
    "rank_at_me",
    "rank_sale_me",
    "rank_debt_me",
    "rank_gp_at",
    "rank_op_at",
    "rank_ni_at",
    "rank_ni_be",
    "rank_cash_at",
    "rank_debt_at",
    "rank_ppe_at",
    "rank_capx_at",
    "rank_rd_sale",
    "rank_rd_at",
    "rank_at_gr1",
    "rank_sale_gr1",
    "rank_inv_gr1",
    "rank_rect_gr1",
    "rank_accruals_at",
    "rank_working_capital_at",
    "rank_turnover_12m",
    "rank_dolvol_12m",
    "rank_ami_12m",
    "rank_zero_volume_12m",
    "rank_beta_60m",
    "rank_rvol_12m",
    "rank_maxret_12m",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_zipped_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one file in {path}, found {names}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh)


def factor_return_path(data_root: Path, weighting: str, name: str = "all_factors") -> Path:
    return (
        data_root
        / "factor_returns"
        / "all_stocks"
        / "usa"
        / "monthly"
        / weighting
        / f"[usa]_[{name}]_[monthly]_[{weighting}].zip"
    )


def load_factor_returns(data_root: Path, weighting: str) -> pd.DataFrame:
    df = read_zipped_csv(factor_return_path(data_root, weighting))
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= MIN_DATE) & (df["date"] <= MAX_DATE)].copy()
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ret", "name", "date"])
    return df


def pivot_factor_returns(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="date", columns="name", values="ret", aggfunc="mean").sort_index()
    wide = wide.loc[:, sorted(wide.columns)]
    return wide


def drawdown_from_returns(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    wealth = (1.0 + ret.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def max_drawdown(ret: pd.Series) -> float:
    return drawdown_from_returns(ret)


def annualized_stats(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "n_months": 0,
        }
    mean_m = ret.mean()
    vol_m = ret.std(ddof=1)
    ann_return = (1.0 + mean_m) ** 12 - 1.0
    ann_vol = vol_m * math.sqrt(12) if pd.notna(vol_m) else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(ret),
        "hit_rate": float((ret > 0).mean()),
        "n_months": int(ret.shape[0]),
    }


def rolling_factor_metrics(wide: pd.DataFrame, long_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    means = wide.rolling(lookback, min_periods=max(24, lookback // 2)).mean()
    vols = wide.rolling(lookback, min_periods=max(24, lookback // 2)).std()
    sharpes = means / vols * math.sqrt(12)
    tstats = means / (vols / np.sqrt(lookback))
    ret12 = wide.rolling(12, min_periods=8).sum()
    ret24 = wide.rolling(24, min_periods=12).sum()
    ret60 = wide.rolling(lookback, min_periods=24).sum()

    rows: list[pd.DataFrame] = []
    for name in wide.columns:
        tmp = pd.DataFrame(
            {
                "date": wide.index,
                "alpha": name,
                "mean60": means[name].to_numpy(),
                "vol60": vols[name].to_numpy(),
                "sharpe60": sharpes[name].to_numpy(),
                "tstat60": tstats[name].to_numpy(),
                "ret12": ret12[name].to_numpy(),
                "ret24": ret24[name].to_numpy(),
                "ret60": ret60[name].to_numpy(),
            }
        )
        rows.append(tmp)
    metrics = pd.concat(rows, ignore_index=True)

    nstocks = (
        long_df[["date", "name", "n_stocks"]]
        .rename(columns={"name": "alpha"})
        .drop_duplicates(["date", "alpha"])
    )
    metrics = metrics.merge(nstocks, on=["date", "alpha"], how="left")
    return metrics


def compute_redundancy(wide: pd.DataFrame, lookback: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dates = wide.index
    cols = list(wide.columns)
    for idx in range(lookback, len(dates)):
        date = dates[idx]
        win = wide.iloc[idx - lookback : idx]
        corr = win.corr(min_periods=max(24, lookback // 2)).abs()
        corr_values = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_values, np.nan)
        corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
        maxcorr = corr.max(axis=1)
        for alpha in cols:
            records.append(
                {
                    "date": date,
                    "alpha": alpha,
                    "max_abs_corr60": float(maxcorr.get(alpha, np.nan)),
                }
            )
    return pd.DataFrame.from_records(records)


def zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        med = out.groupby("date")[col].transform("median")
        std = out.groupby("date")[col].transform("std")
        out[f"z_{col}"] = (out[col] - med) / std.replace(0.0, np.nan)
    return out


def assign_states(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df = zscore_by_date(df, ["sharpe60", "tstat60", "ret12", "ret24", "max_abs_corr60"])
    df["health_score"] = (
        0.40 * df["z_sharpe60"].fillna(0.0)
        + 0.25 * df["z_tstat60"].fillna(0.0)
        + 0.25 * df["z_ret12"].fillna(0.0)
        + 0.10 * df["z_ret24"].fillna(0.0)
        - 0.25 * df["z_max_abs_corr60"].fillna(0.0)
    )
    by_date = df.groupby("date")["health_score"]
    q25 = by_date.transform(lambda x: x.quantile(0.25))
    q10 = by_date.transform(lambda x: x.quantile(0.10))
    df["state"] = np.select(
        [
            (df["health_score"] <= q10) | ((df["ret24"] < 0) & (df["sharpe60"] < 0)),
            (df["health_score"] <= q25) | ((df["ret12"] < 0) & (df["sharpe60"] < 0.15)),
        ],
        ["Decayed", "Warning"],
        default="Healthy",
    )
    return df.sort_values(["date", "alpha"])


def add_future_returns(state_df: pd.DataFrame, wide: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    future = wide.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    rows = []
    for alpha in wide.columns:
        tmp = pd.DataFrame({"date": future.index, "alpha": alpha, f"future{horizon}_ret": future[alpha].to_numpy()})
        rows.append(tmp)
    f = pd.concat(rows, ignore_index=True)
    return state_df.merge(f, on=["date", "alpha"], how="left")


def warning_effectiveness(state_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oos = state_df[state_df["date"] >= OOS_DATE].copy()
    for state, group in oos.groupby("state"):
        fut = group["future12_ret"].dropna()
        rows.append(
            {
                "state": state,
                "n": int(fut.shape[0]),
                "mean_future12_ret": float(fut.mean()) if not fut.empty else np.nan,
                "median_future12_ret": float(fut.median()) if not fut.empty else np.nan,
                "future12_negative_rate": float((fut < 0).mean()) if not fut.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("state")


def portfolio_returns(state_df: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(wide.index).intersection(set(state_df["date"])))
    records = []
    state_df = state_df.set_index(["date", "alpha"]).sort_index()
    for date in dates:
        if date < pd.Timestamp(OOS_DATE):
            continue
        if date not in wide.index:
            continue
        alpha_slice = state_df.loc[date].copy() if date in state_df.index.get_level_values(0) else None
        if alpha_slice is None:
            continue
        available = [a for a in wide.columns if a in alpha_slice.index]
        next_idx = wide.index.get_indexer([date])[0] + 1
        if next_idx >= len(wide.index):
            continue
        next_date = wide.index[next_idx]
        ret_next = wide.loc[next_date, available].dropna()
        if ret_next.empty:
            continue

        s = alpha_slice.loc[ret_next.index]
        static_ret = ret_next.mean()

        sharpe = s["sharpe60"].clip(lower=0).fillna(0.0)
        sharpe_w = sharpe / sharpe.sum() if sharpe.sum() > 0 else pd.Series(1.0 / len(ret_next), index=ret_next.index)
        rolling_sharpe_ret = float((ret_next * sharpe_w.loc[ret_next.index]).sum())

        health = s["health_score"].fillna(0.0)
        state_mult = s["state"].map({"Healthy": 1.0, "Warning": 0.35, "Decayed": 0.15, "Retired": 0.0}).fillna(0.0)
        base = np.maximum(health - health.quantile(0.30), 0.0) * state_mult
        if base.sum() <= 0:
            lifecycle_w = pd.Series(1.0 / len(ret_next), index=ret_next.index)
        else:
            lifecycle_w = base / base.sum()
        lifecycle_ret = float((ret_next * lifecycle_w.loc[ret_next.index]).sum())

        records.append(
            {
                "date": next_date,
                "static_equal_weight": float(static_ret),
                "rolling_sharpe_weight": rolling_sharpe_ret,
                "alphalife_lifecycle": lifecycle_ret,
                "active_alpha_count": int((state_mult > 0).sum()),
                "retired_alpha_count": int((s["state"] == "Retired").sum()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("date")


def summarize_portfolios(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["static_equal_weight", "rolling_sharpe_weight", "alphalife_lifecycle"]:
        stats = annualized_stats(port[col])
        stats["strategy"] = col
        rows.append(stats)
    return pd.DataFrame(rows)[["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "n_months"]]


def repair_experiment(
    state_df: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int = 12,
) -> pd.DataFrame:
    base = returns_by_weighting["vw"]
    alternatives = [w for w in ["vw", "ew", "vw_cap"] if w in returns_by_weighting]
    records = []
    state_oos = state_df[(state_df["date"] >= OOS_DATE) & (state_df["state"].isin(["Warning", "Decayed"]))].copy()
    for _, row in state_oos.iterrows():
        date = row["date"]
        alpha = row["alpha"]
        if alpha not in base.columns or date not in base.index:
            continue
        idx = base.index.get_loc(date)
        if isinstance(idx, slice) or idx < lookback or idx + horizon >= len(base.index):
            continue

        best_weighting = "vw"
        best_score = -np.inf
        for w in alternatives:
            alt = returns_by_weighting[w]
            if alpha not in alt.columns:
                continue
            hist = alt[alpha].iloc[idx - lookback : idx].dropna()
            if hist.shape[0] < 24:
                continue
            score = hist.mean() / hist.std(ddof=1) if hist.std(ddof=1) > 0 else -np.inf
            if score > best_score:
                best_score = score
                best_weighting = w

        future_dates = base.index[idx + 1 : idx + 1 + horizon]
        base_future = base.loc[future_dates, alpha].sum()
        repaired_future = returns_by_weighting[best_weighting].loc[future_dates, alpha].sum()
        records.append(
            {
                "date": date,
                "alpha": alpha,
                "state": row["state"],
                "selected_repair": best_weighting,
                "base_future12_ret": float(base_future),
                "repair_future12_ret": float(repaired_future),
                "repair_improvement": float(repaired_future - base_future),
                "repair_success": bool(repaired_future > base_future and repaired_future > 0),
            }
        )
    return pd.DataFrame.from_records(records)


def diagnose_decays(state_df: pd.DataFrame) -> pd.DataFrame:
    df = state_df[state_df["state"].isin(["Warning", "Decayed", "Retired"])].copy()
    if df.empty:
        return pd.DataFrame()

    def probs(row: pd.Series) -> dict[str, float]:
        scores = {
            "performance_decay": max(0.0, -float(row.get("sharpe60", 0.0) or 0.0)) + max(0.0, -float(row.get("ret24", 0.0) or 0.0)),
            "redundancy": max(0.0, float(row.get("max_abs_corr60", 0.0) or 0.0) - 0.55),
            "coverage_decay": max(0.0, 500.0 - float(row.get("n_stocks", 500.0) or 500.0)) / 500.0,
            "recent_drawdown": max(0.0, -float(row.get("ret12", 0.0) or 0.0)),
            "structural_decay": 0.25,
        }
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()} if total > 0 else {k: 1 / len(scores) for k in scores}

    latest = df.sort_values("date").groupby("alpha").tail(1).copy()
    for key in ["performance_decay", "redundancy", "coverage_decay", "recent_drawdown", "structural_decay"]:
        latest[key] = latest.apply(lambda r, key=key: probs(r)[key], axis=1)
    keep = [
        "date",
        "alpha",
        "state",
        "health_score",
        "sharpe60",
        "ret12",
        "ret24",
        "max_abs_corr60",
        "n_stocks",
        "performance_decay",
        "redundancy",
        "coverage_decay",
        "recent_drawdown",
        "structural_decay",
    ]
    return latest[keep].sort_values(["state", "health_score"]).head(80)


def stock_level_ic(data_root: Path, out_dir: Path, max_features: int) -> pd.DataFrame:
    parquet_dir = data_root / "stock_level_us" / "panel_parquet"
    files = sorted(parquet_dir.glob("us_stock_month_panel_*.parquet"))
    selected = RANK_FEATURES[:max_features]
    records = []
    for path in files:
        year = int(re.search(r"_(\d{4})\.parquet$", path.name).group(1))
        if year < 1990 or year > 2024:
            continue
        cols = ["date", "permno", "ret_excess_lead1m", "rank_market_equity", *selected]
        frame = pd.read_parquet(path, columns=[c for c in cols if c])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.dropna(subset=["ret_excess_lead1m"])
        liquid = frame[frame["rank_market_equity"] >= 0.2]
        for date, g in liquid.groupby("date", sort=False):
            y = g["ret_excess_lead1m"]
            for feat in selected:
                if feat not in g.columns:
                    continue
                x = g[feat]
                valid = x.notna() & y.notna()
                if valid.sum() < 100:
                    continue
                ic = x[valid].corr(y[valid])
                top = g.loc[valid & (x >= 0.8), "ret_excess_lead1m"].mean()
                bot = g.loc[valid & (x <= 0.2), "ret_excess_lead1m"].mean()
                records.append(
                    {
                        "date": date,
                        "feature": feat.replace("rank_", ""),
                        "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                        "top_minus_bottom": float(top - bot) if pd.notna(top) and pd.notna(bot) else np.nan,
                        "n": int(valid.sum()),
                    }
                )
    ic = pd.DataFrame.from_records(records)
    if not ic.empty:
        summary = (
            ic.groupby("feature")
            .agg(
                mean_rank_ic=("rank_ic", "mean"),
                rank_ic_std=("rank_ic", "std"),
                mean_top_minus_bottom=("top_minus_bottom", "mean"),
                n_months=("date", "nunique"),
                avg_n=("n", "mean"),
            )
            .reset_index()
        )
        summary["icir"] = summary["mean_rank_ic"] / summary["rank_ic_std"]
        summary.to_csv(out_dir / "stock_level_ic_summary.csv", index=False)
    ic.to_csv(out_dir / "stock_level_ic_monthly.csv", index=False)
    return ic


def plot_outputs(port: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        plot = port.copy()
        plot = plot.set_index("date")
        wealth = (1.0 + plot[["static_equal_weight", "rolling_sharpe_weight", "alphalife_lifecycle"]]).cumprod()
        ax = wealth.plot(figsize=(10, 6), title="OOS Factor Allocation Wealth")
        ax.set_ylabel("Growth of $1")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / "portfolio_wealth.png", dpi=160)
        plt.close(fig)

        dd = wealth / wealth.cummax() - 1.0
        ax = dd.plot(figsize=(10, 6), title="OOS Drawdowns")
        ax.set_ylabel("Drawdown")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out_dir / "portfolio_drawdowns.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # plotting should not fail the experiment
        (out_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S"))

    metadata = {
        "data_root": str(data_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "min_date": MIN_DATE,
        "oos_date": OOS_DATE,
        "max_date": MAX_DATE,
        "lookback_months": args.lookback,
        "stock_features": args.stock_features,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    returns_by_weighting: dict[str, pd.DataFrame] = {}
    long_by_weighting: dict[str, pd.DataFrame] = {}
    for weighting in ["vw", "ew", "vw_cap"]:
        path = factor_return_path(data_root, weighting)
        if path.exists():
            long = load_factor_returns(data_root, weighting)
            long_by_weighting[weighting] = long
            returns_by_weighting[weighting] = pivot_factor_returns(long)

    if "vw" not in returns_by_weighting:
        raise FileNotFoundError("Missing required USA monthly VW all_factors file.")

    wide = returns_by_weighting["vw"]
    long = long_by_weighting["vw"]
    metrics = rolling_factor_metrics(wide, long, args.lookback)
    redundancy = compute_redundancy(wide, args.lookback)
    metrics = metrics.merge(redundancy, on=["date", "alpha"], how="left")
    states = assign_states(metrics)
    states = add_future_returns(states, wide, 12)
    states.to_parquet(out_dir / "factor_lifecycle_states.parquet", index=False)

    warning = warning_effectiveness(states)
    warning.to_csv(out_dir / "warning_effectiveness.csv", index=False)

    port = portfolio_returns(states, wide)
    port.to_csv(out_dir / "portfolio_returns.csv", index=False)
    portfolio_summary = summarize_portfolios(port)
    portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    plot_outputs(port, out_dir)

    repairs = repair_experiment(states, returns_by_weighting, args.lookback)
    repairs.to_csv(out_dir / "repair_experiment.csv", index=False)
    repair_summary = {}
    if not repairs.empty:
        repair_summary = {
            "n": int(repairs.shape[0]),
            "success_rate": float(repairs["repair_success"].mean()),
            "mean_improvement": float(repairs["repair_improvement"].mean()),
            "median_improvement": float(repairs["repair_improvement"].median()),
            "repair_counts": repairs["selected_repair"].value_counts().to_dict(),
        }
    (out_dir / "repair_summary.json").write_text(json.dumps(repair_summary, indent=2), encoding="utf-8")

    diagnostics = diagnose_decays(states)
    diagnostics.to_csv(out_dir / "decay_diagnostics.csv", index=False)

    if args.stock_ic:
        stock_level_ic(data_root, out_dir, args.stock_features)

    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default="outputs/alphalife_mvp")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--stock-features", type=int, default=24)
    parser.add_argument("--stock-ic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(output)
