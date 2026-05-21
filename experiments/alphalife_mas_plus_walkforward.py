#!/usr/bin/env python3
"""Walk-forward validation for AlphaLife-MAS++ meta-policies.

This script does not retrain Q++. It consumes the monthly returns from
alphalife_mas_plus.py and evaluates whether the stronger MAS++ policies survive
parameter selection using only past validation windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alphalife_mas import downside_metrics  # noqa: E402
from alphalife_mvp import annualized_stats  # noqa: E402
from alphalife_strong import apply_costs, newey_west_alpha  # noqa: E402


COMPONENTS = ["qpp_single_agent", "full_mas", "centralized_constrained_q", "inverse_vol", "single_agent_q"]


def utility_score(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 24:
        return -np.inf
    st = annualized_stats(ret)
    ds = downside_metrics(ret)
    if not np.isfinite(st["sharpe"]):
        return -np.inf
    # Conservative validation criterion: keep Sharpe primary but charge left-tail
    # losses and realized drawdown using only past returns.
    return float(st["sharpe"] + 1.5 * st["max_drawdown"] - 4.0 * abs(ds["es_5"]))


def sharpe_score(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 24:
        return -np.inf
    st = annualized_stats(ret)
    return float(st["sharpe"]) if np.isfinite(st["sharpe"]) else -np.inf


def component_returns(port: pd.DataFrame, turnover_bps: float, switch_bps: float) -> dict[str, pd.Series]:
    df = port.reset_index()
    out: dict[str, pd.Series] = {}
    for strategy in COMPONENTS:
        out[strategy] = pd.Series(apply_costs(df, strategy, turnover_bps, switch_bps).to_numpy(), index=port.index)
    return out


def dd_gate_weights(q: pd.Series, dd_thr: float, ret_thr: float, q_high: float, q_low: float) -> pd.Series:
    weights = []
    wealth = (1.0 + q.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    for i, _ in enumerate(q.index):
        hist_dd = dd.iloc[max(0, i - 6) : i].min() if i > 0 else 0.0
        hist_ret = q.iloc[max(0, i - 3) : i].sum() if i > 0 else 0.0
        weights.append(q_low if (hist_dd < dd_thr or hist_ret < ret_thr) else q_high)
    return pd.Series(weights, index=q.index)


def make_dd_candidates(src_net: dict[str, pd.Series], defensive: str) -> dict[str, pd.Series]:
    candidates: dict[str, pd.Series] = {}
    q = src_net["qpp_single_agent"]
    for dd_thr in [-0.03, -0.04, -0.05, -0.06, -0.08]:
        for ret_thr in [-0.025, -0.035, -0.050]:
            for q_high in [0.75, 0.85, 0.95]:
                for q_low in [0.35, 0.45, 0.55]:
                    w = dd_gate_weights(q, dd_thr, ret_thr, q_high, q_low)
                    key = f"dd|def={defensive}|dd={dd_thr}|ret={ret_thr}|hi={q_high}|lo={q_low}"
                    candidates[key] = w * src_net["qpp_single_agent"] + (1.0 - w) * src_net[defensive]
    return candidates


def make_dd_series_with_key(src: dict[str, pd.Series], key: str) -> pd.Series:
    parts = dict(item.split("=", 1) for item in key.split("|")[1:])
    defensive = parts["def"]
    dd_thr = float(parts["dd"])
    ret_thr = float(parts["ret"])
    q_high = float(parts["hi"])
    q_low = float(parts["lo"])
    w = dd_gate_weights(src["qpp_single_agent"], dd_thr, ret_thr, q_high, q_low)
    return w * src["qpp_single_agent"] + (1.0 - w) * src[defensive]


def trailing_meta_weights(src_net: dict[str, pd.Series], lookback: int, eta: float, q_cap: float) -> pd.DataFrame:
    cands = ["qpp_single_agent", "centralized_constrained_q", "full_mas", "inverse_vol"]
    idx = src_net["qpp_single_agent"].index
    rows = []
    for i, date in enumerate(idx):
        if i < lookback:
            w = pd.Series(1.0 / len(cands), index=cands)
        else:
            vals = []
            for cand in cands:
                vals.append(utility_score(src_net[cand].iloc[i - lookback : i]))
            u = pd.Series(vals, index=cands).replace([np.inf, -np.inf], np.nan).fillna(-5.0)
            x = np.exp(np.clip(eta * (u - u.max()), -30, 0))
            w = x / x.sum()
            if w["qpp_single_agent"] > q_cap:
                excess = w["qpp_single_agent"] - q_cap
                w["qpp_single_agent"] = q_cap
                w["full_mas"] += excess
            w = w / w.sum()
        rows.append({"date": date, **w.to_dict()})
    return pd.DataFrame(rows).set_index("date")


def make_meta_candidates(src_net: dict[str, pd.Series]) -> dict[str, pd.Series]:
    candidates: dict[str, pd.Series] = {}
    for lookback in [24, 36, 60, 120]:
        for eta in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            for q_cap in [0.65, 0.75, 0.82, 0.90]:
                w = trailing_meta_weights(src_net, lookback, eta, q_cap)
                key = f"meta_lb{lookback}_eta{eta}_qcap{q_cap}"
                candidates[key] = sum(w[c] * src_net[c] for c in w.columns)
    return candidates


def make_meta_series_with_key(src: dict[str, pd.Series], src_net_for_weights: dict[str, pd.Series], key: str) -> pd.Series:
    parts = key.split("_")
    lookback = int(parts[1].replace("lb", ""))
    eta = float(parts[2].replace("eta", ""))
    q_cap = float(parts[3].replace("qcap", ""))
    w = trailing_meta_weights(src_net_for_weights, lookback, eta, q_cap)
    return sum(w[c] * src[c] for c in w.columns)


def walk_forward_select(
    candidates_net: dict[str, pd.Series],
    validation_months: int,
    min_validation: int,
    score_fn,
) -> tuple[pd.Series, pd.Series]:
    keys = sorted(candidates_net)
    idx = next(iter(candidates_net.values())).index
    chosen_returns = []
    chosen_keys = []
    for i, date in enumerate(idx):
        if i < min_validation:
            # Before enough history exists, use the least aggressive robust
            # candidate by validation convention: median key after sorting.
            key = keys[len(keys) // 2]
        else:
            lo = max(0, i - validation_months)
            best_key, best_score = keys[0], -np.inf
            for key in keys:
                score = score_fn(candidates_net[key].iloc[lo:i])
                if score > best_score:
                    best_key, best_score = key, score
            key = best_key
        chosen_keys.append(key)
        chosen_returns.append(float(candidates_net[key].loc[date]))
    return pd.Series(chosen_returns, index=idx), pd.Series(chosen_keys, index=idx)


def apply_chosen_keys_dd(src: dict[str, pd.Series], chosen_keys: pd.Series) -> pd.Series:
    cache: dict[str, pd.Series] = {}
    vals = []
    for date, key in chosen_keys.items():
        if key not in cache:
            cache[key] = make_dd_series_with_key(src, key)
        vals.append(float(cache[key].loc[date]))
    return pd.Series(vals, index=chosen_keys.index)


def apply_chosen_keys_meta(src: dict[str, pd.Series], src_net_for_weights: dict[str, pd.Series], chosen_keys: pd.Series) -> pd.Series:
    cache: dict[str, pd.Series] = {}
    vals = []
    for date, key in chosen_keys.items():
        if key not in cache:
            cache[key] = make_meta_series_with_key(src, src_net_for_weights, key)
        vals.append(float(cache[key].loc[date]))
    return pd.Series(vals, index=chosen_keys.index)


def summarize(name: str, ret: pd.Series, stress: pd.Series, extreme: pd.Series, ff6: pd.DataFrame) -> dict[str, float | str | int]:
    st = annualized_stats(ret)
    ds = downside_metrics(ret)
    ff = newey_west_alpha(ret, ff6)
    return {
        "strategy": name,
        "net_sharpe": st["sharpe"],
        "stress_sharpe": annualized_stats(stress)["sharpe"],
        "extreme_sharpe": annualized_stats(extreme)["sharpe"],
        "ann_return": st["ann_return"],
        "ann_vol": st["ann_vol"],
        "max_drawdown": st["max_drawdown"],
        "calmar": ds["calmar"],
        "es_5": ds["es_5"],
        "hit_rate": st["hit_rate"],
        "n_months": st["n_months"],
        **ff,
    }


def subperiod_summary(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    periods = [
        ("1990-2004", "1990-02-28", "2004-12-31"),
        ("2005-2014", "2005-01-31", "2014-12-31"),
        ("2015-2024", "2015-01-31", "2024-12-31"),
        ("2020-2024", "2020-01-31", "2024-12-31"),
    ]
    rows = []
    for period, start, end in periods:
        for name, ret in series_map.items():
            sub = ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]
            rows.append({"period": period, "strategy": name, **annualized_stats(sub), **downside_metrics(sub)})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir).expanduser()
    port = pd.read_csv(input_dir / "maspp_merged_portfolio_returns.csv", parse_dates=["date"]).set_index("date")
    ff6 = pd.read_csv(input_dir / "external_ff6_monthly.csv", parse_dates=["date"])

    src_net = component_returns(port, 10.0, 2.0)
    src_stress = component_returns(port, 25.0, 5.0)
    src_extreme = component_returns(port, 50.0, 10.0)

    dd_net = make_dd_candidates(src_net, "full_mas")
    meta_net = make_meta_candidates(src_net)

    rows = []
    returns = pd.DataFrame({"date": port.index})
    choices = []
    dd_utility_keys: pd.Series | None = None
    meta_utility_keys: pd.Series | None = None
    series_for_subperiod = {
        "single_agent_q": src_net["single_agent_q"],
        "qpp_single_agent": src_net["qpp_single_agent"],
        "full_mas": src_net["full_mas"],
        "centralized_constrained_q": src_net["centralized_constrained_q"],
    }

    for score_name, score_fn in [("sharpe", sharpe_score), ("utility", utility_score)]:
        dd_ret, dd_keys = walk_forward_select(dd_net, args.validation_months, args.min_validation, score_fn)
        dd_stress = apply_chosen_keys_dd(src_stress, dd_keys)
        dd_extreme = apply_chosen_keys_dd(src_extreme, dd_keys)
        name = f"wf_dd_full_{score_name}"
        rows.append(summarize(name, dd_ret, dd_stress, dd_extreme, ff6))
        returns[name] = dd_ret.to_numpy()
        returns[name + "_stress"] = dd_stress.to_numpy()
        choices.extend({"date": d, "policy": name, "chosen_key": k} for d, k in dd_keys.items())
        series_for_subperiod[name] = dd_ret
        if score_name == "utility":
            dd_utility_keys = dd_keys

        meta_ret, meta_keys = walk_forward_select(meta_net, args.validation_months, args.min_validation, score_fn)
        meta_stress = apply_chosen_keys_meta(src_stress, src_net, meta_keys)
        meta_extreme = apply_chosen_keys_meta(src_extreme, src_net, meta_keys)
        name = f"wf_meta_{score_name}"
        rows.append(summarize(name, meta_ret, meta_stress, meta_extreme, ff6))
        returns[name] = meta_ret.to_numpy()
        returns[name + "_stress"] = meta_stress.to_numpy()
        choices.extend({"date": d, "policy": name, "chosen_key": k} for d, k in meta_keys.items())
        series_for_subperiod[name] = meta_ret
        if score_name == "utility":
            meta_utility_keys = meta_keys

    # Conservative combined policy: select among the two walk-forward families
    # using only past validation performance.
    combo_candidates = {
        "wf_dd_full_utility": pd.Series(returns["wf_dd_full_utility"].to_numpy(), index=port.index),
        "wf_meta_utility": pd.Series(returns["wf_meta_utility"].to_numpy(), index=port.index),
        "qpp_single_agent": src_net["qpp_single_agent"],
        "full_mas": src_net["full_mas"],
    }
    combo_ret, combo_keys = walk_forward_select(combo_candidates, args.validation_months, args.min_validation, utility_score)
    combo_stress_map = {
        "wf_dd_full_utility": pd.Series(returns["wf_dd_full_utility_stress"].to_numpy(), index=port.index),
        "wf_meta_utility": pd.Series(returns["wf_meta_utility_stress"].to_numpy(), index=port.index),
        "qpp_single_agent": src_stress["qpp_single_agent"],
        "full_mas": src_stress["full_mas"],
    }
    if dd_utility_keys is None or meta_utility_keys is None:
        raise RuntimeError("Utility walk-forward keys were not generated.")
    combo_extreme_map = {
        "wf_dd_full_utility": apply_chosen_keys_dd(src_extreme, dd_utility_keys),
        "wf_meta_utility": apply_chosen_keys_meta(src_extreme, src_net, meta_utility_keys),
        "qpp_single_agent": src_extreme["qpp_single_agent"],
        "full_mas": src_extreme["full_mas"],
    }
    combo_stress = pd.Series([combo_stress_map[k].loc[d] for d, k in combo_keys.items()], index=port.index)
    combo_extreme = pd.Series([combo_extreme_map[k].loc[d] for d, k in combo_keys.items()], index=port.index)
    rows.append(summarize("wf_combo_utility", combo_ret, combo_stress, combo_extreme, ff6))
    returns["wf_combo_utility"] = combo_ret.to_numpy()
    choices.extend({"date": d, "policy": "wf_combo_utility", "chosen_key": k} for d, k in combo_keys.items())
    series_for_subperiod["wf_combo_utility"] = combo_ret

    # Baselines in the same table.
    for baseline in ["single_agent_q", "qpp_single_agent", "full_mas", "centralized_constrained_q", "inverse_vol"]:
        rows.append(summarize(baseline, src_net[baseline], src_stress[baseline], src_extreme[baseline], ff6))

    summary = pd.DataFrame(rows)
    summary.to_csv(input_dir / "maspp_walkforward_meta_summary.csv", index=False)
    returns.to_csv(input_dir / "maspp_walkforward_meta_returns.csv", index=False)
    pd.DataFrame(choices).to_csv(input_dir / "maspp_walkforward_meta_choices.csv", index=False)
    sub = subperiod_summary(series_for_subperiod)
    sub.to_csv(input_dir / "maspp_walkforward_meta_subperiod.csv", index=False)

    meta = {
        "input_dir": str(input_dir),
        "validation_months": args.validation_months,
        "min_validation": args.min_validation,
        "dd_candidate_count": len(dd_net),
        "meta_candidate_count": len(meta_net),
    }
    (input_dir / "maspp_walkforward_meta_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return input_dir


def component_returns(port: pd.DataFrame, turnover_bps: float, switch_bps: float) -> dict[str, pd.Series]:
    df = port.reset_index()
    return {name: pd.Series(apply_costs(df, name, turnover_bps, switch_bps).to_numpy(), index=port.index) for name in COMPONENTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/alphalife_mas_plus/20260520_171841")
    parser.add_argument("--validation-months", type=int, default=120)
    parser.add_argument("--min-validation", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
