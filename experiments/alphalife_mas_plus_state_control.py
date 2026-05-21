#!/usr/bin/env python3
"""State-contingent AlphaLife-MAS++ control experiments.

This script evaluates whether the strong MAS++ result is more than a static
combination of a return-seeking Repair++ leg and a defensive governance leg.
It treats the combination coefficient as the Governor's auditable repair-risk
capacity, then tests fixed, global dynamic, cluster dynamic, and online
reliability-adjusted budgets using only past validation evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alphalife_extensions import rolling_ff6_residuals, rolling_regime_gate  # noqa: E402
from alphalife_full import build_states  # noqa: E402
from alphalife_mas import STATE_MULT as GOV_STATE_MULT  # noqa: E402
from alphalife_mas import downside_metrics  # noqa: E402
from alphalife_mas_plus import STATE_MULT_PLUS, action_ew_share, choose_actions_for_date, make_action_returns  # noqa: E402
from alphalife_mas_plus_walkforward import dd_gate_weights, utility_score, walk_forward_select  # noqa: E402
from alphalife_mvp import DEFAULT_DATA_ROOT, OOS_DATE, annualized_stats, factor_return_path, load_factor_returns, pivot_factor_returns  # noqa: E402
from alphalife_strong import apply_costs, calc_switch_rate, calc_turnover, newey_west_alpha, normalized, returns_for_actions  # noqa: E402


BUDGET_GRID = [0.35, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


@dataclass
class SleeveRecord:
    date: pd.Timestamp
    q_sleeve: pd.Series
    gov_sleeve: pd.Series
    q_cluster_ret: dict[str, float]
    gov_cluster_ret: dict[str, float]
    avg_disagreement: float
    veto_rate: float
    low_coverage: set[str]


def load_weighting_data(data_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    returns_by_weighting: dict[str, pd.DataFrame] = {}
    long_by_weighting: dict[str, pd.DataFrame] = {}
    for weighting in ["vw", "ew", "vw_cap"]:
        if factor_return_path(data_root, weighting).exists():
            long = load_factor_returns(data_root, weighting)
            long_by_weighting[weighting] = long
            returns_by_weighting[weighting] = pivot_factor_returns(long)
    if "vw" not in returns_by_weighting:
        raise FileNotFoundError(f"Missing VW factor returns under {data_root}")
    return returns_by_weighting, long_by_weighting


def build_static_clusters(base: pd.DataFrame, n_clusters: int, oos_date: str) -> dict[str, str]:
    from sklearn.cluster import AgglomerativeClustering

    train = base.loc[base.index < pd.Timestamp(oos_date)].iloc[-240:]
    corr = train.corr(min_periods=60).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = corr.clip(-0.99, 0.99)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr.to_numpy())))
    np.fill_diagonal(dist, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    labels = model.fit_predict(dist)
    return {alpha: f"C{int(label):02d}" for alpha, label in zip(corr.columns, labels)}


def sleeve_from_weights_actions(weights: pd.Series, actions: pd.Series) -> pd.Series:
    common = weights.dropna().index.intersection(actions.dropna().index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    rows = [(alpha, str(actions.loc[alpha]), float(weights.loc[alpha])) for alpha in common if float(weights.loc[alpha]) > 0]
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.MultiIndex.from_tuples([(a, act) for a, act, _ in rows], names=["alpha", "action"])
    return pd.Series([w for _, _, w in rows], index=idx).groupby(level=[0, 1]).sum()


def normalize_sleeve(sleeve: pd.Series) -> pd.Series:
    sleeve = sleeve.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(sleeve.sum())
    if total <= 0 or sleeve.empty:
        return sleeve
    return sleeve / total


def combine_sleeves(q: pd.Series, gov: pd.Series, budget_by_cluster: dict[str, float], cluster_map: dict[str, str]) -> pd.Series:
    pieces = []
    if not q.empty:
        q_scale = pd.Series(q.index.get_level_values("alpha"), index=q.index).map(cluster_map).map(budget_by_cluster).fillna(0.70)
        pieces.append(q * q_scale.to_numpy(dtype=float))
    if not gov.empty:
        g_scale = 1.0 - pd.Series(gov.index.get_level_values("alpha"), index=gov.index).map(cluster_map).map(budget_by_cluster).fillna(0.70)
        pieces.append(gov * g_scale.to_numpy(dtype=float))
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces).groupby(level=[0, 1]).sum()
    return normalize_sleeve(out)


def sleeve_return(sleeve: pd.Series, returns_by_action: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    vals = []
    weights = []
    for (alpha, action), weight in sleeve.items():
        mat = returns_by_action.get(str(action))
        if mat is None or alpha not in mat.columns or date not in mat.index:
            continue
        val = mat.loc[date, alpha]
        if np.isfinite(val):
            vals.append(float(val))
            weights.append(float(weight))
    if not vals:
        return 0.0
    return float(np.dot(np.asarray(vals), np.asarray(weights)))


def dominant_actions(sleeve: pd.Series) -> pd.Series:
    if sleeve.empty:
        return pd.Series(dtype=object)
    df = sleeve.rename("weight").reset_index()
    idx = df.sort_values(["alpha", "weight"]).groupby("alpha").tail(1)
    return idx.set_index("alpha")["action"].astype(object)


def sleeve_exposure(sleeve: pd.Series, low_coverage: set[str] | None = None) -> dict[str, float]:
    if sleeve.empty:
        return {"ew_exposure": 0.0, "full_ew_weight": 0.0, "low_coverage_ew_weight": 0.0}
    low_coverage = low_coverage or set()
    ew = []
    full = []
    low = []
    for (alpha, action), weight in sleeve.items():
        if str(alpha).startswith("__"):
            continue
        share = float(action_ew_share(str(action)))
        w = float(weight)
        ew.append(w * share)
        full.append(w if share > 0.95 else 0.0)
        low.append(w * share if alpha in low_coverage else 0.0)
    return {
        "ew_exposure": float(np.sum(ew)),
        "full_ew_weight": float(np.sum(full)),
        "low_coverage_ew_weight": float(np.sum(low)),
    }


def sleeve_turnover(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None or previous.empty:
        return 0.0
    idx = current.index.union(previous.index)
    return float(0.5 * (current.reindex(idx).fillna(0.0) - previous.reindex(idx).fillna(0.0)).abs().sum())


def cluster_returns(sleeve: pd.Series, returns_by_action: dict[str, pd.DataFrame], date: pd.Timestamp, cluster_map: dict[str, str]) -> dict[str, float]:
    rows = []
    for (alpha, action), weight in sleeve.items():
        mat = returns_by_action.get(str(action))
        if mat is None or alpha not in mat.columns or date not in mat.index:
            continue
        ret = mat.loc[date, alpha]
        if np.isfinite(ret):
            rows.append((cluster_map.get(alpha, "CXX"), float(weight), float(ret)))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["cluster", "weight", "ret"])
    out = {}
    for cluster, g in df.groupby("cluster"):
        total = float(g["weight"].sum())
        out[cluster] = float((g["weight"] * g["ret"]).sum() / total) if total > 0 else 0.0
    return out


def qpp_sleeve_for_month(
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    names: pd.Index,
    states: pd.DataFrame,
    residual_states: pd.DataFrame,
    gates: pd.DataFrame,
    qpred: pd.DataFrame,
    regime: pd.DataFrame,
    prev_actions: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    s = states.loc[date].loc[names]
    rs = residual_states.loc[date].loc[names] if date in residual_states.index.get_level_values(0) else s.copy()
    q = qpred.loc[[date]].reset_index()
    q = q[q["alpha"].isin(names)]
    if q.empty:
        return pd.Series(dtype=float), pd.Series(dtype=object)
    if date in regime.index:
        reg = regime.loc[date]
    else:
        reg = pd.Series({"repair_opportunity": 0.5, "risk_off": 0.5, "ew_budget": 0.24, "tail_gamma": 1.0, "uncertainty_gamma": 0.6, "value_aggression": 5.0, "smoothing": 0.15})
    gate = gates.reindex(index=[date], columns=names).iloc[0].fillna(1.0) if date in gates.index else pd.Series(1.0, index=names)
    state_mult = s["state"].map(STATE_MULT_PLUS).fillna(0.0)
    health_score = (s["health_score"] - s["health_score"].quantile(0.25)).clip(lower=0.0)
    residual_score = (rs["health_score"] - rs["health_score"].quantile(0.25)).clip(lower=0.0)
    base_weight = normalized((0.58 * health_score + 0.42 * residual_score) * state_mult * gate)
    actions, score, _ = choose_actions_for_date(q, s, reg, prev_actions, "qpp_single_agent")
    common_names = actions.index.intersection(base_weight.index)
    action_score = score.reindex(common_names).fillna(0.0)
    value_boost = np.exp(np.clip(float(reg["value_aggression"]) * action_score, -0.28, 0.55))
    low_cov_penalty = pd.Series(1.0, index=common_names)
    low_cov_penalty.loc[(s.loc[common_names, "n_stocks"] <= s["n_stocks"].quantile(0.20)) & (actions.loc[common_names].map(action_ew_share) > 0.35)] = 0.85
    weights = normalized(base_weight.loc[common_names] * value_boost * low_cov_penalty)
    return sleeve_from_weights_actions(weights, actions.loc[common_names]), actions.loc[common_names]


def gov_sleeve_for_month(
    date: pd.Timestamp,
    names: pd.Index,
    states: pd.DataFrame,
    residual_states: pd.DataFrame,
    gates: pd.DataFrame,
    messages: pd.DataFrame,
    reliability: dict[str, float],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    s = states.loc[date].loc[names]
    rs = residual_states.loc[date].loc[names] if date in residual_states.index.get_level_values(0) else s.copy()
    m = messages.loc[date].loc[names]
    gate = gates.reindex(index=[date], columns=names).iloc[0].fillna(1.0) if date in gates.index else pd.Series(1.0, index=names)
    state_mult = s["state"].map(GOV_STATE_MULT).fillna(0.0)
    health_score = (s["health_score"] - s["health_score"].quantile(0.30)).clip(lower=0.0)
    residual_score = (rs["health_score"] - rs["health_score"].quantile(0.30)).clip(lower=0.0)
    blend_w = normalized((0.65 * health_score + 0.35 * residual_score) * state_mult * gate)
    disagreement_scale = m["disagreement"].rank(pct=True).fillna(0.5)
    rel_repair = reliability["repair"]
    rel_constraint = float(np.mean([reliability["liquidity"], reliability["risk"], reliability["challenge"]]))
    rel_value_boost = np.exp(np.clip(6.0 * rel_repair * m["full_pred"].fillna(0.0), -0.20, 0.35))
    conflict_penalty = 1.0 - 0.18 * rel_constraint * disagreement_scale
    veto_penalty = pd.Series(1.0, index=names)
    veto_penalty.loc[m["any_veto"].astype(bool)] = max(0.70, 1.0 - 0.15 * rel_constraint)
    risk_penalty = pd.Series(1.0, index=names)
    risk_penalty.loc[m["reason_code"].str.contains("RISK", na=False)] = max(0.72, 1.0 - 0.12 * reliability["risk"])
    weights = normalized(blend_w * rel_value_boost * conflict_penalty * veto_penalty * risk_penalty)
    actions = m["full_action"].astype(object)
    return sleeve_from_weights_actions(weights, actions), actions, m


def update_governance_reliability(
    reliability: dict[str, float],
    m: pd.DataFrame,
    q_real: pd.Series,
    gov_real: pd.Series,
) -> dict[str, float]:
    common = q_real.index.intersection(gov_real.index)
    repair_value = float((gov_real.loc[common] - q_real.loc[common]).mean()) if len(common) else 0.0
    vals = {}
    for key, mask in {
        "liquidity": m["reason_code"].str.contains("LIQUIDITY", na=False),
        "risk": m["reason_code"].str.contains("RISK", na=False),
        "challenge": m["reason_code"].str.contains("CHALLENGE", na=False),
    }.items():
        idx = m.index[mask].intersection(common)
        vals[key] = float((gov_real.loc[idx] - q_real.loc[idx]).mean()) if len(idx) else 0.0
    out = reliability.copy()
    out["repair"] = float(np.clip(0.985 * out["repair"] + 0.015 * (1.0 + np.tanh(20.0 * repair_value)), 0.65, 1.35))
    out["liquidity"] = float(np.clip(0.985 * out["liquidity"] + 0.015 * (1.0 + np.tanh(30.0 * vals["liquidity"])), 0.65, 1.35))
    out["risk"] = float(np.clip(0.985 * out["risk"] + 0.015 * (1.0 + np.tanh(30.0 * vals["risk"])), 0.65, 1.35))
    out["challenge"] = float(np.clip(0.985 * out["challenge"] + 0.015 * (1.0 + np.tanh(30.0 * vals["challenge"])), 0.65, 1.35))
    return out


def build_sleeve_records(args: argparse.Namespace, cluster_map: dict[str, str]) -> tuple[list[SleeveRecord], pd.DataFrame]:
    input_dir = Path(args.input_dir).expanduser()
    data_root = Path(args.data_root).expanduser()
    returns_by_weighting, long_by_weighting = load_weighting_data(data_root)
    returns_by_action = make_action_returns(returns_by_weighting)
    ff6 = pd.read_csv(input_dir / "external_ff6_monthly.csv", parse_dates=["date"])
    base = returns_by_weighting["vw"]
    states_raw = build_states(base, long_by_weighting["vw"], args.lookback)
    residual = rolling_ff6_residuals(base, ff6, args.residual_lookback)
    residual_states_raw = build_states(residual, long_by_weighting["vw"], args.lookback)
    states = states_raw.set_index(["date", "alpha"]).sort_index()
    residual_states = residual_states_raw.set_index(["date", "alpha"]).sort_index()
    gates = rolling_regime_gate(base, ff6, args.regime_lookback)
    qpred = pd.read_parquet(input_dir / "maspp_q_predictions.parquet").set_index("date").sort_index()
    messages = pd.read_parquet(input_dir / "baseline_agent_messages.parquet").set_index(["date", "alpha"]).sort_index()
    regime = pd.read_csv(input_dir / "maspp_regime_features.csv", parse_dates=["date"]).set_index("date").sort_index()

    prev_q_actions: pd.Series | None = None
    reliability = {"repair": 1.0, "liquidity": 1.0, "risk": 1.0, "challenge": 1.0}
    records: list[SleeveRecord] = []

    for idx, date in enumerate(base.index[:-1]):
        if date < pd.Timestamp(OOS_DATE):
            continue
        if date not in states.index.get_level_values(0) or date not in qpred.index or date not in messages.index.get_level_values(0):
            continue
        next_date = base.index[idx + 1]
        ret_next = base.loc[next_date].dropna()
        names = ret_next.index.intersection(states.loc[date].index).intersection(messages.loc[date].index)
        if date in residual_states.index.get_level_values(0):
            names = names.intersection(residual_states.loc[date].index)
        if len(names) == 0:
            continue

        q_sleeve, q_actions = qpp_sleeve_for_month(date, next_date, names, states, residual_states, gates, qpred, regime, prev_q_actions)
        gov_sleeve, gov_actions, m = gov_sleeve_for_month(date, names, states, residual_states, gates, messages, reliability)
        if q_sleeve.empty or gov_sleeve.empty:
            continue
        month_state = states.loc[date].loc[names]
        q_real = returns_for_actions(returns_by_action, next_date, q_actions)
        gov_real = returns_for_actions(returns_by_weighting, next_date, gov_actions)
        q_cluster = cluster_returns(q_sleeve, returns_by_action, next_date, cluster_map)
        gov_cluster = cluster_returns(gov_sleeve, returns_by_weighting, next_date, cluster_map)
        records.append(
            SleeveRecord(
                date=next_date,
                q_sleeve=q_sleeve,
                gov_sleeve=gov_sleeve,
                q_cluster_ret=q_cluster,
                gov_cluster_ret=gov_cluster,
                avg_disagreement=float(m["disagreement"].mean()),
                veto_rate=float(m["any_veto"].mean()),
                low_coverage=set(month_state.index[month_state["n_stocks"] <= month_state["n_stocks"].quantile(0.20)]),
            )
        )
        prev_q_actions = q_actions
        reliability = update_governance_reliability(reliability, m, q_real, gov_real)

    cluster_df = pd.DataFrame({"alpha": list(cluster_map), "cluster": [cluster_map[a] for a in cluster_map]})
    cluster_df["n_alphas"] = cluster_df.groupby("cluster")["alpha"].transform("size")
    return records, cluster_df.drop_duplicates("alpha")


def records_to_source_port(records: list[SleeveRecord], returns_by_action: dict[str, pd.DataFrame], returns_by_weighting: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    prev_q: pd.Series | None = None
    prev_g: pd.Series | None = None
    prev_q_act: pd.Series | None = None
    prev_g_act: pd.Series | None = None
    for rec in records:
        q_act = dominant_actions(rec.q_sleeve)
        g_act = dominant_actions(rec.gov_sleeve)
        rows.append(
            {
                "date": rec.date,
                "qpp_rebuilt": sleeve_return(rec.q_sleeve, returns_by_action, rec.date),
                "qpp_rebuilt_turnover": sleeve_turnover(rec.q_sleeve, prev_q),
                "qpp_rebuilt_switch_rate": calc_switch_rate(q_act, prev_q_act),
                **{f"qpp_rebuilt_{k}": v for k, v in sleeve_exposure(rec.q_sleeve, rec.low_coverage).items()},
                "gov_rebuilt": sleeve_return(rec.gov_sleeve, returns_by_weighting, rec.date),
                "gov_rebuilt_turnover": sleeve_turnover(rec.gov_sleeve, prev_g),
                "gov_rebuilt_switch_rate": calc_switch_rate(g_act, prev_g_act),
                **{f"gov_rebuilt_{k}": v for k, v in sleeve_exposure(rec.gov_sleeve, rec.low_coverage).items()},
                "avg_disagreement": rec.avg_disagreement,
                "veto_rate": rec.veto_rate,
            }
        )
        prev_q, prev_g = rec.q_sleeve, rec.gov_sleeve
        prev_q_act, prev_g_act = q_act, g_act
    return pd.DataFrame(rows)


def make_dd_budget_candidates(q_net: pd.Series, gov_net: pd.Series) -> dict[str, pd.Series]:
    out = {}
    for dd_thr in [-0.03, -0.04, -0.05, -0.06, -0.08]:
        for ret_thr in [-0.025, -0.035, -0.050]:
            for q_high in [0.75, 0.85, 0.95]:
                for q_low in [0.35, 0.45, 0.55]:
                    b = dd_gate_weights(q_net, dd_thr, ret_thr, q_high, q_low)
                    key = f"dd|dd={dd_thr}|ret={ret_thr}|hi={q_high}|lo={q_low}"
                    out[key] = b * q_net + (1.0 - b) * gov_net
    return out


def budget_from_key(q_net: pd.Series, key: str) -> pd.Series:
    parts = dict(item.split("=", 1) for item in key.split("|")[1:])
    return dd_gate_weights(q_net, float(parts["dd"]), float(parts["ret"]), float(parts["hi"]), float(parts["lo"]))


def fixed_budget_series(index: pd.Index, b: float) -> pd.Series:
    return pd.Series(float(b), index=index)


def global_dynamic_budget(q_net: pd.Series, gov_net: pd.Series, validation_months: int, min_validation: int) -> tuple[pd.Series, pd.Series]:
    candidates = make_dd_budget_candidates(q_net, gov_net)
    _, keys = walk_forward_select(candidates, validation_months, min_validation, utility_score)
    cache = {}
    vals = []
    for date, key in keys.items():
        if key not in cache:
            cache[key] = budget_from_key(q_net, key)
        vals.append(float(cache[key].loc[date]))
    return pd.Series(vals, index=q_net.index), keys


def cluster_return_frame(records: list[SleeveRecord], which: str) -> pd.DataFrame:
    rows = []
    for rec in records:
        dct = rec.q_cluster_ret if which == "q" else rec.gov_cluster_ret
        rows.append({"date": rec.date, **dct})
    return pd.DataFrame(rows).set_index("date").sort_index()


def select_cluster_budgets(
    q_cluster: pd.DataFrame,
    gov_cluster: pd.DataFrame,
    global_b: pd.Series,
    cluster_sizes: dict[str, int],
    validation_months: int,
    min_validation: int,
    shrink_lambda: float,
) -> pd.DataFrame:
    idx = q_cluster.index
    clusters = sorted(set(q_cluster.columns).intersection(gov_cluster.columns))
    out = pd.DataFrame(index=idx, columns=clusters, dtype=float)
    for i, date in enumerate(idx):
        for cluster in clusters:
            omega = cluster_sizes.get(cluster, 0) / (cluster_sizes.get(cluster, 0) + shrink_lambda)
            if i < min_validation:
                local_b = float(global_b.loc[date])
            else:
                lo = max(0, i - validation_months)
                best_b, best_score = float(global_b.loc[date]), -np.inf
                q = q_cluster[cluster].iloc[lo:i].dropna()
                g = gov_cluster[cluster].iloc[lo:i].dropna()
                common = q.index.intersection(g.index)
                if len(common) >= 24:
                    for b in BUDGET_GRID:
                        score = utility_score(b * q.loc[common] + (1.0 - b) * g.loc[common])
                        if score > best_score:
                            best_b, best_score = b, score
                local_b = best_b
            out.loc[date, cluster] = omega * local_b + (1.0 - omega) * float(global_b.loc[date])
    return out.astype(float)


def apply_reliability_adjustment(
    q_cluster: pd.DataFrame,
    gov_cluster: pd.DataFrame,
    base_budget: pd.DataFrame,
    strength: float = 0.16,
) -> pd.DataFrame:
    out = base_budget.copy()
    rel = {c: 1.0 for c in out.columns}
    wealth = {c: 1.0 for c in out.columns}
    peak = {c: 1.0 for c in out.columns}
    for date in out.index:
        for c in out.columns:
            dd = wealth[c] / peak[c] - 1.0
            risk_cut = 0.12 if dd < -0.06 else (0.06 if dd < -0.035 else 0.0)
            out.loc[date, c] = float(np.clip(base_budget.loc[date, c] + strength * (rel[c] - 1.0) - risk_cut, 0.25, 0.97))
        # Outcome is observed only after the decision for this month.
        for c in out.columns:
            if date in q_cluster.index and date in gov_cluster.index and np.isfinite(q_cluster.loc[date, c]) and np.isfinite(gov_cluster.loc[date, c]):
                delta = float(q_cluster.loc[date, c] - gov_cluster.loc[date, c])
                rel[c] = float(np.clip(0.94 * rel[c] + 0.06 * (1.0 + np.tanh(18.0 * delta)), 0.65, 1.35))
                wealth[c] *= 1.0 + float(q_cluster.loc[date, c])
                peak[c] = max(peak[c], wealth[c])
    return out


def backtest_budget_strategy(
    records: list[SleeveRecord],
    returns_by_action: dict[str, pd.DataFrame],
    returns_by_weighting: dict[str, pd.DataFrame],
    cluster_map: dict[str, str],
    budget: pd.Series | pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    rows = []
    prev: pd.Series | None = None
    prev_act: pd.Series | None = None
    for rec in records:
        if isinstance(budget, pd.Series):
            bmap = {cluster: float(budget.loc[rec.date]) for cluster in set(cluster_map.values())}
        else:
            bmap = budget.loc[rec.date].dropna().to_dict()
        sleeve = combine_sleeves(rec.q_sleeve, rec.gov_sleeve, bmap, cluster_map)
        act = dominant_actions(sleeve)
        rows.append(
            {
                "date": rec.date,
                name: sleeve_return(sleeve, {**returns_by_weighting, **returns_by_action}, rec.date),
                f"{name}_turnover": sleeve_turnover(sleeve, prev),
                f"{name}_switch_rate": calc_switch_rate(act, prev_act),
                f"{name}_repair_budget": float(np.mean(list(bmap.values()))),
                f"{name}_budget_std": float(np.std(list(bmap.values()))),
            }
        )
        prev, prev_act = sleeve, act
    return pd.DataFrame(rows)


def backtest_overlay_strategy(
    records: list[SleeveRecord],
    returns_by_action: dict[str, pd.DataFrame],
    exact_defensive: pd.DataFrame,
    cluster_map: dict[str, str],
    budget: pd.Series | pd.DataFrame,
    name: str,
    defensive_col: str = "full_mas",
) -> pd.DataFrame:
    """Open Q++ repair sleeves by cluster and leave the rest in a defensive endpoint.

    The default defensive leg is the already-audited Full MAS portfolio from
    the original experiment.  Passing a non-MAS defensive column creates a
    centralized cluster-capacity baseline with the same budget machinery but
    without the Full MAS governance endpoint.
    """

    exact = exact_defensive.set_index("date")
    clusters = set(cluster_map.values())
    rows = []
    prev_sleeve: pd.Series | None = None
    prev_act: pd.Series | None = None
    for rec in records:
        if rec.date not in exact.index or defensive_col not in exact.columns:
            continue
        if isinstance(budget, pd.Series):
            bmap = {cluster: float(budget.loc[rec.date]) for cluster in clusters}
        else:
            bmap = budget.loc[rec.date].dropna().to_dict()
        q_scale = pd.Series(rec.q_sleeve.index.get_level_values("alpha"), index=rec.q_sleeve.index).map(cluster_map).map(bmap).fillna(0.70)
        active = rec.q_sleeve * q_scale.to_numpy(dtype=float)
        active_weight = float(active.sum())
        defensive_weight = max(0.0, 1.0 - active_weight)
        synthetic = pd.Series(
            [defensive_weight],
            index=pd.MultiIndex.from_tuples([(f"__{defensive_col.upper()}__", defensive_col)], names=["alpha", "action"]),
        )
        sleeve = normalize_sleeve(pd.concat([active, synthetic]).groupby(level=[0, 1]).sum())
        active_return = sleeve_return(sleeve.drop(index=(f"__{defensive_col.upper()}__", defensive_col), errors="ignore"), returns_by_action, rec.date)
        gross = active_return + float(sleeve.loc[(f"__{defensive_col.upper()}__", defensive_col)]) * float(exact.loc[rec.date, defensive_col])
        act = dominant_actions(sleeve)
        exp = sleeve_exposure(sleeve, rec.low_coverage)
        rows.append(
            {
                "date": rec.date,
                name: gross,
                f"{name}_turnover": sleeve_turnover(sleeve, prev_sleeve),
                f"{name}_switch_rate": calc_switch_rate(act, prev_act),
                f"{name}_repair_budget": float(np.mean(list(bmap.values()))),
                f"{name}_budget_std": float(np.std(list(bmap.values()))),
                f"{name}_active_repair_weight": active_weight,
                f"{name}_ew_exposure": exp["ew_exposure"],
                f"{name}_full_ew_weight": exp["full_ew_weight"],
                f"{name}_low_coverage_ew_weight": exp["low_coverage_ew_weight"],
            }
        )
        prev_sleeve, prev_act = sleeve, act
    return pd.DataFrame(rows)


def select_cluster_budgets_against_fullmas(
    q_cluster: pd.DataFrame,
    fullmas_net: pd.Series,
    global_b: pd.Series,
    cluster_sizes: dict[str, int],
    validation_months: int,
    min_validation: int,
    shrink_lambda: float,
) -> pd.DataFrame:
    idx = q_cluster.index
    clusters = sorted(q_cluster.columns)
    out = pd.DataFrame(index=idx, columns=clusters, dtype=float)
    for i, date in enumerate(idx):
        for cluster in clusters:
            n = cluster_sizes.get(cluster, 0)
            omega = n / (n + shrink_lambda)
            if i < min_validation:
                local_b = float(global_b.loc[date])
            else:
                lo = max(0, i - validation_months)
                q = q_cluster[cluster].iloc[lo:i].dropna()
                g = fullmas_net.iloc[lo:i].dropna()
                common = q.index.intersection(g.index)
                best_b, best_score = float(global_b.loc[date]), -np.inf
                if len(common) >= 24:
                    for b in BUDGET_GRID:
                        score = utility_score(b * q.loc[common] + (1.0 - b) * g.loc[common])
                        if score > best_score:
                            best_b, best_score = b, score
                local_b = best_b
            out.loc[date, cluster] = omega * local_b + (1.0 - omega) * float(global_b.loc[date])
    return out.astype(float)


def apply_reliability_against_fullmas(
    q_cluster: pd.DataFrame,
    fullmas_net: pd.Series,
    base_budget: pd.DataFrame,
    strength: float,
) -> pd.DataFrame:
    out = base_budget.copy()
    rel = {c: 1.0 for c in out.columns}
    wealth = {c: 1.0 for c in out.columns}
    peak = {c: 1.0 for c in out.columns}
    for date in out.index:
        for c in out.columns:
            dd = wealth[c] / peak[c] - 1.0
            risk_cut = 0.12 if dd < -0.06 else (0.06 if dd < -0.035 else 0.0)
            out.loc[date, c] = float(np.clip(base_budget.loc[date, c] + strength * (rel[c] - 1.0) - risk_cut, 0.25, 0.97))
        for c in out.columns:
            if date in q_cluster.index and date in fullmas_net.index and np.isfinite(q_cluster.loc[date, c]):
                delta = float(q_cluster.loc[date, c] - fullmas_net.loc[date])
                rel[c] = float(np.clip(0.94 * rel[c] + 0.06 * (1.0 + np.tanh(18.0 * delta)), 0.65, 1.35))
                wealth[c] *= 1.0 + float(q_cluster.loc[date, c])
                peak[c] = max(peak[c], wealth[c])
    return out


def summarize(name: str, ret: pd.Series, stress: pd.Series, extreme: pd.Series, ff6: pd.DataFrame, turnover: float, switch: float, budget: float | None = None) -> dict[str, float | str | int]:
    st = annualized_stats(ret)
    ds = downside_metrics(ret)
    ff = newey_west_alpha(ret, ff6)
    row: dict[str, float | str | int] = {
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
        "avg_turnover": turnover,
        "avg_switch": switch,
        "n_months": st["n_months"],
        **ff,
    }
    if budget is not None:
        row["avg_repair_budget"] = budget
    return row


def summarize_from_port(port: pd.DataFrame, strategies: list[str], ff6: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        gross = port.set_index("date")[strategy]
        net = pd.Series(apply_costs(port, strategy, 10.0, 2.0).to_numpy(), index=gross.index)
        stress = pd.Series(apply_costs(port, strategy, 25.0, 5.0).to_numpy(), index=gross.index)
        extreme = pd.Series(apply_costs(port, strategy, 50.0, 10.0).to_numpy(), index=gross.index)
        budget = float(port[f"{strategy}_repair_budget"].mean()) if f"{strategy}_repair_budget" in port else None
        row = summarize(
                strategy,
                net,
                stress,
                extreme,
                ff6,
                float(port[f"{strategy}_turnover"].mean()),
                float(port[f"{strategy}_switch_rate"].mean()),
                budget,
            )
        for suffix in ["active_repair_weight", "ew_exposure", "full_ew_weight", "low_coverage_ew_weight"]:
            col = f"{strategy}_{suffix}"
            if col in port:
                row[f"avg_{suffix}"] = float(port[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def deployability_diagnostics(port: pd.DataFrame, strategies: list[str], ff6: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_from_port(port, strategies, ff6)
    cols = [
        "strategy",
        "net_sharpe",
        "stress_sharpe",
        "extreme_sharpe",
        "avg_turnover",
        "avg_switch",
        "avg_repair_budget",
        "avg_active_repair_weight",
        "avg_ew_exposure",
        "avg_full_ew_weight",
        "avg_low_coverage_ew_weight",
    ]
    return summary[[c for c in cols if c in summary.columns]]


def cluster_downside_mechanism(q_cluster: pd.DataFrame, gov_cluster: pd.DataFrame, budget: pd.DataFrame, q_port: pd.Series) -> pd.DataFrame:
    panel = []
    for cluster in sorted(set(q_cluster.columns).intersection(gov_cluster.columns).intersection(budget.columns)):
        df = pd.DataFrame(
            {
                "cluster": cluster,
                "budget": budget[cluster],
                "q": q_cluster[cluster],
                "gov": gov_cluster[cluster],
            }
        ).dropna()
        panel.append(df)
    if not panel:
        return pd.DataFrame()
    df = pd.concat(panel)
    df["gap"] = df["q"] - df["gov"]
    df["q_loss"] = df["q"].clip(upper=0.0)
    df["gov_loss"] = df["gov"].clip(upper=0.0)
    df["bucket"] = pd.qcut(df["budget"], 3, labels=["low", "mid", "high"], duplicates="drop")
    q_tail_dates = set(q_port[q_port <= q_port.quantile(0.05)].index)
    rows = []
    for bucket, sub in df.groupby("bucket", observed=True):
        q_cut = sub["q"].quantile(0.05)
        gov_cut = sub["gov"].quantile(0.05)
        gap_cut = sub["gap"].quantile(0.05)
        tail = sub.loc[sub.index.isin(q_tail_dates)]
        rows.append(
            {
                "budget_bucket": str(bucket),
                "n": int(len(sub)),
                "avg_budget": float(sub["budget"].mean()),
                "q_mean": float(sub["q"].mean()),
                "gov_mean": float(sub["gov"].mean()),
                "q_minus_gov": float(sub["gap"].mean()),
                "q_vol": float(sub["q"].std()),
                "q_es5": float(sub.loc[sub["q"] <= q_cut, "q"].mean()),
                "gov_es5": float(sub.loc[sub["gov"] <= gov_cut, "gov"].mean()),
                "gap_es5": float(sub.loc[sub["gap"] <= gap_cut, "gap"].mean()),
                "q_return_in_qpp_tail_months": float(tail["q"].mean()) if not tail.empty else np.nan,
                "gov_return_in_qpp_tail_months": float(tail["gov"].mean()) if not tail.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subperiod_summary(port: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    periods = [
        ("1990-2004", "1990-02-28", "2004-12-31"),
        ("2005-2014", "2005-01-31", "2014-12-31"),
        ("2020-2024", "2020-01-31", "2024-12-31"),
        ("2015-2024", "2015-01-31", "2024-12-31"),
    ]
    p = port.set_index("date")
    rows = []
    for label, start, end in periods:
        sub = p.loc[(p.index >= pd.Timestamp(start)) & (p.index <= pd.Timestamp(end))]
        for strategy in strategies:
            net = pd.Series(apply_costs(sub.reset_index(), strategy, 10.0, 2.0).to_numpy(), index=sub.index)
            rows.append({"period": label, "strategy": strategy, **annualized_stats(net), **downside_metrics(net)})
    return pd.DataFrame(rows)


def predictive_validity(q_net: pd.Series, gov_net: pd.Series, budget: pd.Series) -> pd.DataFrame:
    delta = (q_net - gov_net).rename("realized_q_minus_gov")
    df = pd.concat([budget.rename("budget"), delta], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    x = df["budget"].to_numpy()
    y = df["realized_q_minus_gov"].to_numpy()
    x1 = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(x1.T @ x1) @ (x1.T @ y)
    resid = y - x1 @ beta
    se = math.sqrt(float((resid @ resid) / max(len(y) - 2, 1)) * np.linalg.pinv(x1.T @ x1)[1, 1])
    rows = [
        {
            "test": "linear_beta",
            "estimate": float(beta[1]),
            "t_stat": float(beta[1] / se) if se > 0 else np.nan,
            "pearson": float(df["budget"].corr(df["realized_q_minus_gov"])),
            "spearman": float(df["budget"].corr(df["realized_q_minus_gov"], method="spearman")),
            "n": int(len(df)),
        }
    ]
    try:
        df["bucket"] = pd.qcut(df["budget"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for bucket, sub in df.groupby("bucket", observed=True):
            rows.append({"test": f"budget_{bucket}", "estimate": float(sub["realized_q_minus_gov"].mean()), "t_stat": np.nan, "pearson": np.nan, "spearman": np.nan, "n": int(len(sub))})
    except ValueError:
        pass
    return pd.DataFrame(rows)


def cluster_return_frame_from_sleeves(
    records: list[SleeveRecord],
    returns_by_action: dict[str, pd.DataFrame],
    returns_by_weighting: dict[str, pd.DataFrame],
    cluster_map: dict[str, str],
    which: str,
) -> pd.DataFrame:
    rows = []
    for rec in records:
        sleeve = rec.q_sleeve if which == "q" else rec.gov_sleeve
        returns = returns_by_action if which == "q" else returns_by_weighting
        rows.append({"date": rec.date, **cluster_returns(sleeve, returns, rec.date, cluster_map)})
    return pd.DataFrame(rows).set_index("date").sort_index()


def cluster_gate_validity(
    q_cluster: pd.DataFrame,
    gov_cluster: pd.DataFrame,
    fullmas_net: pd.Series,
    budget: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target_name, target in [
        ("q_minus_gov_sleeve", q_cluster - gov_cluster.reindex_like(q_cluster)),
        ("q_minus_full_mas", q_cluster.sub(fullmas_net, axis=0)),
    ]:
        panel = []
        for cluster in sorted(set(q_cluster.columns).intersection(budget.columns)):
            tmp = pd.DataFrame(
                {
                    "date": q_cluster.index,
                    "cluster": cluster,
                    "budget": budget[cluster].reindex(q_cluster.index).to_numpy(dtype=float),
                    "gap": target[cluster].reindex(q_cluster.index).to_numpy(dtype=float),
                }
            ).dropna()
            panel.append(tmp)
        df = pd.concat(panel, ignore_index=True) if panel else pd.DataFrame()
        if df.empty:
            continue
        # Cluster fixed effects by demeaning within cluster.
        df["b_dm"] = df["budget"] - df.groupby("cluster")["budget"].transform("mean")
        df["g_dm"] = df["gap"] - df.groupby("cluster")["gap"].transform("mean")
        x = df["b_dm"].to_numpy()
        y = df["g_dm"].to_numpy()
        denom = float(x @ x)
        beta = float((x @ y) / denom) if denom > 0 else np.nan
        resid = y - beta * x
        se = math.sqrt(float((resid @ resid) / max(len(y) - df["cluster"].nunique() - 1, 1)) / denom) if denom > 0 else np.nan
        rows.append(
            {
                "target": target_name,
                "test": "cluster_fe_beta",
                "estimate": beta,
                "t_stat": beta / se if se and se > 0 else np.nan,
                "pearson": float(df["budget"].corr(df["gap"])),
                "spearman": float(df["budget"].corr(df["gap"], method="spearman")),
                "n": int(len(df)),
            }
        )
        df["bucket"] = pd.qcut(df["budget"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for bucket, sub in df.groupby("bucket", observed=True):
            rows.append(
                {
                    "target": target_name,
                    "test": f"budget_{bucket}",
                    "estimate": float(sub["gap"].mean()),
                    "t_stat": np.nan,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def block_bootstrap_metric_diff(a: pd.Series, b: pd.Series, metric: str, block: int = 6, n_boot: int = 1000, seed: int = 20260520) -> dict[str, float]:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"obs": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    def val(x: pd.Series) -> float:
        if metric == "sharpe":
            return annualized_stats(x)["sharpe"]
        if metric == "max_drawdown":
            return annualized_stats(x)["max_drawdown"]
        if metric == "es_5":
            return downside_metrics(x)["es_5"]
        if metric == "stress_sharpe":
            return annualized_stats(x)["sharpe"]
        raise ValueError(metric)

    rng = np.random.default_rng(seed)
    vals = df.to_numpy()
    n = len(vals)
    obs = val(df["a"]) - val(df["b"])
    boot = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend([(start + k) % n for k in range(block)])
        sample = vals[idx[:n]]
        boot.append(val(pd.Series(sample[:, 0])) - val(pd.Series(sample[:, 1])))
    arr = np.asarray(boot, dtype=float)
    centered = arr - np.nanmean(arr)
    return {
        "obs": float(obs),
        "ci_low": float(np.nanpercentile(arr, 2.5)),
        "ci_high": float(np.nanpercentile(arr, 97.5)),
        "p_value": float(np.mean(np.abs(centered) >= abs(obs))) if np.isfinite(obs) else np.nan,
    }


def bootstrap_comparisons(port: pd.DataFrame, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    p = port.set_index("date")
    rows = []
    for a, b in comparisons:
        a_net = pd.Series(apply_costs(port, a, 10.0, 2.0).to_numpy(), index=p.index)
        b_net = pd.Series(apply_costs(port, b, 10.0, 2.0).to_numpy(), index=p.index)
        a_stress = pd.Series(apply_costs(port, a, 25.0, 5.0).to_numpy(), index=p.index)
        b_stress = pd.Series(apply_costs(port, b, 25.0, 5.0).to_numpy(), index=p.index)
        for metric, x, y in [
            ("sharpe", a_net, b_net),
            ("stress_sharpe", a_stress, b_stress),
            ("max_drawdown", a_net, b_net),
            ("es_5", a_net, b_net),
        ]:
            res = block_bootstrap_metric_diff(x, y, "sharpe" if metric == "stress_sharpe" else metric)
            rows.append({"comparison": f"{a} minus {b}", "metric": metric, **res})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir).expanduser()
    out_dir = input_dir
    data_root = Path(args.data_root).expanduser()
    returns_by_weighting, _ = load_weighting_data(data_root)
    returns_by_action = make_action_returns(returns_by_weighting)
    ff6 = pd.read_csv(input_dir / "external_ff6_monthly.csv", parse_dates=["date"])

    cluster_map = build_static_clusters(returns_by_weighting["vw"], args.n_clusters, OOS_DATE)
    records, cluster_df = build_sleeve_records(args, cluster_map)
    cluster_df.to_csv(out_dir / "maspp_state_control_clusters.csv", index=False)

    src = records_to_source_port(records, returns_by_action, returns_by_weighting)
    exact = pd.read_csv(input_dir / "maspp_merged_portfolio_returns.csv", parse_dates=["date"])
    exact = exact[
        [
            "date",
            "qpp_single_agent",
            "qpp_single_agent_turnover",
            "qpp_single_agent_switch_rate",
            "full_mas",
            "full_mas_turnover",
            "full_mas_switch_rate",
            "centralized_constrained_q",
            "centralized_constrained_q_turnover",
            "centralized_constrained_q_switch_rate",
            "inverse_vol",
            "inverse_vol_turnover",
            "inverse_vol_switch_rate",
        ]
    ].merge(
        src[
            [
                "date",
                "qpp_rebuilt",
                "qpp_rebuilt_turnover",
                "qpp_rebuilt_switch_rate",
                "qpp_rebuilt_ew_exposure",
                "qpp_rebuilt_full_ew_weight",
                "qpp_rebuilt_low_coverage_ew_weight",
                "gov_rebuilt",
                "gov_rebuilt_turnover",
                "gov_rebuilt_switch_rate",
                "gov_rebuilt_ew_exposure",
                "gov_rebuilt_full_ew_weight",
                "gov_rebuilt_low_coverage_ew_weight",
            ]
        ],
        on="date",
        how="inner",
    )
    q_net = pd.Series(apply_costs(exact, "qpp_single_agent", 10.0, 2.0).to_numpy(), index=exact["date"])
    gov_net = pd.Series(apply_costs(exact, "full_mas", 10.0, 2.0).to_numpy(), index=exact["date"])
    global_b, global_keys = global_dynamic_budget(q_net, gov_net, args.validation_months, args.min_validation)
    q_cluster = cluster_return_frame(records, "q")
    gov_cluster = cluster_return_frame(records, "gov")
    q_cluster.to_csv(out_dir / "maspp_state_control_q_cluster_returns.csv")
    gov_cluster.to_csv(out_dir / "maspp_state_control_gov_cluster_returns.csv")
    cluster_sizes = cluster_df.drop_duplicates("alpha").groupby("cluster")["alpha"].size().to_dict()
    cluster_b = select_cluster_budgets_against_fullmas(q_cluster, gov_net, global_b, cluster_sizes, args.validation_months, args.min_validation, args.shrink_lambda)
    cluster_rel_b = apply_reliability_against_fullmas(q_cluster, gov_net, cluster_b, strength=args.reliability_strength)

    strategy_frames = [
        exact.rename(
            columns={
                "qpp_single_agent": "qpp_exact",
                "qpp_single_agent_turnover": "qpp_exact_turnover",
                "qpp_single_agent_switch_rate": "qpp_exact_switch_rate",
                "qpp_rebuilt_ew_exposure": "qpp_exact_ew_exposure",
                "qpp_rebuilt_full_ew_weight": "qpp_exact_full_ew_weight",
                "qpp_rebuilt_low_coverage_ew_weight": "qpp_exact_low_coverage_ew_weight",
                "full_mas": "full_mas_exact",
                "full_mas_turnover": "full_mas_exact_turnover",
                "full_mas_switch_rate": "full_mas_exact_switch_rate",
                "gov_rebuilt_ew_exposure": "full_mas_exact_ew_exposure",
                "gov_rebuilt_full_ew_weight": "full_mas_exact_full_ew_weight",
                "gov_rebuilt_low_coverage_ew_weight": "full_mas_exact_low_coverage_ew_weight",
                "centralized_constrained_q": "central_q_exact",
                "centralized_constrained_q_turnover": "central_q_exact_turnover",
                "centralized_constrained_q_switch_rate": "central_q_exact_switch_rate",
                "inverse_vol": "inverse_vol_exact",
                "inverse_vol_turnover": "inverse_vol_exact_turnover",
                "inverse_vol_switch_rate": "inverse_vol_exact_switch_rate",
            }
        )
    ]
    strategy_names = ["qpp_exact", "full_mas_exact", "central_q_exact", "inverse_vol_exact", "qpp_rebuilt", "gov_rebuilt"]
    for b in BUDGET_GRID:
        name = f"maspp_fixed_b{str(b).replace('.', '')}"
        strategy_frames.append(backtest_overlay_strategy(records, returns_by_action, exact, cluster_map, fixed_budget_series(q_net.index, b), name))
        strategy_names.append(name)
    for name, budget in [
        ("maspp_global_dynamic", global_b),
        ("maspp_cluster_dynamic", cluster_b),
        ("maspp_cluster_reliability", cluster_rel_b),
    ]:
        strategy_frames.append(backtest_overlay_strategy(records, returns_by_action, exact, cluster_map, budget, name))
        strategy_names.append(name)

    central_net = pd.Series(apply_costs(strategy_frames[0], "central_q_exact", 10.0, 2.0).to_numpy(), index=exact["date"])
    inv_net = pd.Series(apply_costs(strategy_frames[0], "inverse_vol_exact", 10.0, 2.0).to_numpy(), index=exact["date"])
    central_global_b, _ = global_dynamic_budget(q_net, central_net, args.validation_months, args.min_validation)
    inv_global_b, _ = global_dynamic_budget(q_net, inv_net, args.validation_months, args.min_validation)
    central_cluster_b = select_cluster_budgets_against_fullmas(q_cluster, central_net, central_global_b, cluster_sizes, args.validation_months, args.min_validation, args.shrink_lambda)
    inv_cluster_b = select_cluster_budgets_against_fullmas(q_cluster, inv_net, inv_global_b, cluster_sizes, args.validation_months, args.min_validation, args.shrink_lambda)
    for name, budget, defensive_col in [
        ("central_cluster_dynamic", central_cluster_b, "centralized_constrained_q"),
        ("inversevol_cluster_dynamic", inv_cluster_b, "inverse_vol"),
    ]:
        strategy_frames.append(backtest_overlay_strategy(records, returns_by_action, exact, cluster_map, budget, name, defensive_col=defensive_col))
        strategy_names.append(name)

    port = strategy_frames[0]
    for frame in strategy_frames[1:]:
        port = port.merge(frame, on="date", how="inner")
    port.to_csv(out_dir / "maspp_state_control_returns.csv", index=False)
    summary = summarize_from_port(port, strategy_names, ff6)
    summary.to_csv(out_dir / "maspp_state_control_summary.csv", index=False)
    sub = subperiod_summary(port, strategy_names)
    sub.to_csv(out_dir / "maspp_state_control_subperiod.csv", index=False)
    pd.DataFrame({"date": global_b.index, "global_budget": global_b.to_numpy(), "global_key": global_keys.to_numpy()}).to_csv(out_dir / "maspp_state_control_global_budget.csv", index=False)
    cluster_b.reset_index(names="date").to_csv(out_dir / "maspp_state_control_cluster_budget.csv", index=False)
    cluster_rel_b.reset_index(names="date").to_csv(out_dir / "maspp_state_control_cluster_reliability_budget.csv", index=False)
    central_cluster_b.reset_index(names="date").to_csv(out_dir / "maspp_state_control_central_cluster_budget.csv", index=False)
    inv_cluster_b.reset_index(names="date").to_csv(out_dir / "maspp_state_control_inversevol_cluster_budget.csv", index=False)
    validity = predictive_validity(q_net, gov_net, global_b)
    validity.to_csv(out_dir / "maspp_state_control_gate_validity.csv", index=False)
    cluster_valid = cluster_gate_validity(q_cluster, gov_cluster, gov_net, cluster_b)
    cluster_valid.to_csv(out_dir / "maspp_state_control_cluster_gate_validity.csv", index=False)
    budget_dist = []
    for cluster in cluster_b.columns:
        budget_dist.append(
            {
                "cluster": cluster,
                "n_alphas": int(cluster_sizes.get(cluster, 0)),
                "budget_mean": float(cluster_b[cluster].mean()),
                "budget_p10": float(cluster_b[cluster].quantile(0.10)),
                "budget_p90": float(cluster_b[cluster].quantile(0.90)),
                "budget_std": float(cluster_b[cluster].std()),
                "q_minus_fullmas_mean": float((q_cluster[cluster] - gov_net).mean()) if cluster in q_cluster else np.nan,
                "q_minus_gov_sleeve_mean": float((q_cluster[cluster] - gov_cluster[cluster]).mean()) if cluster in q_cluster and cluster in gov_cluster else np.nan,
            }
        )
    pd.DataFrame(budget_dist).to_csv(out_dir / "maspp_state_control_budget_distribution.csv", index=False)
    boot = bootstrap_comparisons(
        port,
        [
            ("maspp_cluster_dynamic", "qpp_exact"),
            ("maspp_cluster_dynamic", "full_mas_exact"),
            ("maspp_cluster_dynamic", "maspp_fixed_b06"),
            ("maspp_cluster_dynamic", "maspp_global_dynamic"),
            ("maspp_cluster_dynamic", "maspp_cluster_reliability"),
            ("maspp_cluster_dynamic", "central_cluster_dynamic"),
            ("maspp_cluster_dynamic", "inversevol_cluster_dynamic"),
        ],
    )
    boot.to_csv(out_dir / "maspp_state_control_bootstrap.csv", index=False)
    deploy = deployability_diagnostics(
        port,
        [
            "qpp_exact",
            "full_mas_exact",
            "central_q_exact",
            "maspp_fixed_b06",
            "maspp_global_dynamic",
            "maspp_cluster_dynamic",
            "central_cluster_dynamic",
            "inversevol_cluster_dynamic",
            "maspp_cluster_reliability",
        ],
        ff6,
    )
    deploy.to_csv(out_dir / "maspp_state_control_deployability.csv", index=False)
    downside_mech = cluster_downside_mechanism(q_cluster, gov_cluster, cluster_b, q_net)
    downside_mech.to_csv(out_dir / "maspp_state_control_downside_mechanism.csv", index=False)

    robust_frames = []
    for shrink in [5.0, 10.0, 20.0, 40.0, 80.0]:
        b = select_cluster_budgets_against_fullmas(q_cluster, gov_net, global_b, cluster_sizes, args.validation_months, args.min_validation, shrink)
        name = f"cluster_shrink_{int(shrink)}"
        pf = backtest_overlay_strategy(records, returns_by_action, exact, cluster_map, b, name)
        sm = summarize_from_port(pf, [name], ff6)
        sm["robustness_type"] = "shrink_lambda"
        sm["robustness_value"] = shrink
        robust_frames.append(sm)
    for n_clusters in [6, 8, 10, 12]:
        alt_map = build_static_clusters(returns_by_weighting["vw"], n_clusters, OOS_DATE)
        alt_q_cluster = cluster_return_frame_from_sleeves(records, returns_by_action, returns_by_weighting, alt_map, "q")
        alt_sizes = pd.Series(alt_map).groupby(pd.Series(alt_map)).size().to_dict()
        alt_b = select_cluster_budgets_against_fullmas(alt_q_cluster, gov_net, global_b, alt_sizes, args.validation_months, args.min_validation, args.shrink_lambda)
        name = f"cluster_n_{n_clusters}"
        pf = backtest_overlay_strategy(records, returns_by_action, exact, alt_map, alt_b, name)
        sm = summarize_from_port(pf, [name], ff6)
        sm["robustness_type"] = "n_clusters"
        sm["robustness_value"] = n_clusters
        robust_frames.append(sm)
    robust = pd.concat(robust_frames, ignore_index=True) if robust_frames else pd.DataFrame()
    robust.to_csv(out_dir / "maspp_state_control_robustness.csv", index=False)
    meta = {
        "input_dir": str(input_dir),
        "n_records": len(records),
        "n_clusters": args.n_clusters,
        "validation_months": args.validation_months,
        "min_validation": args.min_validation,
        "shrink_lambda": args.shrink_lambda,
        "reliability_strength": args.reliability_strength,
    }
    (out_dir / "maspp_state_control_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/alphalife_mas_plus/20260520_171841")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--residual-lookback", type=int, default=120)
    parser.add_argument("--regime-lookback", type=int, default=120)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--validation-months", type=int, default=120)
    parser.add_argument("--min-validation", type=int, default=60)
    parser.add_argument("--shrink-lambda", type=float, default=20.0)
    parser.add_argument("--reliability-strength", type=float, default=0.16)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
