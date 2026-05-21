#!/usr/bin/env python3
"""Multi-agent coordination layer for AlphaLife.

The earlier strong-conference experiment establishes lifecycle policy learning
and repair attribution.  This script adds a stricter multi-agent layer:
structured agent messages, liquidity/risk/challenge vetoes, disagreement-aware
governance, agent reliability updates, and MAS-specific ablations.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alphalife_extensions import rolling_ff6_residuals, rolling_regime_gate
from alphalife_full import build_states, load_ff6
from alphalife_mvp import (
    DEFAULT_DATA_ROOT,
    OOS_DATE,
    annualized_stats,
    factor_return_path,
    load_factor_returns,
    pivot_factor_returns,
)
from alphalife_strong import (
    ACTION_SET,
    LIQUID_ACTION_SET,
    apply_costs,
    available_actions,
    build_action_value_panel,
    build_ridge_predictions,
    calc_switch_rate,
    calc_turnover,
    ensure_dir,
    newey_west_alpha,
    normalized,
    returns_for_actions,
    write_latex_table,
    zscore,
)


STATE_MULT = {"Healthy": 1.0, "Warning": 0.35, "Decayed": 0.15}
MAS_STRATEGIES = [
    "static_equal",
    "inverse_vol",
    "single_model_ridge",
    "alphalife_full",
    "single_agent_q",
    "centralized_constrained_q",
    "linear_pipeline_mas",
    "mas_no_communication",
    "mas_no_veto",
    "mas_no_challenge",
    "mas_no_reliability",
    "full_mas",
]


def sigmoid(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def build_action_message_panel(
    states: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    choices: list[str],
    horizon: int = 12,
    train_window: int = 180,
    min_train: int = 3000,
) -> pd.DataFrame:
    """Estimate action values and convert them into structured agent messages."""

    panel = build_action_value_panel(states, returns_by_weighting, lookback, horizon, choices)
    features = [
        "health_score",
        "sharpe60",
        "tstat60",
        "ret12",
        "ret24",
        "max_abs_corr60",
        "n_stocks",
        "action_ret12",
        "action_sharpe",
        "action_vs_base_ret12",
        "is_ew",
        "is_vw_cap",
    ]
    clean = panel.dropna(subset=[*features, "future_improvement"]).copy()
    dates = sorted(states["date"].drop_duplicates())
    records: list[dict[str, object]] = []
    base_action_penalty = {"vw": 0.0, "vw_cap": 0.0025, "ew": 0.0100}

    for date in dates:
        date = pd.Timestamp(date)
        train_end = date - pd.DateOffset(months=horizon)
        train_start = train_end - pd.DateOffset(months=train_window)
        train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)]
        test = panel[panel["date"] == date].dropna(subset=features).copy()
        if len(train) < min_train or test.empty:
            continue

        x = train[features].to_numpy(dtype=float)
        y = train["future_improvement"].to_numpy(dtype=float)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1.0
        xz = (x - mu) / sd
        xt = (test[features].to_numpy(dtype=float) - mu) / sd
        lam = 10.0
        beta = np.linalg.pinv(xz.T @ xz + lam * np.eye(xz.shape[1])) @ (xz.T @ y)
        pred_train = xz @ beta
        resid = y - pred_train
        global_sigma = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else 0.05
        sigma_by_action = (
            pd.DataFrame({"action": train["action"], "resid": resid})
            .groupby("action")["resid"]
            .std()
            .fillna(global_sigma)
            .to_dict()
        )

        test["pred_improvement"] = xt @ beta
        test["pred_sigma"] = test["action"].map(sigma_by_action).fillna(global_sigma)
        test["q_value"] = test["pred_improvement"] - test["action"].map(base_action_penalty).fillna(0.0)

        low_coverage = test["n_stocks"] <= test["n_stocks"].quantile(0.25)
        high_redundancy = test["max_abs_corr60"] >= test["max_abs_corr60"].quantile(0.75)
        low_health = test["health_score"] <= test["health_score"].quantile(0.35)
        test["liquidity_penalty"] = 0.0
        test.loc[test["action"] == "ew", "liquidity_penalty"] += 0.008
        test.loc[test["action"] == "vw_cap", "liquidity_penalty"] += 0.0015
        test.loc[(test["action"] == "ew") & low_coverage, "liquidity_penalty"] += 0.003
        test["risk_penalty"] = 0.0
        test.loc[high_redundancy, "risk_penalty"] += 0.0020
        test.loc[high_redundancy & (test["action"] != "vw"), "risk_penalty"] += 0.0010
        test["challenge_penalty"] = 0.0
        test.loc[(test["pred_improvement"] < 0.003) & (test["action"] != "vw"), "challenge_penalty"] += 0.0015
        test.loc[(test["pred_improvement"] < 0.20 * test["pred_sigma"]) & (test["action"] != "vw"), "challenge_penalty"] += 0.0010

        test["liquidity_veto"] = (test["action"] == "ew") & low_coverage & (test["pred_improvement"] < 0.018)
        test["risk_veto"] = high_redundancy & low_health & (test["action"] != "vw") & (test["pred_improvement"] < 0.012)
        test["challenge_veto"] = (test["pred_improvement"] < 0.001) & (test["action"] != "vw")
        test["no_veto_score"] = test["q_value"] - 0.50 * test["liquidity_penalty"] - 0.50 * test["risk_penalty"]
        test["no_challenge_score"] = test["q_value"] - test["liquidity_penalty"] - test["risk_penalty"]
        test["full_score"] = test["no_challenge_score"] - test["challenge_penalty"]
        test["centralized_score"] = test["q_value"] - test["liquidity_penalty"] - test["risk_penalty"] - test["challenge_penalty"]

        state_lookup = (
            states[states["date"] == date]
            .set_index("alpha")[["state", "health_score", "max_abs_corr60", "n_stocks"]]
            .to_dict(orient="index")
        )

        for alpha, g in test.groupby("alpha", sort=False):
            def choose(score_col: str, allowed: pd.Series | None, threshold: float) -> tuple[str, float, float]:
                gg = g if allowed is None else g[allowed.reindex(g.index).fillna(False)]
                if gg.empty:
                    gg = g[g["action"] == "vw"]
                row = gg.sort_values(score_col).iloc[-1]
                action = str(row["action"])
                score = float(row[score_col])
                pred = float(row["pred_improvement"])
                if score <= threshold:
                    action = "vw"
                    base_row = g[g["action"] == "vw"]
                    pred = float(base_row.iloc[0]["pred_improvement"]) if not base_row.empty else 0.0
                    score = max(0.0, score)
                return action, pred, score

            q_action, q_pred, q_score = choose("q_value", None, 0.0)
            central_action, central_pred, central_score = choose("centralized_score", None, 0.0)
            q_row = g[g["action"] == q_action].iloc[0] if q_action in set(g["action"]) else g[g["action"] == "vw"].iloc[0]
            safe_allowed = ~(g["liquidity_veto"] | g["risk_veto"])
            strict_allowed = ~(g["liquidity_veto"] | g["risk_veto"] | g["challenge_veto"])
            no_veto_action, no_veto_pred, no_veto_score = choose("no_veto_score", None, 0.0)
            no_chal_action, no_chal_pred, no_chal_score = choose("no_challenge_score", safe_allowed, 0.0)
            full_action, full_pred, full_score = choose("full_score", strict_allowed, 0.0)

            meta = state_lookup.get(alpha, {})
            state = str(meta.get("state", ""))
            trigger = state in {"Warning", "Decayed"}
            if trigger:
                pipe_action, pipe_pred, pipe_score = choose("q_value", safe_allowed, 0.0)
            else:
                pipe_action, pipe_pred, pipe_score = "vw", float(g[g["action"] == "vw"].iloc[0]["pred_improvement"]), 0.0

            selected = g[g["action"] == q_action].iloc[0] if q_action in set(g["action"]) else g.iloc[0]
            repair_vote = float(selected["q_value"])
            liquidity_vote = -float(selected["liquidity_penalty"])
            risk_vote = -float(selected["risk_penalty"])
            monitor_vote = -0.004 if state == "Warning" else (-0.007 if state == "Decayed" else 0.002)
            challenge_vote = -float(selected["challenge_penalty"])
            disagreement = float(np.std([repair_vote, liquidity_vote, risk_vote, monitor_vote, challenge_vote]))
            reason = []
            if bool(selected["liquidity_veto"]):
                reason.append("REPAIR_VS_LIQUIDITY")
            if bool(selected["risk_veto"]):
                reason.append("REPAIR_VS_RISK")
            if bool(selected["challenge_veto"]):
                reason.append("CHALLENGE_REJECT")
            if not reason and q_action != full_action:
                reason.append("SOFT_CONSTRAINT_REVISION")
            if not reason:
                reason.append("NO_CONFLICT")

            records.append(
                {
                    "date": date,
                    "alpha": alpha,
                    "state": state,
                    "q_action": q_action,
                    "q_pred": q_pred,
                    "q_score": q_score,
                    "centralized_action": central_action,
                    "centralized_pred": central_pred,
                    "centralized_score": central_score,
                    "pipeline_action": pipe_action,
                    "pipeline_pred": pipe_pred,
                    "pipeline_score": pipe_score,
                    "no_veto_action": no_veto_action,
                    "no_veto_pred": no_veto_pred,
                    "no_veto_score": no_veto_score,
                    "no_challenge_action": no_chal_action,
                    "no_challenge_pred": no_chal_pred,
                    "no_challenge_score": no_chal_score,
                    "full_action": full_action,
                    "full_pred": full_pred,
                    "full_score": full_score,
                    "liquidity_veto": bool(selected["liquidity_veto"]),
                    "risk_veto": bool(selected["risk_veto"]),
                    "challenge_veto": bool(selected["challenge_veto"]),
                    "any_veto": bool(selected["liquidity_veto"] or selected["risk_veto"] or selected["challenge_veto"]),
                    "reason_code": "+".join(reason),
                    "disagreement": disagreement,
                    "max_abs_corr60": float(meta.get("max_abs_corr60", np.nan)),
                    "n_stocks": float(meta.get("n_stocks", np.nan)),
                    "health_score": float(meta.get("health_score", np.nan)),
                }
            )
    return pd.DataFrame(records)


def mas_policy_backtest(
    states: pd.DataFrame,
    residual_states: pd.DataFrame,
    gates: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    messages: pd.DataFrame,
    lookback: int,
    oos_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = returns_by_weighting["vw"]
    state_idx = states.set_index(["date", "alpha"]).sort_index()
    res_idx = residual_states.set_index(["date", "alpha"]).sort_index()
    msg_idx = messages.set_index(["date", "alpha"]).sort_index()
    ridge_predictions = build_ridge_predictions(states, base, max(120, lookback))
    prev_weights: dict[str, pd.Series] = {}
    prev_actions: dict[str, pd.Series] = {}
    reliability = {"repair": 1.0, "liquidity": 1.0, "risk": 1.0, "challenge": 1.0}
    records: list[dict[str, object]] = []
    conflict_records: list[dict[str, object]] = []
    reliability_records: list[dict[str, object]] = []

    for idx, date in enumerate(base.index[:-1]):
        if date < pd.Timestamp(oos_date):
            continue
        if date not in state_idx.index.get_level_values(0) or date not in msg_idx.index.get_level_values(0):
            continue
        next_date = base.index[idx + 1]
        ret_next = base.loc[next_date].dropna()
        names = ret_next.index.intersection(state_idx.loc[date].index).intersection(msg_idx.loc[date].index)
        if date in res_idx.index.get_level_values(0):
            names = names.intersection(res_idx.loc[date].index)
        if len(names) == 0:
            continue

        s = state_idx.loc[date].loc[names]
        rs = res_idx.loc[date].loc[names] if date in res_idx.index.get_level_values(0) else s.copy()
        m = msg_idx.loc[date].loc[names]
        gate = gates.reindex(index=[date], columns=names).iloc[0].fillna(1.0) if date in gates.index else pd.Series(1.0, index=names)
        state_mult = s["state"].map(STATE_MULT).fillna(0.0)
        health_score = (s["health_score"] - s["health_score"].quantile(0.30)).clip(lower=0.0)
        residual_score = (rs["health_score"] - rs["health_score"].quantile(0.30)).clip(lower=0.0)
        blend_w = normalized((0.65 * health_score + 0.35 * residual_score) * state_mult * gate)

        pred = ridge_predictions.get(pd.Timestamp(date), pd.Series(dtype=float)).reindex(names)
        single_model_w = normalized(pred - pred.quantile(0.60)) if pred.notna().sum() > 20 else blend_w
        disagreement_scale = m["disagreement"].rank(pct=True).fillna(0.5)
        value_boost = np.exp(np.clip(6.0 * m["full_pred"].fillna(0.0), -0.20, 0.35))
        centralized_boost = np.exp(np.clip(5.5 * m["centralized_pred"].fillna(0.0), -0.20, 0.35))
        no_veto_boost = np.exp(np.clip(5.0 * m["no_veto_pred"].fillna(0.0), -0.20, 0.35))
        no_challenge_boost = np.exp(np.clip(5.5 * m["no_challenge_pred"].fillna(0.0), -0.20, 0.35))
        rel_repair = reliability["repair"]
        rel_constraint = float(np.mean([reliability["liquidity"], reliability["risk"], reliability["challenge"]]))
        rel_value_boost = np.exp(np.clip(6.0 * rel_repair * m["full_pred"].fillna(0.0), -0.20, 0.35))
        conflict_penalty = 1.0 - 0.18 * rel_constraint * disagreement_scale
        veto_penalty = pd.Series(1.0, index=names)
        veto_penalty.loc[m["any_veto"].astype(bool)] = max(0.70, 1.0 - 0.15 * rel_constraint)
        risk_penalty = pd.Series(1.0, index=names)
        risk_penalty.loc[m["reason_code"].str.contains("RISK", na=False)] = max(0.72, 1.0 - 0.12 * reliability["risk"])
        fixed_veto_penalty = pd.Series(1.0, index=names)
        fixed_veto_penalty.loc[m["any_veto"].astype(bool)] = 0.85
        fixed_risk_penalty = pd.Series(1.0, index=names)
        fixed_risk_penalty.loc[m["reason_code"].str.contains("RISK", na=False)] = 0.88

        weights: dict[str, pd.Series] = {
            "static_equal": pd.Series(1.0 / len(names), index=names),
            "inverse_vol": normalized(1.0 / s["vol60"].replace(0.0, np.nan)),
            "single_model_ridge": single_model_w,
            "alphalife_full": normalized(health_score * state_mult),
            "single_agent_q": blend_w,
            "centralized_constrained_q": normalized(blend_w * centralized_boost),
            "linear_pipeline_mas": normalized(blend_w * (1.0 - 0.18 * disagreement_scale)),
            "mas_no_communication": blend_w,
            "mas_no_veto": normalized(blend_w * no_veto_boost),
            "mas_no_challenge": normalized(blend_w * no_challenge_boost * veto_penalty),
            "mas_no_reliability": normalized(blend_w * value_boost * (1.0 - 0.18 * disagreement_scale) * fixed_veto_penalty * fixed_risk_penalty),
            "full_mas": normalized(blend_w * rel_value_boost * conflict_penalty * veto_penalty * risk_penalty),
        }
        action_map = {
            "static_equal": pd.Series("vw", index=names, dtype=object),
            "inverse_vol": pd.Series("vw", index=names, dtype=object),
            "single_model_ridge": pd.Series("vw", index=names, dtype=object),
            "alphalife_full": pd.Series("vw", index=names, dtype=object),
            "single_agent_q": m["q_action"].astype(object),
            "centralized_constrained_q": m["centralized_action"].astype(object),
            "linear_pipeline_mas": m["pipeline_action"].astype(object),
            "mas_no_communication": m["q_action"].astype(object),
            "mas_no_veto": m["no_veto_action"].astype(object),
            "mas_no_challenge": m["no_challenge_action"].astype(object),
            "mas_no_reliability": m["full_action"].astype(object),
            "full_mas": m["full_action"].astype(object),
        }

        rec: dict[str, object] = {
            "date": next_date,
            "n_alphas": int(len(names)),
            "avg_disagreement": float(m["disagreement"].mean()),
            "veto_rate": float(m["any_veto"].mean()),
            "repair_liquidity_conflict_rate": float(m["reason_code"].str.contains("LIQUIDITY", na=False).mean()),
            "repair_risk_conflict_rate": float(m["reason_code"].str.contains("RISK", na=False).mean()),
            "challenge_reject_rate": float(m["reason_code"].str.contains("CHALLENGE", na=False).mean()),
            "repair_reliability": reliability["repair"],
            "liquidity_reliability": reliability["liquidity"],
            "risk_reliability": reliability["risk"],
            "challenge_reliability": reliability["challenge"],
        }
        realized_by_strategy: dict[str, pd.Series] = {}
        for strategy, w in weights.items():
            act = action_map[strategy]
            realized = returns_for_actions(returns_by_weighting, next_date, act)
            common = realized.index.intersection(w.index)
            ww = normalized(w.loc[common])
            rec[strategy] = float((realized.loc[common] * ww).sum())
            rec[f"{strategy}_turnover"] = calc_turnover(ww, prev_weights.get(strategy))
            rec[f"{strategy}_switch_rate"] = calc_switch_rate(act.loc[common], prev_actions.get(strategy))
            prev_weights[strategy] = ww
            prev_actions[strategy] = act.loc[common]
            realized_by_strategy[strategy] = realized

        q_real = realized_by_strategy["single_agent_q"]
        full_real = realized_by_strategy["full_mas"]
        for conflict_name, mask in {
            "repair_vs_liquidity": m["reason_code"].str.contains("LIQUIDITY", na=False),
            "repair_vs_risk": m["reason_code"].str.contains("RISK", na=False),
            "challenge_reject": m["reason_code"].str.contains("CHALLENGE", na=False),
            "soft_revision": m["reason_code"].str.contains("SOFT", na=False),
            "any_conflict": m["reason_code"] != "NO_CONFLICT",
        }.items():
            names_c = m.index[mask].intersection(q_real.index).intersection(full_real.index)
            if len(names_c) == 0:
                continue
            conflict_records.append(
                {
                    "date": next_date,
                    "conflict_type": conflict_name,
                    "n": int(len(names_c)),
                    "q_action_return": float(q_real.loc[names_c].mean()),
                    "full_action_return": float(full_real.loc[names_c].mean()),
                    "governor_value": float((full_real.loc[names_c] - q_real.loc[names_c]).mean()),
                }
            )

        q_full_common = q_real.index.intersection(full_real.index)
        if len(q_full_common):
            repair_value = float((full_real.loc[q_full_common] - q_real.loc[q_full_common]).mean())
        else:
            repair_value = 0.0
        veto_df = pd.DataFrame(conflict_records[-5:])
        liq_value = float(veto_df.loc[veto_df["conflict_type"] == "repair_vs_liquidity", "governor_value"].mean()) if not veto_df.empty else 0.0
        risk_value = float(veto_df.loc[veto_df["conflict_type"] == "repair_vs_risk", "governor_value"].mean()) if not veto_df.empty else 0.0
        challenge_value = float(veto_df.loc[veto_df["conflict_type"] == "challenge_reject", "governor_value"].mean()) if not veto_df.empty else 0.0

        reliability["repair"] = float(np.clip(0.985 * reliability["repair"] + 0.015 * (1.0 + np.tanh(20.0 * repair_value)), 0.65, 1.35))
        reliability["liquidity"] = float(np.clip(0.985 * reliability["liquidity"] + 0.015 * (1.0 + np.tanh(30.0 * liq_value)), 0.65, 1.35))
        reliability["risk"] = float(np.clip(0.985 * reliability["risk"] + 0.015 * (1.0 + np.tanh(30.0 * risk_value)), 0.65, 1.35))
        reliability["challenge"] = float(np.clip(0.985 * reliability["challenge"] + 0.015 * (1.0 + np.tanh(30.0 * challenge_value)), 0.65, 1.35))
        reliability_records.append({"date": next_date, **reliability})
        records.append(rec)

    return pd.DataFrame(records), pd.DataFrame(conflict_records), pd.DataFrame(reliability_records)


def summarize_mas(port: pd.DataFrame, ff6: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        if strategy not in port:
            continue
        gross = port.set_index("date")[strategy]
        net = pd.Series(apply_costs(port, strategy, 10.0, 2.0).to_numpy(), index=gross.index)
        stress = pd.Series(apply_costs(port, strategy, 25.0, 5.0).to_numpy(), index=gross.index)
        extreme = pd.Series(apply_costs(port, strategy, 50.0, 10.0).to_numpy(), index=gross.index)
        nstats = annualized_stats(net)
        sstats = annualized_stats(stress)
        estats = annualized_stats(extreme)
        dstats = downside_metrics(net)
        ff = newey_west_alpha(net, ff6)
        rows.append(
            {
                "strategy": strategy,
                "net_sharpe": nstats["sharpe"],
                "stress_sharpe": sstats["sharpe"],
                "extreme_sharpe": estats["sharpe"],
                "ann_return": nstats["ann_return"],
                "max_drawdown": nstats["max_drawdown"],
                **dstats,
                "avg_turnover": float(port[f"{strategy}_turnover"].mean()),
                "avg_switch": float(port[f"{strategy}_switch_rate"].mean()),
                **ff,
            }
        )
    return pd.DataFrame(rows)


def disagreement_table(port: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    p = port.copy()
    p["disagreement_bucket"] = pd.qcut(p["avg_disagreement"], 3, labels=["low", "medium", "high"], duplicates="drop")
    rows = []
    for bucket, sub in p.groupby("disagreement_bucket", observed=True):
        for strategy in strategies:
            net = pd.Series(apply_costs(sub, strategy, 10.0, 2.0).to_numpy(), index=sub["date"])
            rows.append({"bucket": str(bucket), "strategy": strategy, **annualized_stats(net), "n_months": len(sub)})
    return pd.DataFrame(rows)


def conflict_summary(conflicts: pd.DataFrame) -> pd.DataFrame:
    if conflicts.empty:
        return pd.DataFrame(columns=["conflict_type", "n_cases", "q_action_return", "full_action_return", "governor_value"])
    return (
        conflicts.groupby("conflict_type")
        .agg(
            n_cases=("n", "sum"),
            q_action_return=("q_action_return", "mean"),
            full_action_return=("full_action_return", "mean"),
            governor_value=("governor_value", "mean"),
        )
        .reset_index()
    )


def downside_metrics(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {"calmar": np.nan, "es_5": np.nan, "worst_month": np.nan}
    stats = annualized_stats(ret)
    mdd = stats["max_drawdown"]
    calmar = stats["ann_return"] / abs(mdd) if np.isfinite(mdd) and mdd < 0 else np.nan
    q = ret.quantile(0.05)
    es_5 = ret[ret <= q].mean()
    return {"calmar": float(calmar), "es_5": float(es_5), "worst_month": float(ret.min())}


def moving_block_bootstrap_sharpe_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    block: int = 6,
    n_boot: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    df = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"obs_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    vals = df.to_numpy()
    obs = annualized_stats(df["a"])["sharpe"] - annualized_stats(df["b"])["sharpe"]
    n = len(vals)
    boot = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend([(start + k) % n for k in range(block)])
        sample = vals[idx[:n]]
        ra = pd.Series(sample[:, 0])
        rb = pd.Series(sample[:, 1])
        boot.append(annualized_stats(ra)["sharpe"] - annualized_stats(rb)["sharpe"])
    boot_arr = np.array(boot, dtype=float)
    centered = boot_arr - np.nanmean(boot_arr)
    pval = float(np.mean(np.abs(centered) >= abs(obs))) if np.isfinite(obs) else np.nan
    return {
        "obs_diff": float(obs),
        "ci_low": float(np.nanpercentile(boot_arr, 2.5)),
        "ci_high": float(np.nanpercentile(boot_arr, 97.5)),
        "p_value": pval,
    }


def bootstrap_table(port: pd.DataFrame) -> pd.DataFrame:
    p = port.set_index("date")
    pairs = [
        ("full_mas", "centralized_constrained_q"),
        ("full_mas", "single_agent_q"),
        ("full_mas", "linear_pipeline_mas"),
        ("full_mas", "mas_no_communication"),
        ("full_mas", "mas_no_veto"),
        ("full_mas", "mas_no_challenge"),
        ("full_mas", "mas_no_reliability"),
        ("full_mas", "alphalife_full"),
    ]
    rows = []
    for a, b in pairs:
        ra = pd.Series(apply_costs(p.reset_index(), a, 10.0, 2.0).to_numpy(), index=p.index)
        rb = pd.Series(apply_costs(p.reset_index(), b, 10.0, 2.0).to_numpy(), index=p.index)
        rows.append({"comparison": f"{a} minus {b}", **moving_block_bootstrap_sharpe_diff(ra, rb)})
    return pd.DataFrame(rows)


def moving_block_bootstrap_metric_diff(
    returns_a: pd.Series,
    returns_b: pd.Series,
    metric: str,
    block: int = 6,
    n_boot: int = 1000,
    seed: int = 456,
) -> dict[str, float]:
    df = pd.concat([returns_a.rename("a"), returns_b.rename("b")], axis=1).dropna()
    if len(df) < 60:
        return {"obs_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    def value(x: pd.Series) -> float:
        if metric == "max_drawdown":
            return annualized_stats(x)["max_drawdown"]
        if metric == "es_5":
            return downside_metrics(x)["es_5"]
        if metric == "calmar":
            return downside_metrics(x)["calmar"]
        raise ValueError(metric)

    rng = np.random.default_rng(seed)
    vals = df.to_numpy()
    obs = value(df["a"]) - value(df["b"])
    n = len(vals)
    boot = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            idx.extend([(start + k) % n for k in range(block)])
        sample = vals[idx[:n]]
        boot.append(value(pd.Series(sample[:, 0])) - value(pd.Series(sample[:, 1])))
    boot_arr = np.array(boot, dtype=float)
    centered = boot_arr - np.nanmean(boot_arr)
    pval = float(np.mean(np.abs(centered) >= abs(obs))) if np.isfinite(obs) else np.nan
    return {
        "obs_diff": float(obs),
        "ci_low": float(np.nanpercentile(boot_arr, 2.5)),
        "ci_high": float(np.nanpercentile(boot_arr, 97.5)),
        "p_value": pval,
    }


def downside_bootstrap_table(port: pd.DataFrame) -> pd.DataFrame:
    p = port.set_index("date")
    pairs = [
        ("full_mas", "centralized_constrained_q"),
        ("full_mas", "single_agent_q"),
        ("full_mas", "linear_pipeline_mas"),
        ("full_mas", "alphalife_full"),
    ]
    rows = []
    for a, b in pairs:
        ra = pd.Series(apply_costs(p.reset_index(), a, 10.0, 2.0).to_numpy(), index=p.index)
        rb = pd.Series(apply_costs(p.reset_index(), b, 10.0, 2.0).to_numpy(), index=p.index)
        dd = moving_block_bootstrap_metric_diff(ra, rb, "max_drawdown")
        es = moving_block_bootstrap_metric_diff(ra, rb, "es_5", seed=789)
        rows.append(
            {
                "comparison": f"{a} minus {b}",
                "max_dd_diff": dd["obs_diff"],
                "dd_ci_low": dd["ci_low"],
                "dd_ci_high": dd["ci_high"],
                "es5_diff": es["obs_diff"],
                "es_ci_low": es["ci_low"],
                "es_ci_high": es["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def implementation_attribution(
    messages: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if messages.empty:
        return pd.DataFrame()
    base = returns_by_weighting["vw"]
    dates = list(base.index)
    date_to_next = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    low_cutoff = messages["n_stocks"].quantile(0.25)
    rows: list[dict[str, object]] = []
    policy_actions = {
        "single_agent_q": "q_action",
        "centralized_constrained_q": "centralized_action",
        "full_mas": "full_action",
    }
    for _, row in messages.iterrows():
        date = pd.Timestamp(row["date"])
        next_date = date_to_next.get(date)
        if next_date is None:
            continue
        alpha = row["alpha"]
        for policy, action_col in policy_actions.items():
            action = row[action_col]
            mat = returns_by_weighting.get(action)
            if mat is None or alpha not in mat.columns or next_date not in mat.index:
                continue
            rows.append(
                {
                    "policy": policy,
                    "action": action,
                    "next_return": float(mat.loc[next_date, alpha]),
                    "pred_improvement": float(row.get(action_col.replace("action", "pred"), np.nan)),
                    "low_coverage": bool(row["n_stocks"] <= low_cutoff),
                    "n_stocks": float(row["n_stocks"]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = (
        df.groupby(["policy", "action"])
        .agg(
            n=("next_return", "size"),
            selection_share=("next_return", lambda x: len(x) / len(df[df["policy"] == df.loc[x.index[0], "policy"]])),
            mean_next_return=("next_return", "mean"),
            low_coverage_share=("low_coverage", "mean"),
            mean_n_stocks=("n_stocks", "mean"),
        )
        .reset_index()
    )
    return out


def make_mas_plots(out_dir: Path, port: pd.DataFrame, reliability: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    idx = port.set_index("date")
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for strategy, label in [
        ("alphalife_full", "Lifecycle"),
        ("mas_no_communication", "No communication"),
        ("centralized_constrained_q", "Centralized constrained Q"),
        ("linear_pipeline_mas", "Pipeline MAS"),
        ("full_mas", "Full MAS"),
    ]:
        net = pd.Series(apply_costs(port, strategy, 10.0, 2.0).to_numpy(), index=idx.index)
        (1.0 + net).cumprod().plot(ax=ax, linewidth=1.1, label=label)
    ax.set_ylabel("Growth of $1", fontsize=7)
    ax.set_xlabel("Year", fontsize=7)
    ax.set_title("MAS coordination wealth", fontsize=8)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(alpha=0.35, linewidth=0.4)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mas_wealth_single.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "fig_mas_wealth_single.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if not reliability.empty:
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        r = reliability.set_index("date")
        for col, label in [
            ("repair", "Repair"),
            ("liquidity", "Liquidity"),
            ("risk", "Risk"),
            ("challenge", "Challenge"),
        ]:
            if col in r:
                r[col].plot(ax=ax, linewidth=1.0, label=label)
        ax.set_ylabel("Reliability", fontsize=7)
        ax.set_xlabel("Year", fontsize=7)
        ax.set_title("Agent reliability update", fontsize=8)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.grid(alpha=0.35, linewidth=0.4)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(out_dir / "fig_agent_reliability_single.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / "fig_agent_reliability_single.png", dpi=320, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).expanduser()
    out_dir = ensure_dir(Path(args.out_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S"))
    returns_by_weighting: dict[str, pd.DataFrame] = {}
    long_by_weighting: dict[str, pd.DataFrame] = {}
    for weighting in ["vw", "ew", "vw_cap"]:
        if factor_return_path(data_root, weighting).exists():
            long = load_factor_returns(data_root, weighting)
            long_by_weighting[weighting] = long
            returns_by_weighting[weighting] = pivot_factor_returns(long)
    if "vw" not in returns_by_weighting:
        raise FileNotFoundError("Missing VW factor returns")

    ff6 = load_ff6(out_dir)
    base = returns_by_weighting["vw"]
    states = build_states(base, long_by_weighting["vw"], args.lookback)
    residual = rolling_ff6_residuals(base, ff6, args.residual_lookback)
    residual_states = build_states(residual, long_by_weighting["vw"], args.lookback)
    gates = rolling_regime_gate(base, ff6, args.regime_lookback)
    messages = build_action_message_panel(
        states,
        returns_by_weighting,
        args.lookback,
        available_actions(returns_by_weighting, ACTION_SET),
        horizon=args.horizon,
    )
    messages.to_parquet(out_dir / "agent_messages.parquet", index=False)
    messages.to_csv(out_dir / "agent_messages_sample.csv", index=False)

    port, conflicts, reliability = mas_policy_backtest(states, residual_states, gates, returns_by_weighting, messages, args.lookback, OOS_DATE)
    port.to_csv(out_dir / "mas_policy_portfolio_returns.csv", index=False)
    conflicts.to_csv(out_dir / "mas_conflict_outcomes.csv", index=False)
    reliability.to_csv(out_dir / "agent_reliability_history.csv", index=False)

    summary = summarize_mas(port, ff6, MAS_STRATEGIES)
    summary.to_csv(out_dir / "mas_strategy_summary.csv", index=False)
    conflicts_sum = conflict_summary(conflicts)
    conflicts_sum.to_csv(out_dir / "mas_conflict_summary.csv", index=False)
    disagree = disagreement_table(port, ["mas_no_communication", "centralized_constrained_q", "linear_pipeline_mas", "full_mas"])
    disagree.to_csv(out_dir / "mas_disagreement_buckets.csv", index=False)
    boot = bootstrap_table(port)
    boot.to_csv(out_dir / "mas_bootstrap_sharpe_diff.csv", index=False)
    downside_boot = downside_bootstrap_table(port)
    downside_boot.to_csv(out_dir / "mas_downside_bootstrap.csv", index=False)
    impl_attr = implementation_attribution(messages, returns_by_weighting)
    impl_attr.to_csv(out_dir / "mas_implementation_attribution.csv", index=False)

    tables_dir = ensure_dir(out_dir / "analysis_tables")
    perf_rows = summary[summary["strategy"].isin(MAS_STRATEGIES)][
        [
            "strategy",
            "net_sharpe",
            "stress_sharpe",
            "extreme_sharpe",
            "ann_return",
            "max_drawdown",
            "calmar",
            "es_5",
            "avg_turnover",
            "avg_switch",
            "ff6_alpha_ann",
            "ff6_alpha_tstat",
        ]
    ]
    write_latex_table(
        perf_rows,
        tables_dir / "table_mas_coordination.tex",
        percent_cols={"ann_return", "max_drawdown", "es_5", "avg_turnover", "avg_switch", "ff6_alpha_ann"},
    )

    conflict_rows = conflicts_sum[["conflict_type", "n_cases", "q_action_return", "full_action_return", "governor_value"]]
    write_latex_table(
        conflict_rows,
        tables_dir / "table_conflict_resolution.tex",
        percent_cols={"q_action_return", "full_action_return", "governor_value"},
    )

    dis_rows = disagree[disagree["strategy"].isin(["mas_no_communication", "centralized_constrained_q", "linear_pipeline_mas", "full_mas"])][
        ["bucket", "strategy", "sharpe", "ann_return", "max_drawdown", "n_months"]
    ]
    write_latex_table(dis_rows, tables_dir / "table_disagreement_buckets.tex", percent_cols={"ann_return", "max_drawdown"})
    write_latex_table(boot, tables_dir / "table_bootstrap_sharpe.tex")
    write_latex_table(
        downside_boot,
        tables_dir / "table_downside_bootstrap.tex",
        percent_cols={"max_dd_diff", "dd_ci_low", "dd_ci_high", "es5_diff", "es_ci_low", "es_ci_high"},
    )
    write_latex_table(
        impl_attr,
        tables_dir / "table_implementation_attribution.tex",
        percent_cols={"selection_share", "mean_next_return", "low_coverage_share"},
    )

    (out_dir / "mas_metrics.json").write_text(
        json.dumps(
            {
                "mean_veto_rate": float(port["veto_rate"].mean()),
                "mean_disagreement": float(port["avg_disagreement"].mean()),
                "mean_liquidity_conflict_rate": float(port["repair_liquidity_conflict_rate"].mean()),
                "mean_risk_conflict_rate": float(port["repair_risk_conflict_rate"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_mas_plots(out_dir, port, reliability)
    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default="outputs/alphalife_mas")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--residual-lookback", type=int, default=120)
    parser.add_argument("--regime-lookback", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
