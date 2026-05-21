#!/usr/bin/env python3
"""Strong-conference AlphaLife-MAS empirical extensions.

This script turns the existing lifecycle backtest into a more explicit
constrained offline decision experiment.  It adds cost-aware evaluation,
stronger dynamic allocation baselines, repair placebo tests, fold-level
walk-forward reporting, probabilistic lifecycle-risk calibration, and compact
paper tables.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alphalife_full import add_future_horizons, build_states, load_ff6, tstat_mean
from alphalife_extensions import rolling_ff6_residuals, rolling_regime_gate
from alphalife_mvp import (
    DEFAULT_DATA_ROOT,
    OOS_DATE,
    annualized_stats,
    factor_return_path,
    load_factor_returns,
    pivot_factor_returns,
)


STATE_MULT = {"Healthy": 1.0, "Warning": 0.35, "Decayed": 0.15}
ACTION_SET = ["vw", "ew", "vw_cap"]
LIQUID_ACTION_SET = ["vw", "vw_cap"]
MAIN_STRATEGIES = [
    "static_equal",
    "rolling_sharpe",
    "inverse_vol",
    "factor_momentum",
    "online_hedge",
    "single_model_ridge",
    "alphalife_full",
    "no_redundancy_penalty",
    "alphalife_triggered_repair",
    "liquidity_triggered_repair",
    "q_policy_repair",
    "q_policy_liquid_repair",
    "always_on_repair",
    "random_action_repair",
    "random_trigger_repair",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalized(w: pd.Series) -> pd.Series:
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(w.sum()) <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / w.sum()


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / sd


def available_actions(returns_by_weighting: dict[str, pd.DataFrame], choices: list[str]) -> list[str]:
    return [a for a in choices if a in returns_by_weighting]


def trailing_best_action(
    returns_by_weighting: dict[str, pd.DataFrame],
    alpha: str,
    idx: int,
    lookback: int,
    choices: list[str],
) -> str:
    best, best_score = choices[0], -np.inf
    for action in choices:
        mat = returns_by_weighting[action]
        if alpha not in mat.columns:
            continue
        hist = mat[alpha].iloc[max(0, idx - lookback) : idx].dropna()
        if len(hist) < max(24, lookback // 2):
            continue
        vol = hist.std(ddof=1)
        score = hist.mean() / vol if vol and vol > 0 else -np.inf
        if score > best_score:
            best, best_score = action, score
    return best


def oracle_best_action(
    returns_by_weighting: dict[str, pd.DataFrame],
    alpha: str,
    idx: int,
    horizon: int,
    choices: list[str],
) -> str:
    best, best_return = choices[0], -np.inf
    for action in choices:
        mat = returns_by_weighting[action]
        if alpha not in mat.columns or idx + horizon >= len(mat.index):
            continue
        fdates = mat.index[idx + 1 : idx + 1 + horizon]
        fut = mat.loc[fdates, alpha].sum()
        if np.isfinite(fut) and fut > best_return:
            best, best_return = action, float(fut)
    return best


def returns_for_actions(
    returns_by_weighting: dict[str, pd.DataFrame],
    next_date: pd.Timestamp,
    actions: pd.Series,
) -> pd.Series:
    vals = {}
    for alpha, action in actions.items():
        mat = returns_by_weighting.get(action)
        if mat is None or alpha not in mat.columns or next_date not in mat.index:
            continue
        vals[alpha] = mat.loc[next_date, alpha]
    return pd.Series(vals, dtype=float).dropna()


def calc_turnover(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None or previous.empty:
        return 0.0
    idx = current.index.union(previous.index)
    cur = current.reindex(idx).fillna(0.0)
    prev = previous.reindex(idx).fillna(0.0)
    return float(0.5 * (cur - prev).abs().sum())


def calc_switch_rate(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None or previous.empty:
        return 0.0
    idx = current.index.intersection(previous.index)
    if len(idx) == 0:
        return 0.0
    return float((current.loc[idx] != previous.loc[idx]).mean())


def newey_west_alpha(strategy_returns: pd.Series, ff6: pd.DataFrame, lags: int = 6) -> dict[str, float]:
    ff = ff6.set_index("date")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
    merged = pd.concat([strategy_returns.rename("ret"), ff], axis=1, join="inner").dropna()
    if len(merged) < 60:
        return {"ff6_alpha_ann": np.nan, "ff6_alpha_tstat": np.nan, "ff6_r2": np.nan}
    y = merged["ret"].to_numpy()
    x = np.column_stack([np.ones(len(merged)), merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].to_numpy()])
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    resid = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for t in range(len(y)):
        xt = x[t : t + 1].T
        meat += resid[t] ** 2 * (xt @ xt.T)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = np.zeros_like(meat)
        for t in range(lag, len(y)):
            xt = x[t : t + 1].T
            xl = x[t - lag : t - lag + 1].T
            gamma += resid[t] * resid[t - lag] * (xt @ xl.T)
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    pred = x @ beta
    sst = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1.0 - float(((y - pred) @ (y - pred))) / sst if sst > 0 else np.nan
    return {
        "ff6_alpha_ann": float(beta[0] * 12.0),
        "ff6_alpha_tstat": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        "ff6_r2": float(r2),
    }


def build_ridge_predictions(states: pd.DataFrame, base: pd.DataFrame, lookback: int) -> dict[pd.Timestamp, pd.Series]:
    features = ["health_score", "sharpe60", "tstat60", "ret12", "ret24", "max_abs_corr60", "n_stocks"]
    label = base.shift(-1).stack().rename("next_ret").reset_index()
    label.columns = ["date", "alpha", "next_ret"]
    panel = states[["date", "alpha", *features]].merge(label, on=["date", "alpha"], how="left")
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_ret"])
    out: dict[pd.Timestamp, pd.Series] = {}
    dates = sorted(panel["date"].drop_duplicates())
    for date in dates:
        lo = pd.Timestamp(date) - pd.DateOffset(months=lookback)
        train = panel[(panel["date"] < date) & (panel["date"] >= lo)].dropna(subset=features)
        test = panel[panel["date"] == date].dropna(subset=features)
        if len(train) < 1000 or test.empty:
            continue
        x = train[features].to_numpy(dtype=float)
        y = train["next_ret"].to_numpy(dtype=float)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1.0
        x = (x - mu) / sd
        xt = (test[features].to_numpy(dtype=float) - mu) / sd
        lam = 5.0
        beta = np.linalg.pinv(x.T @ x + lam * np.eye(x.shape[1])) @ (x.T @ y)
        pred = xt @ beta
        out[pd.Timestamp(date)] = pd.Series(pred, index=test["alpha"].to_numpy())
    return out


def build_action_value_panel(
    states: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int,
    choices: list[str],
) -> pd.DataFrame:
    base = returns_by_weighting["vw"]
    base_future = base.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    feature_cols = ["date", "alpha", "health_score", "sharpe60", "tstat60", "ret12", "ret24", "max_abs_corr60", "n_stocks"]
    state_features = states[feature_cols].copy()
    frames = []
    base_ret12 = base.rolling(12, min_periods=8).sum()
    for action in choices:
        mat = returns_by_weighting[action]
        fut = mat.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
        ret12 = mat.rolling(12, min_periods=8).sum()
        mean = mat.rolling(lookback, min_periods=max(24, lookback // 2)).mean()
        vol = mat.rolling(lookback, min_periods=max(24, lookback // 2)).std()
        sharpe = mean / vol * math.sqrt(12)
        df = fut.stack().rename("action_future").reset_index()
        df.columns = ["date", "alpha", "action_future"]
        df["action"] = action
        base_f = base_future.stack().rename("base_future").reset_index()
        base_f.columns = ["date", "alpha", "base_future"]
        action_ret12 = ret12.stack().rename("action_ret12").reset_index()
        action_ret12.columns = ["date", "alpha", "action_ret12"]
        base12 = base_ret12.stack().rename("base_ret12").reset_index()
        base12.columns = ["date", "alpha", "base_ret12"]
        action_sharpe = sharpe.stack().rename("action_sharpe").reset_index()
        action_sharpe.columns = ["date", "alpha", "action_sharpe"]
        df = df.merge(base_f, on=["date", "alpha"], how="left")
        df = df.merge(action_ret12, on=["date", "alpha"], how="left")
        df = df.merge(base12, on=["date", "alpha"], how="left")
        df = df.merge(action_sharpe, on=["date", "alpha"], how="left")
        df["action_vs_base_ret12"] = df["action_ret12"] - df["base_ret12"]
        df["future_improvement"] = df["action_future"] - df["base_future"]
        df["is_ew"] = 1.0 if action == "ew" else 0.0
        df["is_vw_cap"] = 1.0 if action == "vw_cap" else 0.0
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(state_features, on=["date", "alpha"], how="left")
    return panel.replace([np.inf, -np.inf], np.nan)


def learn_action_policy(
    states: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    choices: list[str],
    horizon: int = 12,
    train_window: int = 180,
    min_train: int = 3000,
) -> dict[pd.Timestamp, pd.Series]:
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
    out: dict[pd.Timestamp, pd.Series] = {}
    action_penalty = {"vw": 0.0, "vw_cap": 0.0025, "ew": 0.0100}
    for date in dates:
        train_end = pd.Timestamp(date) - pd.DateOffset(months=horizon)
        train_start = train_end - pd.DateOffset(months=train_window)
        train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)]
        test = panel[panel["date"] == date].dropna(subset=features).copy()
        if len(train) < min_train or test.empty:
            continue
        x = train[features].to_numpy(dtype=float)
        y = train["future_improvement"].to_numpy(dtype=float)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1.0
        x = (x - mu) / sd
        xt = (test[features].to_numpy(dtype=float) - mu) / sd
        lam = 10.0
        beta = np.linalg.pinv(x.T @ x + lam * np.eye(x.shape[1])) @ (x.T @ y)
        test["pred_improvement"] = xt @ beta
        test["adjusted_value"] = test["pred_improvement"] - test["action"].map(action_penalty).fillna(0.0)
        best = test.sort_values(["alpha", "adjusted_value"]).groupby("alpha").tail(1)
        choices_series = pd.Series(best["action"].to_numpy(), index=best["alpha"].to_numpy(), dtype=object)
        values = pd.Series(best["adjusted_value"].to_numpy(), index=best["alpha"].to_numpy(), dtype=float)
        choices_series.loc[values <= 0.0] = "vw"
        out[pd.Timestamp(date)] = choices_series
    return out


def policy_backtest(
    states: pd.DataFrame,
    residual_states: pd.DataFrame,
    gates: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    oos_date: str,
    end_date: str | None = None,
    seed: int = 7,
    q_actions: dict[pd.Timestamp, pd.Series] | None = None,
    q_liquid_actions: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = returns_by_weighting["vw"]
    state_idx = states.set_index(["date", "alpha"]).sort_index()
    res_idx = residual_states.set_index(["date", "alpha"]).sort_index()
    ridge_predictions = build_ridge_predictions(states, base, max(120, lookback))
    all_actions = available_actions(returns_by_weighting, ACTION_SET)
    liquid_actions = available_actions(returns_by_weighting, LIQUID_ACTION_SET)

    prev_weights: dict[str, pd.Series] = {}
    prev_actions: dict[str, pd.Series] = {}
    records = []
    for idx, date in enumerate(base.index[:-1]):
        if date < pd.Timestamp(oos_date):
            continue
        if end_date is not None and date > pd.Timestamp(end_date):
            continue
        if date not in state_idx.index.get_level_values(0):
            continue
        next_date = base.index[idx + 1]
        ret_next = base.loc[next_date].dropna()
        names = ret_next.index.intersection(state_idx.loc[date].index)
        if date in res_idx.index.get_level_values(0):
            names = names.intersection(res_idx.loc[date].index)
        if len(names) == 0:
            continue
        ret_next = ret_next.loc[names]
        s = state_idx.loc[date].loc[names]
        if date in res_idx.index.get_level_values(0):
            rs = res_idx.loc[date].loc[names]
        else:
            rs = s.copy()
        gate = gates.reindex(index=[date], columns=names).iloc[0].fillna(1.0) if date in gates.index else pd.Series(1.0, index=names)

        state_mult = s["state"].map(STATE_MULT).fillna(0.0)
        health_score = (s["health_score"] - s["health_score"].quantile(0.30)).clip(lower=0.0)
        lifecycle_w = normalized(health_score * state_mult)
        no_red_score = (s["health_no_redundancy"] - s["health_no_redundancy"].quantile(0.30)).clip(lower=0.0)
        no_red_w = normalized(no_red_score * state_mult)
        residual_score = (rs["health_score"] - rs["health_score"].quantile(0.30)).clip(lower=0.0)
        blend_w = normalized((0.65 * health_score + 0.35 * residual_score) * state_mult * gate)

        weights: dict[str, pd.Series] = {
            "static_equal": pd.Series(1.0 / len(names), index=names),
            "rolling_sharpe": normalized(s["sharpe60"]),
            "inverse_vol": normalized(1.0 / s["vol60"].replace(0.0, np.nan)),
            "factor_momentum": pd.Series(0.0, index=names),
            "alphalife_full": lifecycle_w,
            "no_redundancy_penalty": no_red_w,
            "alphalife_triggered_repair": blend_w,
            "liquidity_triggered_repair": blend_w,
            "q_policy_repair": blend_w,
            "q_policy_liquid_repair": blend_w,
            "always_on_repair": blend_w,
            "random_action_repair": blend_w,
            "random_trigger_repair": blend_w,
        }
        mom_names = s.index[s["ret12"] >= s["ret12"].quantile(0.70)]
        if len(mom_names):
            weights["factor_momentum"].loc[mom_names] = 1.0 / len(mom_names)
        else:
            weights["factor_momentum"] = weights["static_equal"]
        hedge_signal = zscore(base.loc[:date, names].tail(lookback).sum())
        weights["online_hedge"] = normalized(np.exp(np.clip(1.25 * hedge_signal, -5, 5)))
        pred = ridge_predictions.get(pd.Timestamp(date), pd.Series(dtype=float)).reindex(names)
        weights["single_model_ridge"] = normalized(pred - pred.quantile(0.60)) if pred.notna().sum() > 20 else lifecycle_w

        trigger = s["state"].isin(["Warning", "Decayed"])
        n_trigger = int(trigger.sum())
        learned_actions = pd.Series("vw", index=names, dtype=object)
        liquid_learned_actions = pd.Series("vw", index=names, dtype=object)
        always_actions = pd.Series("vw", index=names, dtype=object)
        random_action = pd.Series("vw", index=names, dtype=object)
        random_trigger_action = pd.Series("vw", index=names, dtype=object)
        q_action = q_actions.get(pd.Timestamp(date), pd.Series(dtype=object)).reindex(names).fillna("vw") if q_actions else pd.Series("vw", index=names, dtype=object)
        q_liq_action = (
            q_liquid_actions.get(pd.Timestamp(date), pd.Series(dtype=object)).reindex(names).fillna("vw")
            if q_liquid_actions
            else pd.Series("vw", index=names, dtype=object)
        )
        random_trigger_names = rng.choice(np.array(names), size=n_trigger, replace=False) if n_trigger > 0 else []
        for alpha in names:
            best = trailing_best_action(returns_by_weighting, alpha, idx, lookback, all_actions)
            best_liq = trailing_best_action(returns_by_weighting, alpha, idx, lookback, liquid_actions)
            always_actions.loc[alpha] = best
            if trigger.loc[alpha]:
                learned_actions.loc[alpha] = best
                liquid_learned_actions.loc[alpha] = best_liq
                random_action.loc[alpha] = rng.choice(all_actions)
            if alpha in set(random_trigger_names):
                random_trigger_action.loc[alpha] = best

        action_map = {
            "static_equal": pd.Series("vw", index=names, dtype=object),
            "rolling_sharpe": pd.Series("vw", index=names, dtype=object),
            "inverse_vol": pd.Series("vw", index=names, dtype=object),
            "factor_momentum": pd.Series("vw", index=names, dtype=object),
            "online_hedge": pd.Series("vw", index=names, dtype=object),
            "single_model_ridge": pd.Series("vw", index=names, dtype=object),
            "alphalife_full": pd.Series("vw", index=names, dtype=object),
            "no_redundancy_penalty": pd.Series("vw", index=names, dtype=object),
            "alphalife_triggered_repair": learned_actions,
            "liquidity_triggered_repair": liquid_learned_actions,
            "q_policy_repair": q_action,
            "q_policy_liquid_repair": q_liq_action,
            "always_on_repair": always_actions,
            "random_action_repair": random_action,
            "random_trigger_repair": random_trigger_action,
        }

        rec: dict[str, object] = {
            "date": next_date,
            "n_alphas": int(len(names)),
            "trigger_rate": float(trigger.mean()),
            "avg_gate": float(gate.mean()),
        }
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
        records.append(rec)
    return pd.DataFrame(records)


def apply_costs(port: pd.DataFrame, strategy: str, turnover_bps: float, switch_bps: float) -> pd.Series:
    cost = turnover_bps / 10000.0 * port[f"{strategy}_turnover"] + switch_bps / 10000.0 * port[f"{strategy}_switch_rate"]
    return port[strategy] - cost


def summarize_cost_aware(port: pd.DataFrame, ff6: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        if strategy not in port:
            continue
        gross = port.set_index("date")[strategy]
        net = pd.Series(apply_costs(port, strategy, 10.0, 2.0).to_numpy(), index=gross.index)
        stress = pd.Series(apply_costs(port, strategy, 25.0, 5.0).to_numpy(), index=gross.index)
        gstats = annualized_stats(gross)
        nstats = annualized_stats(net)
        sstats = annualized_stats(stress)
        ff = newey_west_alpha(net, ff6)
        rows.append(
            {
                "strategy": strategy,
                "gross_sharpe": gstats["sharpe"],
                "net_sharpe_10bps": nstats["sharpe"],
                "stress_net_sharpe": sstats["sharpe"],
                "ann_return_net": nstats["ann_return"],
                "ann_vol_net": nstats["ann_vol"],
                "max_drawdown_net": nstats["max_drawdown"],
                "avg_turnover": float(port[f"{strategy}_turnover"].mean()),
                "avg_switch_rate": float(port[f"{strategy}_switch_rate"].mean()),
                "cost_drag_ann": float((gross - net).mean() * 12.0),
                **ff,
                "n_months": int(gross.dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def repair_attribution(
    states: pd.DataFrame,
    returns_by_weighting: dict[str, pd.DataFrame],
    lookback: int,
    horizon: int,
    oos_date: str,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = returns_by_weighting["vw"]
    all_actions = available_actions(returns_by_weighting, ACTION_SET)
    liquid_actions = available_actions(returns_by_weighting, LIQUID_ACTION_SET)
    state_idx = states.set_index(["date", "alpha"]).sort_index()
    records = []
    for idx, date in enumerate(base.index):
        if date < pd.Timestamp(oos_date) or idx < lookback or idx + horizon >= len(base.index):
            continue
        if date not in state_idx.index.get_level_values(0):
            continue
        s = state_idx.loc[date]
        names = s.index.intersection(base.columns)
        trigger_names = s.index[s["state"].isin(["Warning", "Decayed"])].intersection(base.columns)
        random_names = rng.choice(np.array(names), size=len(trigger_names), replace=False) if len(trigger_names) else []
        fdates = base.index[idx + 1 : idx + 1 + horizon]

        def add_case(alpha: str, trigger_type: str, action_type: str, action: str) -> None:
            base_future = base.loc[fdates, alpha].sum()
            action_future = returns_by_weighting[action].loc[fdates, alpha].sum()
            records.append(
                {
                    "date": date,
                    "alpha": alpha,
                    "trigger_type": trigger_type,
                    "action_type": action_type,
                    "action": action,
                    "base_future_ret": float(base_future),
                    "action_future_ret": float(action_future),
                    "improvement": float(action_future - base_future),
                    "success": bool(action_future > base_future and action_future > 0),
                }
            )

        for alpha in trigger_names:
            learned = trailing_best_action(returns_by_weighting, alpha, idx, lookback, all_actions)
            liquid = trailing_best_action(returns_by_weighting, alpha, idx, lookback, liquid_actions)
            random = rng.choice(all_actions)
            oracle = oracle_best_action(returns_by_weighting, alpha, idx, horizon, all_actions)
            add_case(alpha, "lifecycle", "learned_action", learned)
            add_case(alpha, "lifecycle", "liquidity_action", liquid)
            add_case(alpha, "lifecycle", "random_action", random)
            add_case(alpha, "lifecycle", "oracle_action", oracle)
        for alpha in random_names:
            learned = trailing_best_action(returns_by_weighting, alpha, idx, lookback, all_actions)
            add_case(alpha, "random", "learned_action", learned)
        for alpha in names:
            learned = trailing_best_action(returns_by_weighting, alpha, idx, lookback, all_actions)
            add_case(alpha, "always_on", "learned_action", learned)
    trials = pd.DataFrame(records)
    summary = (
        trials.groupby(["trigger_type", "action_type"])
        .agg(
            n=("improvement", "size"),
            success_rate=("success", "mean"),
            mean_improvement=("improvement", "mean"),
            median_improvement=("improvement", "median"),
            mean_action_future=("action_future_ret", "mean"),
            mean_base_future=("base_future_ret", "mean"),
        )
        .reset_index()
    )
    oracle_mean = summary.loc[(summary["trigger_type"] == "lifecycle") & (summary["action_type"] == "oracle_action"), "mean_improvement"]
    denom = float(oracle_mean.iloc[0]) if not oracle_mean.empty and abs(float(oracle_mean.iloc[0])) > 1e-12 else np.nan
    summary["oracle_capture"] = summary["mean_improvement"] / denom
    return trials, summary


def subperiod_summary(port: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    periods = [
        ("1990-2004", "1990-02-28", "2004-12-31"),
        ("2005-2014", "2005-01-31", "2014-12-31"),
        ("2015-2024", "2015-01-31", "2024-12-31"),
        ("2020-2024", "2020-01-31", "2024-12-31"),
    ]
    rows = []
    p = port.set_index("date")
    for label, start, end in periods:
        sub = p.loc[(p.index >= pd.Timestamp(start)) & (p.index <= pd.Timestamp(end))]
        for strategy in strategies:
            if strategy in sub:
                rows.append({"period": label, "strategy": strategy, **annualized_stats(sub[strategy])})
    return pd.DataFrame(rows)


def nested_walk_forward(
    data_root: Path,
    returns_by_weighting: dict[str, pd.DataFrame],
    long_by_weighting: dict[str, pd.DataFrame],
    ff6: pd.DataFrame,
    out_dir: Path,
    lookbacks: list[int],
) -> pd.DataFrame:
    folds = [
        ("1990-2004", "1990-02-28", "2004-12-31"),
        ("2005-2014", "2005-01-31", "2014-12-31"),
        ("2015-2024", "2015-01-31", "2024-12-31"),
    ]
    ports_by_lb = {}
    for lb in lookbacks:
        states = build_states(returns_by_weighting["vw"], long_by_weighting["vw"], lb)
        residual = rolling_ff6_residuals(returns_by_weighting["vw"], ff6, 120)
        residual_states = build_states(residual, long_by_weighting["vw"], lb)
        gates = rolling_regime_gate(returns_by_weighting["vw"], ff6, 120)
        ports_by_lb[lb] = policy_backtest(states, residual_states, gates, returns_by_weighting, lb, "1975-01-31")
    rows = []
    for fold, start, end in folds:
        start_ts = pd.Timestamp(start)
        val_end = start_ts - pd.DateOffset(months=12)
        val_start = val_end - pd.DateOffset(years=10)
        best_lb, best_score = lookbacks[0], -np.inf
        for lb, port in ports_by_lb.items():
            sub = port[(port["date"] >= val_start) & (port["date"] <= val_end)]
            if len(sub) < 36:
                continue
            net = apply_costs(sub, "alphalife_triggered_repair", 10.0, 2.0)
            score = annualized_stats(net)["sharpe"]
            if np.isfinite(score) and score > best_score:
                best_lb, best_score = lb, score
        test = ports_by_lb[best_lb]
        test = test[(test["date"] >= pd.Timestamp(start)) & (test["date"] <= pd.Timestamp(end))]
        for strategy in ["static_equal", "rolling_sharpe", "alphalife_full", "alphalife_triggered_repair"]:
            net = pd.Series(apply_costs(test, strategy, 10.0, 2.0).to_numpy(), index=test["date"])
            rows.append(
                {
                    "fold": fold,
                    "selected_lookback": best_lb,
                    "validation_net_sharpe": best_score,
                    "strategy": strategy,
                    **annualized_stats(net),
                }
            )
    return pd.DataFrame(rows)


def lifecycle_calibration(states: pd.DataFrame, base: pd.DataFrame, oos_date: str) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return pd.DataFrame(), {"brier": np.nan, "auc": np.nan, "ece": np.nan}

    features = ["health_score", "sharpe60", "tstat60", "ret12", "ret24", "max_abs_corr60", "n_stocks"]
    label = base.rolling(12, min_periods=12).sum().shift(-12).stack().rename("future12").reset_index()
    label.columns = ["date", "alpha", "future12"]
    df = states[["date", "alpha", *features]].merge(label, on=["date", "alpha"], how="left")
    df["cross_median"] = df.groupby("date")["future12"].transform("median")
    df["underperform"] = (df["future12"] < df["cross_median"]).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[*features, "underperform"])
    train = df[df["date"] < pd.Timestamp(oos_date)]
    test = df[df["date"] >= pd.Timestamp(oos_date)]
    if len(train) < 1000 or len(test) < 1000:
        return pd.DataFrame(), {"brier": np.nan, "auc": np.nan, "ece": np.nan}
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train[features])
    xte = scaler.transform(test[features])
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(xtr, train["underperform"])
    prob = model.predict_proba(xte)[:, 1]
    test = test.copy()
    test["prob"] = prob
    test["bin"] = pd.qcut(test["prob"], 10, labels=False, duplicates="drop")
    cal = (
        test.groupby("bin")
        .agg(mean_pred=("prob", "mean"), realized_rate=("underperform", "mean"), n=("underperform", "size"))
        .reset_index()
    )
    ece = float((cal["n"] * (cal["mean_pred"] - cal["realized_rate"]).abs()).sum() / cal["n"].sum())
    metrics = {
        "brier": float(brier_score_loss(test["underperform"], test["prob"])),
        "auc": float(roc_auc_score(test["underperform"], test["prob"])),
        "ece": ece,
    }
    return cal, metrics


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x: float, digits: int = 3) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def write_latex_table(df: pd.DataFrame, path: Path, percent_cols: set[str] | None = None, digits: int = 3) -> None:
    percent_cols = percent_cols or set()
    out = df.copy()
    for col in out.columns:
        if col in percent_cols:
            out[col] = out[col].map(fmt_pct)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: fmt_num(v, digits))
    path.write_text(out.to_latex(index=False, escape=True, column_format="l" + "r" * (len(out.columns) - 1)), encoding="utf-8")


def make_plots(out_dir: Path, port: pd.DataFrame, repair_summary: pd.DataFrame, calibration: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    idx = port.set_index("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in ["static_equal", "rolling_sharpe", "alphalife_full", "alphalife_triggered_repair", "liquidity_triggered_repair"]:
        if strategy not in idx:
            continue
        net = pd.Series(apply_costs(port, strategy, 10.0, 2.0).to_numpy(), index=idx.index)
        (1.0 + net).cumprod().plot(ax=ax, label=strategy)
    ax.set_title("Cost-aware wealth curves")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cost_aware_wealth.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = repair_summary.copy()
    plot_df["case"] = plot_df["trigger_type"] + "/" + plot_df["action_type"]
    plot_df = plot_df[plot_df["case"].isin([
        "lifecycle/learned_action",
        "lifecycle/random_action",
        "random/learned_action",
        "always_on/learned_action",
        "lifecycle/oracle_action",
    ])]
    ax.bar(plot_df["case"], plot_df["mean_improvement"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Repair attribution and placebo tests")
    ax.set_ylabel("Mean 12M improvement")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_repair_attribution.png", dpi=170)
    plt.close(fig)

    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.plot(calibration["mean_pred"], calibration["realized_rate"], marker="o")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Lifecycle risk calibration")
        ax.set_xlabel("Predicted underperformance probability")
        ax.set_ylabel("Realized underperformance rate")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_lifecycle_calibration.png", dpi=170)
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
    states = add_future_horizons(states, base, [12])
    residual = rolling_ff6_residuals(base, ff6, args.residual_lookback)
    residual_states = build_states(residual, long_by_weighting["vw"], args.lookback)
    gates = rolling_regime_gate(base, ff6, args.regime_lookback)
    q_actions = learn_action_policy(states, returns_by_weighting, args.lookback, available_actions(returns_by_weighting, ACTION_SET))
    q_liquid_actions = learn_action_policy(states, returns_by_weighting, args.lookback, available_actions(returns_by_weighting, LIQUID_ACTION_SET))

    port = policy_backtest(
        states,
        residual_states,
        gates,
        returns_by_weighting,
        args.lookback,
        OOS_DATE,
        seed=args.seed,
        q_actions=q_actions,
        q_liquid_actions=q_liquid_actions,
    )
    port.to_csv(out_dir / "strong_policy_portfolio_returns.csv", index=False)
    summary = summarize_cost_aware(port, ff6, MAIN_STRATEGIES)
    summary.to_csv(out_dir / "cost_aware_strategy_summary.csv", index=False)

    trials, repair_summary = repair_attribution(states, returns_by_weighting, args.lookback, 12, OOS_DATE, seed=args.seed + 1)
    trials.to_csv(out_dir / "repair_placebo_trials.csv", index=False)
    repair_summary.to_csv(out_dir / "repair_placebo_summary.csv", index=False)

    selected = ["static_equal", "rolling_sharpe", "online_hedge", "single_model_ridge", "alphalife_full", "alphalife_triggered_repair"]
    subperiod = subperiod_summary(port, selected)
    subperiod.to_csv(out_dir / "subperiod_summary.csv", index=False)

    nested = nested_walk_forward(data_root, returns_by_weighting, long_by_weighting, ff6, out_dir, args.lookbacks)
    nested.to_csv(out_dir / "nested_walk_forward_summary.csv", index=False)

    calibration, cal_metrics = lifecycle_calibration(states, base, OOS_DATE)
    calibration.to_csv(out_dir / "lifecycle_calibration_bins.csv", index=False)
    (out_dir / "lifecycle_calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2), encoding="utf-8")

    tables_dir = ensure_dir(out_dir / "analysis_tables")
    main_rows = summary[summary["strategy"].isin([
        "static_equal",
        "rolling_sharpe",
        "inverse_vol",
        "online_hedge",
        "single_model_ridge",
        "alphalife_full",
        "alphalife_triggered_repair",
        "liquidity_triggered_repair",
        "q_policy_repair",
        "q_policy_liquid_repair",
        "always_on_repair",
    ])][
        [
            "strategy",
            "gross_sharpe",
            "net_sharpe_10bps",
            "stress_net_sharpe",
            "ann_return_net",
            "max_drawdown_net",
            "avg_turnover",
            "avg_switch_rate",
            "ff6_alpha_ann",
            "ff6_alpha_tstat",
        ]
    ]
    write_latex_table(
        main_rows,
        tables_dir / "table_cost_aware_main.tex",
        percent_cols={"ann_return_net", "max_drawdown_net", "avg_turnover", "avg_switch_rate", "ff6_alpha_ann"},
    )

    ablation_rows = summary[summary["strategy"].isin([
        "alphalife_full",
        "no_redundancy_penalty",
        "alphalife_triggered_repair",
        "liquidity_triggered_repair",
        "q_policy_repair",
        "q_policy_liquid_repair",
        "random_action_repair",
        "random_trigger_repair",
        "always_on_repair",
    ])][["strategy", "net_sharpe_10bps", "max_drawdown_net", "avg_turnover", "avg_switch_rate", "ff6_alpha_tstat"]]
    write_latex_table(
        ablation_rows,
        tables_dir / "table_agent_ablation.tex",
        percent_cols={"max_drawdown_net", "avg_turnover", "avg_switch_rate"},
    )

    repair_rows = repair_summary[repair_summary["trigger_type"].isin(["lifecycle", "random", "always_on"])][
        ["trigger_type", "action_type", "n", "success_rate", "mean_improvement", "median_improvement", "oracle_capture"]
    ]
    write_latex_table(
        repair_rows,
        tables_dir / "table_repair_placebo.tex",
        percent_cols={"success_rate", "mean_improvement", "median_improvement", "oracle_capture"},
    )

    sub = subperiod[subperiod["strategy"].isin(["static_equal", "rolling_sharpe", "alphalife_full", "alphalife_triggered_repair"])][
        ["period", "strategy", "sharpe", "ann_return", "max_drawdown", "hit_rate"]
    ]
    write_latex_table(sub, tables_dir / "table_subperiod.tex", percent_cols={"ann_return", "max_drawdown", "hit_rate"})

    nested_rows = nested[nested["strategy"].isin(["static_equal", "rolling_sharpe", "alphalife_full", "alphalife_triggered_repair"])][
        ["fold", "selected_lookback", "strategy", "sharpe", "ann_return", "max_drawdown"]
    ]
    write_latex_table(nested_rows, tables_dir / "table_nested_walk_forward.tex", percent_cols={"ann_return", "max_drawdown"})

    cal_table = pd.DataFrame([cal_metrics])
    write_latex_table(cal_table, tables_dir / "table_calibration_metrics.tex")

    make_plots(out_dir, port, repair_summary, calibration)
    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default="outputs/alphalife_strong")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--lookbacks", type=int, nargs="+", default=[36, 60, 120])
    parser.add_argument("--residual-lookback", type=int, default=120)
    parser.add_argument("--regime-lookback", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
