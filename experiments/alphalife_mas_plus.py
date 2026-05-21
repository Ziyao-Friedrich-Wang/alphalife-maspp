#!/usr/bin/env python3
"""AlphaLife-MAS++ performance experiment.

This script keeps the existing AlphaLife-MAS data and evaluation protocol but
upgrades the decision layer:

1. distributional multi-horizon action-value ensemble;
2. meta-label filtering for action execution;
3. soft EW/capacity budgets using blended implementation actions;
4. regime-adaptive risk budgets;
5. a joint action-weight governor with lightweight turnover smoothing.

It intentionally does not edit the paper.  Outputs are written under
outputs/alphalife_mas_plus/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from alphalife_extensions import rolling_ff6_residuals, rolling_regime_gate
from alphalife_full import build_states, load_ff6
from alphalife_mas import (
    MAS_STRATEGIES,
    build_action_message_panel,
    disagreement_table,
    implementation_attribution,
    mas_policy_backtest,
    summarize_mas,
)
from alphalife_mvp import DEFAULT_DATA_ROOT, OOS_DATE, annualized_stats, factor_return_path, load_factor_returns, pivot_factor_returns
from alphalife_strong import (
    ACTION_SET,
    apply_costs,
    available_actions,
    build_action_value_panel,
    calc_switch_rate,
    calc_turnover,
    ensure_dir,
    newey_west_alpha,
    normalized,
    returns_for_actions,
)


PLUS_ACTIONS = ["vw", "vw_cap", "ew15_vwcap85", "ew25_vwcap75", "ew35_vwcap65", "ew50_vwcap50", "ew"]
PLUS_STRATEGIES = [
    "qpp_single_agent",
    "qpp_meta",
    "maspp_soft_capacity",
    "maspp_regime",
    "maspp_full",
]
STATE_MULT_PLUS = {"Healthy": 1.0, "Warning": 0.55, "Decayed": 0.35}


def safe_z(x: pd.Series, window: int = 120) -> pd.Series:
    mean = x.rolling(window, min_periods=max(24, window // 3)).mean()
    std = x.rolling(window, min_periods=max(24, window // 3)).std()
    return ((x - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sigmoid(x: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def action_ew_share(action: str) -> float:
    if action == "ew":
        return 1.0
    if action == "ew50_vwcap50":
        return 0.50
    if action == "ew35_vwcap65":
        return 0.35
    if action == "ew25_vwcap75":
        return 0.25
    if action == "ew15_vwcap85":
        return 0.15
    return 0.0


def make_action_returns(returns_by_weighting: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = dict(returns_by_weighting)
    if "ew" in out and "vw_cap" in out:
        out["ew15_vwcap85"] = 0.15 * out["ew"] + 0.85 * out["vw_cap"]
        out["ew25_vwcap75"] = 0.25 * out["ew"] + 0.75 * out["vw_cap"]
        out["ew35_vwcap65"] = 0.35 * out["ew"] + 0.65 * out["vw_cap"]
        out["ew50_vwcap50"] = 0.50 * out["ew"] + 0.50 * out["vw_cap"]
    return out


def future_improvement_frames(
    returns_by_action: dict[str, pd.DataFrame],
    horizons: list[int],
    actions: list[str],
) -> pd.DataFrame:
    base = returns_by_action["vw"]
    frames: list[pd.DataFrame] = []
    for action in actions:
        mat = returns_by_action[action]
        df: pd.DataFrame | None = None
        for horizon in horizons:
            base_f = base.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
            action_f = mat.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
            imp = action_f - base_f
            tmp = imp.stack().rename(f"future_improvement_{horizon}").reset_index()
            tmp.columns = ["date", "alpha", f"future_improvement_{horizon}"]
            df = tmp if df is None else df.merge(tmp, on=["date", "alpha"], how="outer")
        assert df is not None
        df["action"] = action
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_trailing_action_features(panel: pd.DataFrame, returns_by_action: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for action in sorted(panel["action"].dropna().unique()):
        mat = returns_by_action[action]
        vol12 = mat.rolling(12, min_periods=8).std()
        vol36 = mat.rolling(36, min_periods=18).std()
        ret36 = mat.rolling(36, min_periods=18).sum()
        downside = mat.mask(mat > 0.0).rolling(12, min_periods=8).mean().fillna(0.0)
        draw = mat.rolling(12, min_periods=8).sum().rolling(12, min_periods=8).min()
        pieces = []
        for name, wide in [
            ("action_vol12", vol12),
            ("action_vol36", vol36),
            ("action_ret36", ret36),
            ("action_downside12", downside),
            ("action_drawdown12", draw),
        ]:
            tmp = wide.stack().rename(name).reset_index()
            tmp.columns = ["date", "alpha", name]
            pieces.append(tmp)
        df = pieces[0]
        for tmp in pieces[1:]:
            df = df.merge(tmp, on=["date", "alpha"], how="outer")
        df["action"] = action
        frames.append(df)
    feat = pd.concat(frames, ignore_index=True)
    out = panel.merge(feat, on=["date", "alpha", "action"], how="left")
    out["ew_share"] = out["action"].map(action_ew_share).fillna(0.0)
    out["is_blend"] = out["action"].str.contains("ew").astype(float)
    out["is_full_ew"] = (out["action"] == "ew").astype(float)
    out["is_low_ew_blend"] = out["action"].isin(["ew15_vwcap85", "ew25_vwcap75"]).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def build_plus_action_panel(
    states: pd.DataFrame,
    returns_by_action: dict[str, pd.DataFrame],
    lookback: int,
    actions: list[str],
) -> pd.DataFrame:
    panel = build_action_value_panel(states, returns_by_action, lookback, 12, actions)
    extra = future_improvement_frames(returns_by_action, [3, 6], actions)
    panel = panel.merge(extra, on=["date", "alpha", "action"], how="left")
    panel = panel.rename(columns={"future_improvement": "future_improvement_12"})
    panel = add_trailing_action_features(panel, returns_by_action, lookback)
    panel["target_utility"] = (
        0.20 * panel["future_improvement_3"].fillna(0.0)
        + 0.30 * panel["future_improvement_6"].fillna(0.0)
        + 0.50 * panel["future_improvement_12"].fillna(0.0)
    )
    panel["success_label"] = (panel["target_utility"] > 0.0025 + 0.0035 * panel["ew_share"]).astype(float)
    panel["tail_label"] = (panel["future_improvement_12"] < -0.035 - 0.01 * panel["ew_share"]).astype(float)
    return panel


def build_regime_features(base: pd.DataFrame, returns_by_action: dict[str, pd.DataFrame], ff6: pd.DataFrame) -> pd.DataFrame:
    idx = base.index
    ff = ff6.set_index("date").reindex(idx)
    port = base.mean(axis=1)
    ew_spread = (returns_by_action.get("ew", base) - base).mean(axis=1)
    vwc_spread = (returns_by_action.get("vw_cap", base) - base).mean(axis=1)
    dispersion = base.std(axis=1)
    corr_vals = []
    for i, date in enumerate(idx):
        if i < 36:
            corr_vals.append(np.nan)
            continue
        corr = base.iloc[i - 36 : i].corr(min_periods=18).to_numpy()
        if corr.shape[0] <= 1:
            corr_vals.append(np.nan)
        else:
            corr_vals.append(float(np.nanmean(corr[np.triu_indices_from(corr, 1)])))
    reg = pd.DataFrame(index=idx)
    reg["alpha_ret12"] = port.rolling(12, min_periods=8).sum()
    reg["alpha_disp12"] = dispersion.rolling(12, min_periods=8).mean()
    reg["ew_vw_spread12"] = ew_spread.rolling(12, min_periods=8).sum()
    reg["vwc_vw_spread12"] = vwc_spread.rolling(12, min_periods=8).sum()
    reg["cross_corr36"] = corr_vals
    reg["market12"] = ff["Mkt-RF"].rolling(12, min_periods=8).sum()
    reg["smb12"] = ff["SMB"].rolling(12, min_periods=8).sum()
    reg["mom12"] = ff["Mom"].rolling(12, min_periods=8).sum()
    for col in list(reg.columns):
        reg[f"z_{col}"] = safe_z(reg[col], 120)
    repair = sigmoid(
        0.65 * reg["z_alpha_ret12"]
        + 0.45 * reg["z_alpha_disp12"]
        + 0.45 * reg["z_ew_vw_spread12"]
        + 0.25 * reg["z_mom12"]
        - 0.55 * reg["z_cross_corr36"]
    )
    risk_off = sigmoid(-0.80 * reg["z_market12"] + 0.85 * reg["z_cross_corr36"] - 0.40 * reg["z_alpha_disp12"])
    reg["repair_opportunity"] = repair.fillna(0.5)
    reg["risk_off"] = risk_off.fillna(0.5)
    reg["ew_budget"] = (0.18 + 0.24 * reg["repair_opportunity"] - 0.15 * reg["risk_off"]).clip(0.10, 0.42)
    reg["tail_gamma"] = (0.75 + 0.90 * reg["risk_off"] - 0.20 * reg["repair_opportunity"]).clip(0.55, 1.75)
    reg["uncertainty_gamma"] = (0.45 + 0.50 * reg["risk_off"]).clip(0.35, 1.10)
    reg["value_aggression"] = (4.5 + 3.0 * reg["repair_opportunity"] - 1.8 * reg["risk_off"]).clip(2.8, 7.5)
    reg["smoothing"] = (0.10 + 0.20 * reg["risk_off"] - 0.06 * reg["repair_opportunity"]).clip(0.05, 0.32)
    reg = reg.reset_index().rename(columns={"index": "date"})
    return reg


def _fit_model_bundle(train: pd.DataFrame, features: list[str], seed: int) -> dict[str, object]:
    x_train_raw = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["target_utility"].fillna(0.0).to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)

    models = [
        Ridge(alpha=20.0),
        HistGradientBoostingRegressor(max_iter=90, learning_rate=0.055, max_leaf_nodes=15, min_samples_leaf=90, l2_regularization=0.10, random_state=seed),
    ]
    for model in models:
        model.fit(x_train, y)
    train_preds = np.mean(np.vstack([np.asarray(model.predict(x_train), dtype=float) for model in models]), axis=0)
    train_resid = y - train_preds
    resid_df = pd.DataFrame({"action": train["action"].to_numpy(), "resid": train_resid})
    global_q = np.nanquantile(train_resid, [0.05, 0.50, 0.95])
    action_q = resid_df.groupby("action")["resid"].quantile([0.05, 0.50, 0.95]).unstack()
    action_q = action_q.rename(columns={0.05: "r05", 0.50: "r50", 0.95: "r95"})
    action_sd = resid_df.groupby("action")["resid"].std().replace(0.0, np.nan).fillna(float(np.nanstd(train_resid)))
    return {
        "features": features,
        "scaler": scaler,
        "models": models,
        "global_q": global_q,
        "action_q": action_q,
        "action_sd": action_sd,
        "global_sd": float(np.nanstd(train_resid)),
    }


def _predict_model_bundle(bundle: dict[str, object], test: pd.DataFrame) -> pd.DataFrame:
    features = bundle["features"]  # type: ignore[assignment]
    out = test[["date", "alpha", "action"]].copy()
    x_test_raw = test[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)  # type: ignore[index]
    scaler: StandardScaler = bundle["scaler"]  # type: ignore[assignment]
    x_test = scaler.transform(x_test_raw)
    models = bundle["models"]  # type: ignore[assignment]
    preds = [np.asarray(model.predict(x_test), dtype=float) for model in models]  # type: ignore[union-attr]
    out["pred_mu"] = np.mean(np.vstack(preds), axis=0)
    out["model_uncertainty"] = np.std(np.vstack(preds), axis=0)
    global_q = bundle["global_q"]  # type: ignore[assignment]
    action_q: pd.DataFrame = bundle["action_q"]  # type: ignore[assignment]
    action_sd: pd.Series = bundle["action_sd"]  # type: ignore[assignment]
    global_sd = float(bundle["global_sd"])
    r05 = test["action"].map(action_q["r05"]).fillna(float(global_q[0])).to_numpy(dtype=float)
    r50 = test["action"].map(action_q["r50"]).fillna(float(global_q[1])).to_numpy(dtype=float)
    r95 = test["action"].map(action_q["r95"]).fillna(float(global_q[2])).to_numpy(dtype=float)
    sigma = test["action"].map(action_sd).fillna(global_sd).to_numpy(dtype=float)
    sigma = np.maximum(np.nan_to_num(sigma, nan=global_sd), 1e-6)
    out["q05"] = out["pred_mu"] + r05
    out["q50"] = out["pred_mu"] + r50
    out["q95"] = out["pred_mu"] + r95
    cost_threshold = 0.0025 + 0.0035 * test["action"].map(action_ew_share).fillna(0.0).to_numpy(dtype=float)
    out["p_win"] = sigmoid((out["pred_mu"].to_numpy(dtype=float) - cost_threshold) / (0.75 * sigma))
    out["p_tail"] = sigmoid((-0.035 - out["q05"].to_numpy(dtype=float)) / (0.90 * sigma))
    return out


def rolling_qpp_predictions(
    panel: pd.DataFrame,
    actions: list[str],
    refit_months: int = 12,
    train_window: int = 180,
    min_train: int = 12000,
    seed: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "action_vol12",
        "action_vol36",
        "action_ret36",
        "action_downside12",
        "action_drawdown12",
        "ew_share",
        "is_blend",
        "is_full_ew",
        "is_low_ew_blend",
    ]
    needed = [*features, "target_utility", "success_label", "tail_label"]
    clean = panel[panel["action"].isin(actions)].dropna(subset=needed).copy()
    all_dates = sorted(panel["date"].drop_duplicates())
    pred_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    last_refit: pd.Timestamp | None = None
    model_cache: tuple[pd.Timestamp, dict[str, object]] | None = None

    for date in all_dates:
        date = pd.Timestamp(date)
        test = panel[(panel["date"] == date) & (panel["action"].isin(actions))].dropna(subset=features).copy()
        if test.empty:
            continue
        should_refit = last_refit is None or (date.year - last_refit.year) * 12 + (date.month - last_refit.month) >= refit_months
        if should_refit:
            train_end = date - pd.DateOffset(months=12)
            train_start = train_end - pd.DateOffset(months=train_window)
            train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)].copy()
            if len(train) < min_train:
                continue
            bundle = _fit_model_bundle(train, features, seed + date.year)
            last_refit = date
            model_cache = (date, bundle)
        else:
            assert model_cache is not None
            bundle = model_cache[1]
        preds = _predict_model_bundle(bundle, test)
        pred_frames.append(preds)
        joined = preds.merge(test[["date", "alpha", "action", "target_utility", "future_improvement_12"]], on=["date", "alpha", "action"], how="left")
        if len(joined) > 20:
            top = joined.sort_values(["alpha", "pred_mu"]).groupby("alpha").tail(1)
            diag_rows.append(
                {
                    "date": date,
                    "n_rows": int(len(joined)),
                    "mean_top_future_utility": float(top["target_utility"].mean()),
                    "top_action_success": float((top["target_utility"] > 0.0).mean()),
                    "top_ew_share": float(top["action"].map(action_ew_share).mean()),
                    "rank_ic_mu_utility": float(joined[["pred_mu", "target_utility"]].corr(method="spearman").iloc[0, 1]),
                    "mean_p_win": float(top["p_win"].mean()),
                }
            )
    pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    diag = pd.DataFrame(diag_rows)
    return pred, diag


def choose_actions_for_date(
    q: pd.DataFrame,
    s: pd.DataFrame,
    reg: pd.Series,
    previous_actions: pd.Series | None,
    strategy: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    q = q.copy()
    q["ew_share"] = q["action"].map(action_ew_share).fillna(0.0)
    low_coverage = s["n_stocks"] <= s["n_stocks"].quantile(0.25)
    high_corr = s["max_abs_corr60"] >= s["max_abs_corr60"].quantile(0.75)
    q["low_coverage"] = q["alpha"].map(low_coverage.to_dict()).fillna(False).astype(bool)
    q["high_corr"] = q["alpha"].map(high_corr.to_dict()).fillna(False).astype(bool)
    q["capacity_charge"] = (
        (0.0015 + 0.0100 * reg["risk_off"]) * q["ew_share"]
        + (0.0045 + 0.0090 * reg["risk_off"]) * q["ew_share"] * q["low_coverage"].astype(float)
        + 0.0025 * q["ew_share"] * q["high_corr"].astype(float)
    )
    q["tail_charge"] = reg["tail_gamma"] * q["p_tail"].fillna(0.5) * np.maximum(-q["q05"].fillna(0.0), 0.0)
    q["uncertainty_charge"] = reg["uncertainty_gamma"] * q["model_uncertainty"].fillna(0.0)
    q["switch_charge"] = 0.0
    if previous_actions is not None and not previous_actions.empty:
        prev = q["alpha"].map(previous_actions.to_dict())
        q.loc[prev.notna() & (prev != q["action"]), "switch_charge"] = 0.0015
    q["margin_score"] = q["pred_mu"].fillna(0.0) + 0.20 * q["q50"].fillna(0.0) - q["tail_charge"] - q["uncertainty_charge"]
    p0 = 0.50 + 0.12 * reg["risk_off"] - 0.08 * reg["repair_opportunity"]

    if strategy == "qpp_single_agent":
        q["score"] = q["pred_mu"].fillna(0.0) - 0.003 * q["ew_share"]
    elif strategy == "qpp_meta":
        q["score"] = q["margin_score"] - 0.0025 * q["ew_share"]
        q.loc[(q["p_win"] < p0) & (q["action"] != "vw"), "score"] -= 0.020
    elif strategy == "maspp_soft_capacity":
        q["score"] = q["margin_score"] - 0.75 * q["capacity_charge"]
        q.loc[(q["p_win"] < p0 + 0.02) & (q["action"] != "vw"), "score"] -= 0.015
    elif strategy == "maspp_regime":
        q["score"] = (
            reg["value_aggression"] * q["pred_mu"].fillna(0.0)
            + 0.35 * q["q50"].fillna(0.0)
            - q["tail_charge"]
            - q["uncertainty_charge"]
            - q["capacity_charge"]
            - q["switch_charge"]
        )
        q.loc[(q["p_win"] < p0 + 0.01) & (q["action"] != "vw"), "score"] -= 0.012
    else:
        conflict_discount = 0.003 * q["high_corr"].astype(float) + 0.002 * q["low_coverage"].astype(float)
        q["score"] = (
            reg["value_aggression"] * q["pred_mu"].fillna(0.0)
            + 0.25 * q["q50"].fillna(0.0)
            + 0.05 * q["q95"].fillna(0.0)
            - q["tail_charge"]
            - q["uncertainty_charge"]
            - 1.15 * q["capacity_charge"]
            - q["switch_charge"]
            - conflict_discount
        )
        q.loc[(q["p_win"] < p0) & (q["action"] != "vw"), "score"] -= 0.014

    best = q.sort_values(["alpha", "score"]).groupby("alpha").tail(1).set_index("alpha")
    # Do not take negative-value non-base actions.
    bad = (best["score"] <= 0.0) & (best["action"] != "vw")
    if bad.any():
        base = q[q["action"] == "vw"].set_index("alpha")
        best.loc[bad, ["action", "score", "ew_share"]] = base.loc[best.index[bad], ["action", "score", "ew_share"]].to_numpy()

    selected = best["action"].astype(object)
    selected_score = best["score"].astype(float)

    # Greedy soft EW budget: downgrade lowest benefit-per-EW choices first.
    if strategy in {"maspp_soft_capacity", "maspp_regime", "maspp_full"}:
        base_weight_proxy = s["health_score"].rank(pct=True).reindex(selected.index).fillna(0.5)
        ew_exposure = float((base_weight_proxy * selected.map(action_ew_share)).sum() / base_weight_proxy.sum())
        budget = float(reg["ew_budget"])
        if ew_exposure > budget:
            q2 = q.set_index(["alpha", "action"])
            candidates = []
            for alpha, action in selected.items():
                ew = action_ew_share(str(action))
                if ew <= 0:
                    continue
                alpha_rows = q[q["alpha"] == alpha].sort_values("score", ascending=False)
                lower = alpha_rows[alpha_rows["ew_share"] < ew]
                if lower.empty:
                    continue
                alt = lower.iloc[0]
                cur_score = float(q2.loc[(alpha, action), "score"])
                loss = cur_score - float(alt["score"])
                candidates.append((loss / max(ew - float(alt["ew_share"]), 1e-6), alpha, str(alt["action"]), float(alt["score"])))
            for _, alpha, alt_action, alt_score in sorted(candidates):
                selected.loc[alpha] = alt_action
                selected_score.loc[alpha] = alt_score
                ew_exposure = float((base_weight_proxy * selected.map(action_ew_share)).sum() / base_weight_proxy.sum())
                if ew_exposure <= budget:
                    break
    return selected, selected_score, q


def mas_plus_backtest(
    states: pd.DataFrame,
    residual_states: pd.DataFrame,
    gates: pd.DataFrame,
    returns_by_action: dict[str, pd.DataFrame],
    qpred: pd.DataFrame,
    regime: pd.DataFrame,
    lookback: int,
    oos_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = returns_by_action["vw"]
    state_idx = states.set_index(["date", "alpha"]).sort_index()
    res_idx = residual_states.set_index(["date", "alpha"]).sort_index()
    q_idx = qpred.set_index("date").sort_index()
    reg_idx = regime.set_index("date").sort_index()
    prev_weights: dict[str, pd.Series] = {}
    prev_actions: dict[str, pd.Series] = {}
    records: list[dict[str, object]] = []
    action_records: list[dict[str, object]] = []

    for idx, date in enumerate(base.index[:-1]):
        if date < pd.Timestamp(oos_date):
            continue
        if date not in state_idx.index.get_level_values(0) or date not in q_idx.index:
            continue
        next_date = base.index[idx + 1]
        ret_next = base.loc[next_date].dropna()
        names = ret_next.index.intersection(state_idx.loc[date].index)
        if date in res_idx.index.get_level_values(0):
            names = names.intersection(res_idx.loc[date].index)
        if len(names) == 0:
            continue
        s = state_idx.loc[date].loc[names]
        rs = res_idx.loc[date].loc[names] if date in res_idx.index.get_level_values(0) else s.copy()
        q = q_idx.loc[[date]].reset_index()
        q = q[q["alpha"].isin(names)]
        if q.empty:
            continue
        reg = reg_idx.loc[date] if date in reg_idx.index else pd.Series({"repair_opportunity": 0.5, "risk_off": 0.5, "ew_budget": 0.24, "tail_gamma": 1.0, "uncertainty_gamma": 0.6, "value_aggression": 5.0, "smoothing": 0.15})
        gate = gates.reindex(index=[date], columns=names).iloc[0].fillna(1.0) if date in gates.index else pd.Series(1.0, index=names)
        state_mult = s["state"].map(STATE_MULT_PLUS).fillna(0.0)
        health_score = (s["health_score"] - s["health_score"].quantile(0.25)).clip(lower=0.0)
        residual_score = (rs["health_score"] - rs["health_score"].quantile(0.25)).clip(lower=0.0)
        base_weight = normalized((0.58 * health_score + 0.42 * residual_score) * state_mult * gate)

        rec: dict[str, object] = {
            "date": next_date,
            "n_alphas": int(len(names)),
            "repair_opportunity": float(reg["repair_opportunity"]),
            "risk_off": float(reg["risk_off"]),
            "ew_budget": float(reg["ew_budget"]),
        }
        for strategy in PLUS_STRATEGIES:
            actions, score, full_q = choose_actions_for_date(q, s, reg, prev_actions.get(strategy), strategy)
            common_names = actions.index.intersection(base_weight.index)
            action_score = score.reindex(common_names).fillna(0.0)
            value_boost = np.exp(np.clip(float(reg["value_aggression"]) * action_score, -0.28, 0.55))
            redundancy_penalty = pd.Series(1.0, index=common_names)
            redundancy_penalty.loc[s.loc[common_names, "max_abs_corr60"] >= s["max_abs_corr60"].quantile(0.80)] = 0.90 if strategy != "qpp_single_agent" else 1.0
            low_cov_penalty = pd.Series(1.0, index=common_names)
            low_cov_penalty.loc[(s.loc[common_names, "n_stocks"] <= s["n_stocks"].quantile(0.20)) & (actions.loc[common_names].map(action_ew_share) > 0.35)] = 0.85
            target_w = normalized(base_weight.loc[common_names] * value_boost * redundancy_penalty * low_cov_penalty)
            if strategy in {"maspp_regime", "maspp_full"} and strategy in prev_weights and not prev_weights[strategy].empty:
                idx_union = target_w.index.union(prev_weights[strategy].index)
                sm = float(reg["smoothing"])
                target_w = normalized((1.0 - sm) * target_w.reindex(idx_union).fillna(0.0) + sm * prev_weights[strategy].reindex(idx_union).fillna(0.0))
            realized = returns_for_actions(returns_by_action, next_date, actions)
            common = realized.index.intersection(target_w.index)
            ww = normalized(target_w.loc[common])
            rec[strategy] = float((realized.loc[common] * ww).sum())
            rec[f"{strategy}_turnover"] = calc_turnover(ww, prev_weights.get(strategy))
            rec[f"{strategy}_switch_rate"] = calc_switch_rate(actions.loc[common], prev_actions.get(strategy))
            rec[f"{strategy}_ew_exposure"] = float((ww * actions.loc[common].map(action_ew_share)).sum())
            rec[f"{strategy}_full_ew_weight"] = float(ww.loc[actions.loc[common] == "ew"].sum()) if (actions.loc[common] == "ew").any() else 0.0
            prev_weights[strategy] = ww
            prev_actions[strategy] = actions.loc[common]
            action_records.append(
                {
                    "date": next_date,
                    "strategy": strategy,
                    "ew_exposure": rec[f"{strategy}_ew_exposure"],
                    "full_ew_weight": rec[f"{strategy}_full_ew_weight"],
                    "avg_action_score": float(action_score.reindex(common).mean()),
                    "avg_p_win": float(full_q[full_q["alpha"].isin(common)].groupby("alpha")["p_win"].max().mean()),
                    "avg_model_uncertainty": float(full_q[full_q["alpha"].isin(common)].groupby("alpha")["model_uncertainty"].mean().mean()),
                }
            )
        records.append(rec)
    return pd.DataFrame(records), pd.DataFrame(action_records)


def downside_metrics(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {"calmar": np.nan, "es_5": np.nan, "worst_month": np.nan}
    stats = annualized_stats(ret)
    mdd = stats["max_drawdown"]
    calmar = stats["ann_return"] / abs(mdd) if np.isfinite(mdd) and mdd < 0 else np.nan
    q = ret.quantile(0.05)
    return {"calmar": float(calmar), "es_5": float(ret[ret <= q].mean()), "worst_month": float(ret.min())}


def summarize_all(port: pd.DataFrame, ff6: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
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
        ff = newey_west_alpha(net, ff6)
        row = {
            "strategy": strategy,
            "net_sharpe": nstats["sharpe"],
            "stress_sharpe": sstats["sharpe"],
            "extreme_sharpe": estats["sharpe"],
            "ann_return": nstats["ann_return"],
            "ann_vol": nstats["ann_vol"],
            "max_drawdown": nstats["max_drawdown"],
            **downside_metrics(net),
            "avg_turnover": float(port[f"{strategy}_turnover"].mean()),
            "avg_switch": float(port[f"{strategy}_switch_rate"].mean()),
            "ff6_alpha_ann": ff["ff6_alpha_ann"],
            "ff6_alpha_tstat": ff["ff6_alpha_tstat"],
            "n_months": int(net.dropna().shape[0]),
        }
        if f"{strategy}_ew_exposure" in port:
            row["avg_ew_exposure"] = float(port[f"{strategy}_ew_exposure"].mean())
            row["avg_full_ew_weight"] = float(port[f"{strategy}_full_ew_weight"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


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
                net = pd.Series(apply_costs(sub.reset_index(), strategy, 10.0, 2.0).to_numpy(), index=sub.index)
                rows.append({"period": label, "strategy": strategy, **annualized_stats(net), **downside_metrics(net)})
    return pd.DataFrame(rows)


def action_selection_diagnostics(qpred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    joined = qpred.merge(
        panel[["date", "alpha", "action", "target_utility", "future_improvement_12", "success_label", "tail_label"]],
        on=["date", "alpha", "action"],
        how="left",
    ).dropna(subset=["target_utility"])
    if joined.empty:
        return pd.DataFrame()
    best = joined.sort_values(["date", "alpha", "pred_mu"]).groupby(["date", "alpha"]).tail(1).copy()
    best["ew_share"] = best["action"].map(action_ew_share)
    rows = []
    for name, df in [("all_best_mu", best), ("accepted_pwin_gt_55", best[best["p_win"] > 0.55]), ("high_conf_margin", best[(best["pred_mu"] - best["model_uncertainty"]) > 0])]:
        if df.empty:
            continue
        rows.append(
            {
                "selection": name,
                "n": int(len(df)),
                "mean_future_utility": float(df["target_utility"].mean()),
                "median_future_utility": float(df["target_utility"].median()),
                "success_rate": float((df["target_utility"] > 0).mean()),
                "tail_rate": float(df["tail_label"].mean()),
                "mean_ew_share": float(df["ew_share"].mean()),
                "rank_ic_pred_utility": float(joined[["pred_mu", "target_utility"]].corr(method="spearman").iloc[0, 1]),
            }
        )
    return pd.DataFrame(rows)


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

    ff_cache = next(Path("outputs").glob("**/external_ff6_monthly.csv"), None)
    if ff_cache is not None:
        ff6 = pd.read_csv(ff_cache, parse_dates=["date"])
        ff6.to_csv(out_dir / "external_ff6_monthly.csv", index=False)
    else:
        ff6 = load_ff6(out_dir)

    base = returns_by_weighting["vw"]
    states = build_states(base, long_by_weighting["vw"], args.lookback)
    residual = rolling_ff6_residuals(base, ff6, args.residual_lookback)
    residual_states = build_states(residual, long_by_weighting["vw"], args.lookback)
    gates = rolling_regime_gate(base, ff6, args.regime_lookback)
    returns_by_action = make_action_returns(returns_by_weighting)
    actions = [a for a in PLUS_ACTIONS if a in returns_by_action]

    print(f"[mas++] building action panel for {len(actions)} actions", flush=True)
    panel = build_plus_action_panel(states, returns_by_action, args.lookback, actions)
    panel.to_parquet(out_dir / "maspp_action_panel.parquet", index=False)
    print(f"[mas++] action panel rows={len(panel):,}", flush=True)
    regime = build_regime_features(base, returns_by_action, ff6)
    regime.to_csv(out_dir / "maspp_regime_features.csv", index=False)
    print("[mas++] fitting rolling Q++ ensemble", flush=True)
    qpred, qdiag = rolling_qpp_predictions(panel, actions, refit_months=args.refit_months, train_window=args.train_window, min_train=args.min_train, seed=args.seed)
    qpred.to_parquet(out_dir / "maspp_q_predictions.parquet", index=False)
    qdiag.to_csv(out_dir / "maspp_q_diagnostics_by_month.csv", index=False)
    action_diag = action_selection_diagnostics(qpred, panel)
    action_diag.to_csv(out_dir / "maspp_action_selection_diagnostics.csv", index=False)

    print(f"[mas++] Q predictions rows={len(qpred):,}", flush=True)
    print("[mas++] backtesting MAS++ policies", flush=True)
    plus_port, plus_actions = mas_plus_backtest(states, residual_states, gates, returns_by_action, qpred, regime, args.lookback, OOS_DATE)
    plus_port.to_csv(out_dir / "maspp_policy_portfolio_returns.csv", index=False)
    plus_actions.to_csv(out_dir / "maspp_action_records.csv", index=False)

    print("[mas++] rerunning baseline MAS policies", flush=True)
    old_messages = build_action_message_panel(states, returns_by_weighting, args.lookback, available_actions(returns_by_weighting, ACTION_SET), horizon=12)
    old_port, old_conflicts, old_reliability = mas_policy_backtest(states, residual_states, gates, returns_by_weighting, old_messages, args.lookback, OOS_DATE)
    old_port.to_csv(out_dir / "baseline_mas_policy_portfolio_returns.csv", index=False)
    old_messages.to_parquet(out_dir / "baseline_agent_messages.parquet", index=False)

    merged = old_port.merge(plus_port, on=["date"], how="inner", suffixes=("", "_plusdup"))
    merged.to_csv(out_dir / "maspp_merged_portfolio_returns.csv", index=False)
    strategies = [s for s in MAS_STRATEGIES if s in merged.columns] + PLUS_STRATEGIES
    summary = summarize_all(merged, ff6, strategies)
    summary.to_csv(out_dir / "maspp_strategy_summary.csv", index=False)
    sub = subperiod_summary(merged, ["single_agent_q", "centralized_constrained_q", "full_mas", *PLUS_STRATEGIES])
    sub.to_csv(out_dir / "maspp_subperiod_summary.csv", index=False)
    disagree = disagreement_table(merged, ["single_agent_q", "centralized_constrained_q", "full_mas", "maspp_full"])
    disagree.to_csv(out_dir / "maspp_disagreement_buckets.csv", index=False)
    impl_attr = implementation_attribution(old_messages, returns_by_weighting)
    impl_attr.to_csv(out_dir / "baseline_implementation_attribution.csv", index=False)

    metrics = {
        "out_dir": str(out_dir),
        "n_q_predictions": int(len(qpred)),
        "n_action_panel": int(len(panel)),
        "actions": actions,
        "qdiag_mean_top_future_utility": float(qdiag["mean_top_future_utility"].mean()) if not qdiag.empty else None,
        "qdiag_top_action_success": float(qdiag["top_action_success"].mean()) if not qdiag.empty else None,
        "mean_repair_opportunity": float(regime["repair_opportunity"].mean()),
        "mean_risk_off": float(regime["risk_off"].mean()),
    }
    (out_dir / "maspp_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default="outputs/alphalife_mas_plus")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--residual-lookback", type=int, default=120)
    parser.add_argument("--regime-lookback", type=int, default=120)
    parser.add_argument("--train-window", type=int, default=180)
    parser.add_argument("--refit-months", type=int, default=12)
    parser.add_argument("--min-train", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
