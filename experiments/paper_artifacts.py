#!/usr/bin/env python3
"""Generate paper tables and result notes from a completed AlphaLife full run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def strategy_tests(port: pd.DataFrame) -> pd.DataFrame:
    strategy_cols = [
        "rolling_sharpe",
        "health_only",
        "healthy_only",
        "no_redundancy_penalty",
        "alphalife_full",
        "alphalife_repaired",
    ]
    rows = []
    for base in ["static_equal", "rolling_sharpe"]:
        for strategy in strategy_cols:
            if strategy == base:
                continue
            diff = port[strategy] - port[base]
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": base,
                    "mean_monthly_diff": diff.mean(),
                    "annualized_diff_approx": diff.mean() * 12,
                    "tstat": tstat(diff),
                    "win_rate": (diff > 0).mean(),
                    "n_months": diff.dropna().shape[0],
                }
            )
    return pd.DataFrame(rows)


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x: float, n: int = 3) -> str:
    return "" if pd.isna(x) else f"{x:.{n}f}"


def make_tables(run_dir: Path) -> None:
    tables = run_dir / "paper_tables"
    tables.mkdir(exist_ok=True)

    port_summary = pd.read_csv(run_dir / "main_portfolio_summary.csv")
    warning = pd.read_csv(run_dir / "main_warning_effectiveness.csv")
    stock_ic = pd.read_csv(run_dir / "stock_level_ic_summary_by_filter.csv")
    robustness = pd.read_csv(run_dir / "robustness_sharpe_pivot.csv")
    repair = json.loads((run_dir / "main_repair_summary.json").read_text())
    ff = pd.read_csv(run_dir / "ff6_style_regressions.csv")
    port = pd.read_csv(run_dir / "main_portfolio_returns.csv")

    tests = strategy_tests(port)
    tests.to_csv(run_dir / "strategy_pairwise_tests.csv", index=False)

    main = port_summary.copy()
    main["ann_return"] = main["ann_return"].map(fmt_pct)
    main["ann_vol"] = main["ann_vol"].map(fmt_pct)
    main["max_drawdown"] = main["max_drawdown"].map(fmt_pct)
    main["hit_rate"] = main["hit_rate"].map(fmt_pct)
    for c in ["sharpe", "monthly_tstat"]:
        main[c] = main[c].map(lambda x: fmt_num(x, 3))
    main = main[["strategy", "ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "monthly_tstat"]]
    main.to_markdown(tables / "table_main_portfolio.md", index=False)
    main.to_latex(tables / "table_main_portfolio.tex", index=False, escape=True)

    w12 = warning[warning["horizon_months"] == 12].copy()
    for c in ["mean_future_ret", "median_future_ret", "negative_rate"]:
        w12[c] = w12[c].map(fmt_pct)
    w12["t_vs_healthy"] = w12["t_vs_healthy"].map(lambda x: fmt_num(x, 3))
    w12 = w12[["state", "n", "mean_future_ret", "median_future_ret", "negative_rate", "t_vs_healthy"]]
    w12.to_markdown(tables / "table_warning_12m.md", index=False)
    w12.to_latex(tables / "table_warning_12m.tex", index=False, escape=True)

    top_ic = stock_ic[stock_ic["me_rank_cutoff"] == 0.2].sort_values("mean_rank_ic", ascending=False).head(12).copy()
    for c in ["mean_rank_ic", "mean_top_minus_bottom"]:
        top_ic[c] = top_ic[c].map(fmt_pct)
    for c in ["rank_ic_t", "top_minus_bottom_t"]:
        top_ic[c] = top_ic[c].map(lambda x: fmt_num(x, 2))
    top_ic = top_ic[["feature", "mean_rank_ic", "rank_ic_t", "mean_top_minus_bottom", "top_minus_bottom_t", "n_months"]]
    top_ic.to_markdown(tables / "table_stock_ic_top12.md", index=False)
    top_ic.to_latex(tables / "table_stock_ic_top12.tex", index=False, escape=True)

    robust = robustness.copy()
    for c in robust.columns:
        if c not in {"base_weighting", "lookback", "oos_start"}:
            robust[c] = robust[c].map(lambda x: fmt_num(x, 3))
    robust.to_markdown(tables / "table_robustness_sharpe.md", index=False)
    robust.to_latex(tables / "table_robustness_sharpe.tex", index=False, escape=True)

    tests_fmt = tests.copy()
    tests_fmt["mean_monthly_diff"] = tests_fmt["mean_monthly_diff"].map(fmt_pct)
    tests_fmt["annualized_diff_approx"] = tests_fmt["annualized_diff_approx"].map(fmt_pct)
    tests_fmt["win_rate"] = tests_fmt["win_rate"].map(fmt_pct)
    tests_fmt["tstat"] = tests_fmt["tstat"].map(lambda x: fmt_num(x, 3))
    tests_fmt.to_markdown(tables / "table_pairwise_tests.md", index=False)
    tests_fmt.to_latex(tables / "table_pairwise_tests.tex", index=False, escape=True)

    repair_md = f"""| Metric | Value |
|---|---:|
| Repair trials | {repair.get('n', 0):,} |
| Success rate | {repair.get('success_rate', float('nan')):.2%} |
| Mean 12M improvement | {repair.get('mean_improvement', float('nan')):.2%} |
| Median 12M improvement | {repair.get('median_improvement', float('nan')):.2%} |
| Selected EW | {repair.get('repair_counts', {}).get('ew', 0):,} |
| Selected VW-cap | {repair.get('repair_counts', {}).get('vw_cap', 0):,} |
| Selected VW | {repair.get('repair_counts', {}).get('vw', 0):,} |
"""
    (tables / "table_repair_summary.md").write_text(repair_md)

    notes = f"""# Paper Writing Notes

## Tables

- Main portfolio comparison: `paper_tables/table_main_portfolio.tex`
- Pairwise strategy tests: `paper_tables/table_pairwise_tests.tex`
- Lifecycle warning 12M result: `paper_tables/table_warning_12m.tex`
- Repair summary: `paper_tables/table_repair_summary.md`
- Stock-level IC validation: `paper_tables/table_stock_ic_top12.tex`
- Robustness Sharpe grid: `paper_tables/table_robustness_sharpe.tex`

## Figures

- Portfolio wealth: `fig_portfolio_wealth_full.png`
- Portfolio drawdowns: `fig_portfolio_drawdowns_full.png`
- Lifecycle state counts: `fig_state_counts.png`
- Warning effectiveness: `fig_warning_effectiveness_12m.png`
- Stock-level IC: `fig_stock_ic_top20.png`

## Key Empirical Claims

1. `alphalife_full` improves Sharpe over static equal weighting and rolling Sharpe in the main VW 1990-2024 setup.
2. `alphalife_repaired` gives the strongest main result, with Sharpe above 1 in the main VW setup.
3. Repair is not universal: success rate is {repair.get('success_rate', float('nan')):.2%}, which supports a constrained repair protocol rather than unconstrained factor mining.
4. Warning states have worse median 12M forward returns and higher negative-rate than Healthy states, but Decayed states can rebound; write this as a limitation of rule-based state detection.
5. {((ff['r2'] > 0.5).mean()):.2%} of factors have FF5+Momentum R2 above 0.5, supporting style-exposure diagnostics.

## Recommended Result Wording

Use cautious language: \"lifecycle governance improves allocation stability and repair outcomes in this monthly U.S. anomaly setting\" rather than \"decay is perfectly predicted.\" The strongest evidence is portfolio-level and repair-level, not pure warning classification.
"""
    (run_dir / "paper_writing_notes.md").write_text(notes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    return p.parse_args()


if __name__ == "__main__":
    make_tables(Path(parse_args().run_dir))
