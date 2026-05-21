#!/usr/bin/env python3
"""Validate that the committed result snapshots contain the expected tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = pd.read_csv(root / "results" / "maspp_state_control_summary.csv")
    required = {
        "qpp_exact",
        "full_mas_exact",
        "maspp_fixed_b06",
        "maspp_global_dynamic",
        "maspp_cluster_dynamic",
        "maspp_cluster_reliability",
        "central_cluster_dynamic",
        "inversevol_cluster_dynamic",
    }
    found = set(summary["strategy"])
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(f"Missing result rows: {missing}")
    cluster = summary.loc[summary["strategy"] == "maspp_cluster_dynamic"].iloc[0]
    if cluster["net_sharpe"] <= 0:
        raise RuntimeError("Cluster Dynamic snapshot has a non-positive Sharpe ratio.")
    print("Result snapshots verified.")


if __name__ == "__main__":
    main()
