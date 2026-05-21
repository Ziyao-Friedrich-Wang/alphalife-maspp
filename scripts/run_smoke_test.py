#!/usr/bin/env python3
"""Run a deterministic end-to-end smoke test on synthetic data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_command(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/smoke_mvp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / args.data_root).exists():
        run_command([sys.executable, "scripts/generate_synthetic_data.py", "--out", str(args.data_root)], root)
    run_command(
        [
            sys.executable,
            "experiments/alphalife_mvp.py",
            "--data-root",
            str(args.data_root),
            "--out-dir",
            str(args.out_dir),
            "--lookback",
            "60",
        ],
        root,
    )
    latest = sorted((root / args.out_dir).glob("*"))[-1]
    summary = pd.read_csv(latest / "portfolio_summary.csv")
    required = {"static_equal_weight", "rolling_sharpe_weight", "alphalife_lifecycle"}
    found = set(summary["strategy"])
    missing = required - found
    if missing:
        raise RuntimeError(f"Smoke test missing strategies: {sorted(missing)}")
    print(f"Smoke test completed: {latest}")


if __name__ == "__main__":
    main()
