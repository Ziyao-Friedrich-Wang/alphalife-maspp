#!/usr/bin/env python3
"""Create a deterministic synthetic factor-return panel for smoke tests."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _write_zip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_name = path.with_suffix(".csv").name
    payload = frame.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, payload)


def make_factor_returns(out: Path, n_alphas: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1963-01-31", "2024-12-31", freq="ME")
    names = [f"alpha_{i:03d}" for i in range(n_alphas)]

    market = rng.normal(0.004, 0.035, len(dates))
    style_1 = rng.normal(0.001, 0.020, len(dates))
    style_2 = rng.normal(0.000, 0.018, len(dates))
    regime = np.sin(np.linspace(0, 9 * np.pi, len(dates)))

    rows_by_weighting: dict[str, list[dict[str, object]]] = {"vw": [], "ew": [], "vw_cap": []}
    for j, name in enumerate(names):
        beta_m = rng.normal(0.20, 0.08)
        beta_1 = rng.normal(0.35 if j % 3 == 0 else -0.10, 0.10)
        beta_2 = rng.normal(0.25 if j % 5 == 0 else 0.00, 0.10)
        alpha_base = 0.0008 + 0.0005 * ((j % 7) - 3) / 3.0
        noise = rng.normal(0.0, 0.025 + 0.003 * (j % 4), len(dates))
        base = alpha_base + beta_m * market + beta_1 * style_1 + beta_2 * style_2 + 0.004 * regime * ((j % 6) - 2.5) / 2.5 + noise

        ew_spread = 0.0015 + 0.003 * (j % 4 == 0) + rng.normal(0.0, 0.012, len(dates))
        cap_spread = 0.0005 + rng.normal(0.0, 0.007, len(dates))
        n_stocks = int(140 + 8 * j + rng.integers(0, 90))
        for date, vw_ret, ew_ret, cap_ret in zip(dates, base, base + ew_spread, base + cap_spread):
            for weighting, value in [("vw", vw_ret), ("ew", ew_ret), ("vw_cap", cap_ret)]:
                rows_by_weighting[weighting].append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "name": name,
                        "ret": float(np.clip(value, -0.35, 0.35)),
                        "n_stocks": n_stocks,
                    }
                )

    for weighting, rows in rows_by_weighting.items():
        path = out / "factor_returns" / "all_stocks" / "usa" / "monthly" / weighting / f"[usa]_[all_factors]_[monthly]_[{weighting}].zip"
        _write_zip_csv(pd.DataFrame(rows), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-alphas", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_factor_returns(args.out, args.n_alphas, args.seed)
    print(args.out)


if __name__ == "__main__":
    main()
