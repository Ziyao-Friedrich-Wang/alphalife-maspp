# AlphaLife-MAS++

AlphaLife-MAS++ is a reproducible research codebase for auditable alpha
lifecycle governance. The project evaluates a library of already-discovered
quantitative alphas as dynamic research assets: each alpha can be monitored,
repaired, capacity-limited, down-weighted, or routed through a governance
endpoint.

The main model is the **Cluster Dynamic Governor**. It allocates
repair-risk capacity across alpha clusters under structured governance
constraints. The return-seeking endpoint is a Repair++ action-value policy;
the governance endpoint is a Full MAS policy with liquidity, risk,
challenge, and audit controls. The empirical claim is not that the system
discovers new alphas. The claim is that cluster-level repair-risk capacity
allocation improves the realized risk-adjusted frontier in a factor-return
proxy setting.

## Repository layout

```text
experiments/              Core experiment scripts
scripts/                  Data generation, smoke tests, and run wrappers
results/                  Compact result snapshots for verification
data/README.md            Expected empirical data layout
```

Large raw datasets, virtual environments, run outputs, logs, and credentials
are intentionally excluded.

## Installation

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The main empirical scripts also download Fama-French factor files from the
Ken French data library when a cached `external_ff6_monthly.csv` file is not
available.

## Smoke test without external data

The smoke test creates a deterministic synthetic factor-return panel and runs
the MVP lifecycle allocation experiment. It is intended to verify that the
environment, paths, and core computation work.

```bash
python scripts/generate_synthetic_data.py --out data/synthetic
python scripts/run_smoke_test.py --data-root data/synthetic
```

The generated files and test outputs are ignored by git.

## Reproducing the MAS++ result snapshots

Place the empirical JKP-style monthly U.S. anomaly factor-return files under
`data/jkp` as described in `data/README.md`, then run:

```bash
bash scripts/run_maspp_pipeline.sh data/jkp outputs/alphalife_mas_plus
```

This performs two stages:

1. `experiments/alphalife_mas_plus.py` trains the Repair++ and governance
   endpoint policies and writes intermediate artifacts.
2. `experiments/alphalife_mas_plus_state_control.py` builds the fixed,
   global, cluster-dynamic, reliability, and non-MAS cluster-capacity
   comparisons.

The key output files are written into the latest timestamped directory under
`outputs/alphalife_mas_plus/`.

To verify the committed result snapshots:

```bash
python scripts/verify_result_snapshots.py
```

## Data requirements

The main experiment expects monthly factor-return files with columns:

```text
date,name,ret,n_stocks
```

The empirical evaluation uses value-weighted, equal-weighted, and value-weighted
with capacity proxy implementations. The repository does not ship the raw
empirical panel because it is large and may be subject to redistribution
constraints.

## Security and credentials

No external service credentials, local absolute data paths, or cloud provider
credentials are required or stored in this repository. If an
environment-specific data path is needed, pass it explicitly through
`--data-root` or set `ALPHALIFE_DATA_ROOT`.

## Limitations

The included empirical code works at the factor-return level. Turnover,
implementation switching, and transaction costs are proxy diagnostics rather
than holdings-level execution costs. Live deployment would require
reconstructing underlying holdings, stock-level turnover, bid-ask spreads,
borrow constraints, impact costs, and AUM capacity curves.
