# Data Layout

The repository does not include the empirical factor-return panel used in the
paper. To reproduce the reported tables, place JKP-style monthly U.S. anomaly
factor return files under a local data root with this layout:

```text
data/jkp/
  factor_returns/
    all_stocks/
      usa/
        monthly/
          vw/
            [usa]_[all_factors]_[monthly]_[vw].zip
          ew/
            [usa]_[all_factors]_[monthly]_[ew].zip
          vw_cap/
            [usa]_[all_factors]_[monthly]_[vw_cap].zip
```

Each zip file should contain one CSV with at least these columns:

```text
date,name,ret,n_stocks
```

Dates must be month-end compatible. Returns are monthly decimal returns. The
evaluation in the paper uses histories that start before 1990 and evaluates
out-of-sample months from 1990 through 2024.

Optional stock-level validation files can be placed under:

```text
data/jkp/stock_level_us/panel_parquet/us_stock_month_panel_YYYY.parquet
```

Those files are only needed for the stock-level RankIC diagnostics in the
older full experiment. The MAS++ paper tables use factor-return data.

For a local smoke test that does not require external data, run:

```bash
python scripts/generate_synthetic_data.py --out data/synthetic
python scripts/run_smoke_test.py --data-root data/synthetic
```
