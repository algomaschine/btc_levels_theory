# btc_levels_theory

This package implements a falsifiable theory of support/resistance as **market memory** (Liquidity Potential Model).

## Inputs
- BTCUSDT 1‑minute CSV named `BTCUSDT_1min_binance.csv`
  - Place it in the project root (same folder as `run.py`), or pass `--csv /path/to/file.csv`.

Expected columns (flexible):
- `open, high, low, close, volume, Date` (others ignored)

## Quick start
```bash
python run.py --csv BTCUSDT_1min_binance.csv --out ./output
```

## What it produces
- `output/levels_*.csv` discovered levels per timeframe
- `output/hypothesis_tests.json` bootstrap test results
- `output/model_metrics.json` out-of-sample metrics
- `output/backtest_trades.csv` trades
- `output/report.md` summary report

## Notes
- This code avoids heavy dependencies. It uses pandas/numpy/matplotlib only.
- If you want, you can swap in sklearn/xgboost for more powerful modeling.
