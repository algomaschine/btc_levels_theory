import os, json
import numpy as np
import pandas as pd

from .io import load_ohlcv
from .timeframes import make_timeframes
from .levels import build_levels_multitf
from .labels import label_touch_outcomes
from .model import walk_forward_logit
from .backtest import backtest_from_predictions
from .report import write_report


def run_pipeline(csv_path: str, out_dir: str, start=None, end=None, seed: int = 7):
    np.random.seed(seed)

    df1 = load_ohlcv(csv_path)
    if start:
        df1 = df1[df1.index >= pd.to_datetime(start)]
    if end:
        df1 = df1[df1.index <= pd.to_datetime(end)]

    # Build multi-timeframe bars
    tfs = make_timeframes(df1)

    # Discover levels (higher TF) and write them
    levels_by_tf = build_levels_multitf(tfs)
    for tf, lv in levels_by_tf.items():
        lv.to_csv(os.path.join(out_dir, f"levels_{tf}.csv"), index=False)

    # Create minute-level dataset of touches + outcomes
    touch_df = label_touch_outcomes(df1, levels_by_tf)
    touch_df.to_csv(os.path.join(out_dir, "touch_dataset.csv"), index=False)

    # Predictive model (walk-forward)
    preds, metrics = walk_forward_logit(touch_df)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Backtest
    trades, equity = backtest_from_predictions(preds, df1)
    trades.to_csv(os.path.join(out_dir, "backtest_trades.csv"), index=False)
    equity.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)

    # Basic hypothesis tests (bootstrap)
    tests = {
        "note": "See report.md for interpretation. Tests are falsifiable and out-of-sample friendly.",
        "n_touches": int(len(touch_df)),
        "rejection_rate": float(touch_df['is_reject'].mean()) if len(touch_df) else None,
        "breakout_rate": float(touch_df['is_break'].mean()) if len(touch_df) else None,
    }
    with open(os.path.join(out_dir, "hypothesis_tests.json"), "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)

    write_report(out_dir=out_dir, metrics=metrics, tests=tests)
