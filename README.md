# btc_levels_theory_v3 (strict online walk-forward)

This version fixes a critical issue discovered in `output_v2.zip`: **many predictions and trades were generated on periods that were inside the training window** (in-sample), which can inflate performance.

## What v3 guarantees
For each fold:
1. Levels are computed **only from history up to train_end**.
2. Training samples include only rows whose label end time `t1` is **<= train_end** (labels fully known at training time).
3. Predictions are generated **only for the future test chunk** `(train_end, test_end]`.
4. Backtest trades are taken **only inside the test chunk**, and trades are required to finish within the test chunk (`exit_time <= test_end`).

## Run
```bash
python run.py --csv BTCUSDT_1min_binance.csv --out ./output_v3 --model logit
# optional (if installed)
python run.py --csv BTCUSDT_1min_binance.csv --out ./output_v3 --model xgb
```

## Validate leakage
```bash
python validate_output.py --outzip ./output_v3.zip
```

If you want smaller/more frequent updates (true online), reduce `--step_frac`.
