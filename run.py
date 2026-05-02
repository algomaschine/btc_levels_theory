#!/usr/bin/env python3
import argparse, os
from src.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to BTCUSDT_1min_binance.csv")
    ap.add_argument("--out", default="./output", help="Output directory")
    ap.add_argument("--start", default=None, help="Optional start datetime (e.g. 2020-01-01)")
    ap.add_argument("--end", default=None, help="Optional end datetime")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_pipeline(csv_path=args.csv, out_dir=args.out, start=args.start, end=args.end, seed=args.seed)


if __name__ == "__main__":
    main()
