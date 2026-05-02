#!/usr/bin/env python3
import argparse, os
from src.pipeline_online import run_pipeline_online


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./output_v3')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--horizon_min', type=int, default=60)
    ap.add_argument('--level_lookback_days', type=int, default=180)
    ap.add_argument('--model', choices=['logit','xgb'], default='logit')
    ap.add_argument('--fd_d', type=float, default=0.4)
    ap.add_argument('--fd_thresh', type=float, default=1e-3)
    ap.add_argument('--p_enter', type=float, default=0.65)
    ap.add_argument('--train_frac', type=float, default=0.6)
    ap.add_argument('--step_frac', type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_pipeline_online(
        csv_path=args.csv,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        seed=args.seed,
        horizon_min=args.horizon_min,
        level_lookback_days=args.level_lookback_days,
        model=args.model,
        fd_d=args.fd_d,
        fd_thresh=args.fd_thresh,
        p_enter=args.p_enter,
        train_frac=args.train_frac,
        step_frac=args.step_frac,
    )

if __name__=='__main__':
    main()
