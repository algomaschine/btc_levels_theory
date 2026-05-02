import os, json
import numpy as np
import pandas as pd

from .io import load_ohlcv
from .timeframes import make_timeframes
from .levels import build_levels_multitf
from .features_labels import build_touch_dataset
from .fracdiff_features import fracdiff_series
from .model_online import fit_predict
from .backtest import backtest_non_overlapping
from .report_html import build_html_report


def run_pipeline_online(csv_path: str, out_dir: str, start=None, end=None, seed: int = 7,
                        horizon_min: int = 60, level_lookback_days: int = 180,
                        model: str = 'logit', fd_d: float = 0.4, fd_thresh: float = 1e-3,
                        p_enter: float = 0.65,
                        train_frac: float = 0.6, step_frac: float = 0.1):

    np.random.seed(seed)
    df1 = load_ohlcv(csv_path)
    if start:
        df1 = df1[df1.index >= pd.to_datetime(start)]
    if end:
        df1 = df1[df1.index <= pd.to_datetime(end)]

    df1 = df1.copy()
    df1['fd_close'] = fracdiff_series(df1['close'], d=fd_d, thresh=fd_thresh)

    times = df1.index
    n = len(times)
    train_size = int(n * train_frac)
    step = int(n * step_frac)

    all_preds=[]
    fold_metrics={'folds': []}
    levels_rows=[]

    fold_id=0
    start_i=0
    while start_i + train_size + 5000 < n:
        train_end_i = start_i + train_size
        test_end_i = min(train_end_i + step, n)

        train_end_time = times[train_end_i-1]
        test_start_time = times[train_end_i]
        test_end_time = times[test_end_i-1]

        # levels built from historical window only
        lb_start = train_end_time - pd.Timedelta(days=level_lookback_days)
        hist = df1[(df1.index >= lb_start) & (df1.index <= train_end_time)]
        tfs = make_timeframes(hist)
        levels_by_tf = build_levels_multitf(tfs)

        for tf, lv in levels_by_tf.items():
            if lv is None or len(lv)==0:
                continue
            tmp=lv.copy()
            tmp['fold_id']=fold_id
            tmp['levels_end_time']=train_end_time
            levels_rows.append(tmp)

        # build touch dataset on segment needed for labels
        # We include from (train_end - horizon) so labels inside train can be computed,
        # but we will enforce label-availability constraints below.
        seg_start = times[start_i]
        seg_end = test_end_time
        seg = df1[(df1.index >= seg_start) & (df1.index <= seg_end)].copy()

        touch = build_touch_dataset(seg, levels_by_tf, horizon_min=horizon_min)
        if len(touch) < 5000:
            fold_metrics['folds'].append({
                'fold_id': fold_id,
                'train_end': str(train_end_time),
                'test_end': str(test_end_time),
                'error': 'not enough touches'
            })
            start_i += step
            fold_id += 1
            continue

        # join fd_close feature by timestamp
        touch = touch.sort_values('time')
        touch['fd_close'] = touch['time'].map(seg['fd_close'])

        # IMPORTANT: online availability constraint
        # at training time train_end_time, you only know labels for samples whose t1 <= train_end_time
        train_touch = touch[(touch['t1'] <= train_end_time)].copy()
        # strictly future test chunk
        test_touch = touch[(touch['time'] > train_end_time) & (touch['time'] <= test_end_time)].copy()
        # also keep test labels in-period (optional but cleaner for analysis)
        test_touch = test_touch[test_touch['t1'] <= test_end_time]

        feats = ['dist_norm','side','trend','vol_60','level_strength','fd_close']

        preds, met = fit_predict(train_touch, test_touch, feats=feats, model=model, seed=seed)
        if len(preds)==0:
            fold_metrics['folds'].append({
                'fold_id': fold_id,
                'train_end': str(train_end_time),
                'test_end': str(test_end_time),
                'error': met.get('error','no preds')
            })
        else:
            preds['fold_id']=fold_id
            all_preds.append(preds)
            fold_metrics['folds'].append({
                'fold_id': fold_id,
                'train_end': str(train_end_time),
                'test_start': str(test_start_time),
                'test_end': str(test_end_time),
                'metrics': met,
            })

        start_i += step
        fold_id += 1

    if not all_preds:
        raise RuntimeError('No out-of-sample predictions produced. Try adjusting train_frac/step_frac.')

    pred_all = pd.concat(all_preds, ignore_index=True).sort_values('time')
    pred_all.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    # Backtest ONLY on these OOS predictions; attach fold_id to trades.
    trades_all=[]
    eq_all=[]
    equity=1.0

    # to avoid overlapping trades across folds, we run in time order with a single next_free_time
    close = df1['close'].copy(); close.index = pd.to_datetime(close.index)
    next_free_time = close.index.min()

    for fid, grp in pred_all.groupby('fold_id'):
        # fold bounds
        fb = next((f for f in fold_metrics['folds'] if f.get('fold_id')==int(fid) and f.get('test_end')), None)
        test_end_time = pd.to_datetime(fb['test_end']) if fb else None

        g = grp.sort_values('time').copy()
        # filter: entry after next_free_time (global) and exit within fold test_end
        # We implement exit constraint in backtest by slicing preds such that time + horizon <= test_end in pipeline.

        # Use existing backtest_non_overlapping but it has its own next_free_time internal.
        # We'll implement minimal non-overlap here with global next_free_time.
        horizon = horizon_min
        fee = 4.0/10000.0

        for _, r in g.iterrows():
            t0 = pd.to_datetime(r['time'])
            if t0 < next_free_time:
                continue
            if t0 not in close.index:
                continue
            i0 = close.index.get_loc(t0)
            i1 = i0 + horizon
            if i1 >= len(close):
                break
            exit_t = close.index[i1]
            if test_end_time is not None and exit_t > test_end_time:
                continue

            p0=float(close.iloc[i0]); p1=float(close.iloc[i1])
            lvl=float(r['level']); p_rej=float(r['p_reject'])

            style=None
            if p_rej >= p_enter:
                style='reject'
            elif p_rej <= 1-p_enter:
                style='break'
            else:
                continue

            side = 'below' if p0 < lvl else 'above'
            if style=='reject':
                direction = -1 if side=='below' else +1
            else:
                direction = +1 if side=='below' else -1

            ret = direction*(p1/p0 - 1.0) - fee
            equity *= (1.0 + ret)

            trades_all.append({
                'time': t0,
                'exit_time': exit_t,
                'fold_id': int(fid),
                'style': style,
                'direction': int(direction),
                'p0': p0,
                'p1': p1,
                'ret': float(ret),
                'equity': float(equity),
                'p_reject': p_rej,
            })
            eq_all.append({'time': exit_t, 'equity': float(equity)})
            next_free_time = exit_t

    trades_df = pd.DataFrame(trades_all)
    eq_df = pd.DataFrame(eq_all)

    trades_df.to_csv(os.path.join(out_dir, 'backtest_trades.csv'), index=False)
    eq_df.to_csv(os.path.join(out_dir, 'equity_curve.csv'), index=False)

    if levels_rows:
        lv_all = pd.concat(levels_rows, ignore_index=True)
        lv_all.to_csv(os.path.join(out_dir, 'levels_walkforward.csv'), index=False)

    with open(os.path.join(out_dir, 'fold_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_metrics, f, indent=2)

    # HTML report
    html = build_html_report(out_dir, df1, pred_all, trades_df, eq_df, fold_metrics)
    with open(os.path.join(out_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)
