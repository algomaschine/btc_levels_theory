#!/usr/bin/env python3
import argparse, zipfile, json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outzip', required=True)
    args = ap.parse_args()

    with zipfile.ZipFile(args.outzip,'r') as z:
        names=set(z.namelist())
        # find output dir prefix
        prefix = ''
        for cand in ['output_v3/','output_v2/','output/']:
            if any(n.startswith(cand) for n in names):
                prefix=cand
                break
        if not prefix:
            raise SystemExit('Could not find output folder in zip')

        folds = json.load(z.open(prefix+'fold_metrics.json'))
        preds = pd.read_csv(z.open(prefix+'predictions.csv'))
        trades = pd.read_csv(z.open(prefix+'backtest_trades.csv'))

    preds['time']=pd.to_datetime(preds['time'])
    trades['time']=pd.to_datetime(trades['time'])
    trades['exit_time']=pd.to_datetime(trades['exit_time'])

    bounds={int(f['fold_id']):(pd.to_datetime(f['train_end']), pd.to_datetime(f['test_end'])) for f in folds['folds']}

    # Check in-sample predictions
    bad_preds=0
    for fid,(tr_end, te_end) in bounds.items():
        sub=preds[preds['fold_id']==fid]
        bad_preds += int((sub['time']<=tr_end).sum())

    # Check trades outside test
    bad_tr=0
    for fid,(tr_end, te_end) in bounds.items():
        sub=trades[trades['fold_id']==fid]
        bad_tr += int(((sub['time']<=tr_end) | (sub['exit_time']>te_end)).sum())

    total_preds=len(preds)
    total_trades=len(trades)

    print('Total preds:', total_preds)
    print('In-sample preds:', bad_preds, 'pct', (bad_preds/total_preds if total_preds else None))
    print('Total trades:', total_trades)
    print('Bad trades (entered before train_end or exit after test_end):', bad_tr, 'pct', (bad_tr/total_trades if total_trades else None))

if __name__=='__main__':
    main()
