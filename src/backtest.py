import numpy as np
import pandas as pd


def backtest_non_overlapping(preds: pd.DataFrame, df1: pd.DataFrame, horizon_min: int = 60,
                             p_enter: float = 0.65, fee_bps: float = 4.0):
    # Enter at most one trade at a time; exit after fixed horizon.
    # Trade logic matches v1 but avoids compounding overlapping exposure.

    if preds is None or len(preds)==0:
        return pd.DataFrame(), pd.DataFrame()

    fee = fee_bps/10000.0

    close = df1['close'].copy()
    close.index = pd.to_datetime(close.index)

    preds = preds.copy()
    preds['time'] = pd.to_datetime(preds['time'])
    preds = preds.sort_values('time')

    trades=[]
    eq=[]
    equity=1.0
    next_free_time = close.index.min()

    for _, r in preds.iterrows():
        t0 = r['time']
        if t0 < next_free_time:
            continue
        if t0 not in close.index:
            continue
        i0 = close.index.get_loc(t0)
        i1 = i0 + horizon_min
        if i1 >= len(close):
            break

        p0=float(close.iloc[i0])
        p1=float(close.iloc[i1])
        lvl=float(r['level'])
        p_rej=float(r['p_reject'])

        # decide style
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
        trades.append({
            'time': t0,
            'exit_time': close.index[i1],
            'style': style,
            'direction': int(direction),
            'p0': p0,
            'p1': p1,
            'ret': float(ret),
            'equity': float(equity),
            'p_reject': p_rej,
        })
        eq.append({'time': close.index[i1], 'equity': float(equity)})
        next_free_time = close.index[i1]

    return pd.DataFrame(trades), pd.DataFrame(eq)
