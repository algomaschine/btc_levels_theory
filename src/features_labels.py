import numpy as np
import pandas as pd


def _nearest_level(price: float, levels: np.ndarray):
    if len(levels)==0:
        return None, None
    j = int(np.argmin(np.abs(levels-price)))
    return float(levels[j]), float(abs(levels[j]-price))


def build_touch_dataset(df1: pd.DataFrame, levels_by_tf: dict, horizon_min: int = 60):
    closes = df1['close'].values
    highs = df1['high'].values
    lows = df1['low'].values
    idx = df1.index

    # simple regime features from higher timeframe-like smoothing
    ma_fast = pd.Series(closes, index=idx).rolling(240).mean()   # 4h on 1m
    ma_slow = pd.Series(closes, index=idx).rolling(1440).mean()  # 1d on 1m
    trend = (ma_fast - ma_slow) / ma_slow

    # realized vol
    ret1 = np.diff(closes, prepend=closes[0]) / np.maximum(1e-12, closes)
    vol_60 = pd.Series(ret1, index=idx).rolling(60).std().bfill().values

    # Prepare levels arrays
    tf_levels, tf_tol, tf_strength = {}, {}, {}
    for tf, lv in levels_by_tf.items():
        if lv is None or len(lv)==0:
            tf_levels[tf]=np.array([]); tf_tol[tf]=0.0; tf_strength[tf]={}
        else:
            tf_levels[tf]=lv['level'].values.astype(float)
            tf_tol[tf]=float(lv['tol'].iloc[0])
            tf_strength[tf]={float(a): float(b) for a,b in zip(lv['level'].values, lv['strength'].values)}

    rows=[]
    for i in range(1, len(df1)-horizon_min-2):
        c=float(closes[i]); h=float(highs[i]); lo=float(lows[i])

        # nearest level across TFs
        best=None
        for tf in tf_levels:
            lvls=tf_levels[tf]
            if len(lvls)==0: continue
            lvl, dist = _nearest_level(c, lvls)
            if lvl is None: continue
            tol = tf_tol[tf] if tf_tol[tf] > 0 else max(1e-9, c*0.001)
            if best is None or dist < best['dist']:
                best={'tf':tf,'level':lvl,'dist':dist,'tol':tol}

        if best is None:
            continue

        lvl=best['level']; tol=best['tol']

        touched = (lo - tol <= lvl <= h + tol)
        if not touched:
            continue

        side = 'below' if c < lvl else 'above'

        future = closes[i+1:i+1+horizon_min]
        if len(future) < horizon_min:
            continue

        if side=='below':
            break_cond = future >= (lvl + tol)
            reject_cond = future <= (lvl - 2*tol)
        else:
            break_cond = future <= (lvl - tol)
            reject_cond = future >= (lvl + 2*tol)

        t_break = np.argmax(break_cond) if break_cond.any() else None
        t_rej = np.argmax(reject_cond) if reject_cond.any() else None

        if t_break is None and t_rej is None:
            continue

        is_break=0; is_reject=0
        if t_break is None:
            is_reject=1
        elif t_rej is None:
            is_break=1
        else:
            if t_break < t_rej:
                is_break=1
            else:
                is_reject=1

        strength = tf_strength.get(best['tf'], {}).get(float(lvl), 0.0)

        rows.append({
            'time': idx[i],
            't1': idx[i] + pd.Timedelta(minutes=horizon_min),
            'close': c,
            'level': lvl,
            'tf': best['tf'],
            'dist': float(abs(c-lvl)),
            'dist_norm': float(abs(c-lvl)/max(1e-9,tol)),
            'side': 1 if side=='below' else -1,
            'trend': float(trend.iloc[i]) if np.isfinite(trend.iloc[i]) else 0.0,
            'vol_60': float(vol_60[i]),
            'is_break': is_break,
            'is_reject': is_reject,
            'level_strength': float(strength),
        })

    out=pd.DataFrame(rows)
    if len(out):
        out['time']=pd.to_datetime(out['time'])
        out['t1']=pd.to_datetime(out['t1'])
    return out
