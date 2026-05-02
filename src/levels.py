import numpy as np
import pandas as pd


def _pivots(df: pd.DataFrame, k: int = 3):
    h = df['high'].values
    l = df['low'].values
    piv_hi = np.zeros(len(df), dtype=bool)
    piv_lo = np.zeros(len(df), dtype=bool)

    for i in range(k, len(df)-k):
        if h[i] == np.max(h[i-k:i+k+1]):
            piv_hi[i] = True
        if l[i] == np.min(l[i-k:i+k+1]):
            piv_lo[i] = True

    out=[]
    for p,t in zip(h[piv_hi], df.index[piv_hi]):
        out.append({'time':t,'price':float(p),'kind':'pivot_high'})
    for p,t in zip(l[piv_lo], df.index[piv_lo]):
        out.append({'time':t,'price':float(p),'kind':'pivot_low'})
    return pd.DataFrame(out)


def _volume_profile_peaks(df: pd.DataFrame, bins: int = 250, topk: int = 15):
    price = (df['high'] + df['low'] + df['close'])/3.0
    vol = df['volume'].fillna(0.0).values

    pmin, pmax = float(np.nanmin(price)), float(np.nanmax(price))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return pd.DataFrame(columns=['price','strength','kind'])

    edges = np.linspace(pmin, pmax, bins+1)
    centers = (edges[:-1] + edges[1:])/2
    inds = np.clip(np.digitize(price.values, edges)-1, 0, bins-1)

    vp = np.zeros(bins)
    for i,v in zip(inds, vol):
        vp[i] += v

    peaks=[]
    for i in range(1, bins-1):
        if vp[i] > vp[i-1] and vp[i] > vp[i+1]:
            peaks.append(i)

    if not peaks:
        return pd.DataFrame(columns=['price','strength','kind'])

    peaks = sorted(peaks, key=lambda i: vp[i], reverse=True)[:min(topk,len(peaks))]
    total = vp.sum() if vp.sum() > 0 else 1.0
    out=[]
    for i in peaks:
        out.append({'price':float(centers[i]), 'strength':float(vp[i]/total), 'kind':'vp_peak'})
    return pd.DataFrame(out)


def _cluster_prices(prices: np.ndarray, tol: float):
    prices = np.array([p for p in prices if np.isfinite(p)], dtype=float)
    if len(prices)==0:
        return np.array([]), np.array([])
    prices.sort()
    clusters=[]
    cur=[prices[0]]
    for p in prices[1:]:
        if abs(p - np.mean(cur)) <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur=[p]
    clusters.append(cur)
    centers=np.array([float(np.mean(c)) for c in clusters])
    sizes=np.array([len(c) for c in clusters], dtype=float)
    strengths = sizes/sizes.sum() if sizes.sum()>0 else sizes
    return centers, strengths


def build_levels_for_tf(df: pd.DataFrame, tf_name: str):
    piv = _pivots(df, k=3)
    vp = _volume_profile_peaks(df, bins=250)

    tr = (df['high'] - df['low']).median()
    tol = float(tr*0.75) if np.isfinite(tr) and tr>0 else float(df['close'].median()*0.001)

    prices=[]
    if len(piv): prices += list(piv['price'].values)
    if len(vp): prices += list(vp['price'].values)

    centers, strengths = _cluster_prices(np.array(prices), tol=tol)
    out = pd.DataFrame({'tf': tf_name, 'level': centers, 'strength': strengths, 'tol': tol})
    out = out.sort_values('strength', ascending=False).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out)+1)
    return out


def build_levels_multitf(tfs: dict):
    levels_by_tf={}
    for tf in ['15m','1h','4h','1d']:
        df = tfs.get(tf)
        if df is None or len(df) < 50:
            levels_by_tf[tf] = pd.DataFrame(columns=['tf','level','strength','tol','rank'])
        else:
            levels_by_tf[tf] = build_levels_for_tf(df, tf)
    return levels_by_tf
