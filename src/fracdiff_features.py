import numpy as np
import pandas as pd


def _weights_ffd(d: float, thresh: float = 1e-3, max_size: int = 10000):
    # Fixed-width fractional differencing weights (De Prado style)
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1], dtype=float)  # reverse for convolution
    return w


def fracdiff_series(x: pd.Series, d: float = 0.4, thresh: float = 1e-3):
    # Try external package first
    try:
        import fracdiff  # type: ignore
        # Many fracdiff packages expose fracdiff.fdiff or fracdiff.fd
        # We'll fall back if API mismatch.
        if hasattr(fracdiff, 'fdiff'):
            y = fracdiff.fdiff(x.values, d=d)
            return pd.Series(y, index=x.index)
    except Exception:
        pass

    w = _weights_ffd(d, thresh=thresh)
    width = len(w)
    if width < 2:
        return x.copy()

    arr = x.values.astype(float)
    out = np.full_like(arr, np.nan, dtype=float)

    for i in range(width-1, len(arr)):
        window = arr[i-width+1:i+1]
        if np.any(~np.isfinite(window)):
            continue
        out[i] = float(np.dot(w, window))

    return pd.Series(out, index=x.index)
