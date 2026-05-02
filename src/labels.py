import numpy as np
import pandas as pd


def _nearest_level(price: float, levels: np.ndarray):
    if len(levels) == 0:
        return None, None
    j = int(np.argmin(np.abs(levels - price)))
    return float(levels[j]), float(abs(levels[j] - price))


def label_touch_outcomes(df1: pd.DataFrame, levels_by_tf: dict, horizon_min: int = 60):
    # Build a dataset of touch events with features and labels
    # Touch definition: level within bar range expanded by tolerance

    # Flatten levels and include tf strength as feature
    rows = []
    closes = df1["close"].values
    highs = df1["high"].values
    lows = df1["low"].values
    idx = df1.index

    # Precompute global vol proxy
    ret1 = np.diff(np.log(closes), prepend=np.log(closes[0]))
    vol_30 = pd.Series(ret1, index=idx).rolling(30).std().fillna(method="bfill").values

    # Prepare per-tf arrays
    tf_levels = {}
    tf_tol = {}
    tf_strength = {}
    for tf, lv in levels_by_tf.items():
        lv = lv.copy()
        if len(lv) == 0:
            tf_levels[tf] = np.array([])
            tf_tol[tf] = 0.0
            tf_strength[tf] = {}
            continue
        tf_levels[tf] = lv["level"].values.astype(float)
        tf_tol[tf] = float(lv["tol"].iloc[0]) if "tol" in lv.columns and len(lv) else 0.0
        tf_strength[tf] = {float(a): float(b) for a, b in zip(lv["level"].values, lv["strength"].values)}

    for i in range(1, len(df1) - horizon_min - 1):
        c = float(closes[i])
        h = float(highs[i])
        lo = float(lows[i])
        v = float(df1["volume"].iloc[i]) if "volume" in df1.columns else 0.0

        # Find nearest level among all TFs
        best = None
        for tf in tf_levels:
            lvls = tf_levels[tf]
            if len(lvls) == 0:
                continue
            lvl, dist = _nearest_level(c, lvls)
            if lvl is None:
                continue
            tol = tf_tol[tf]
            if best is None or dist < best["dist"]:
                best = {"tf": tf, "level": lvl, "dist": dist, "tol": tol}

        if best is None:
            continue

        lvl = best["level"]
        tol = best["tol"] if best["tol"] > 0 else max(1e-9, c * 0.001)

        # Is this bar a touch?
        touched = (lo - tol <= lvl <= h + tol)
        if not touched:
            continue

        # Outcome definitions (falsifiable):
        # - breakout: within horizon, close exceeds level by +tol (if approaching from below) or below by -tol (from above)
        # - reject: within horizon, price moves away at least 2*tolerance in opposite direction without breaking first
        side = "below" if c < lvl else "above"

        future = closes[i+1:i+1+horizon_min]
        if len(future) < horizon_min:
            continue

        break_cond = None
        reject_cond = None
        if side == "below":
            break_cond = future >= (lvl + tol)
            reject_cond = future <= (lvl - 2*tol)
        else:
            break_cond = future <= (lvl - tol)
            reject_cond = future >= (lvl + 2*tol)

        # Determine which happens first
        t_break = np.argmax(break_cond) if break_cond.any() else None
        t_rej = np.argmax(reject_cond) if reject_cond.any() else None

        is_break = 0
        is_reject = 0
        if t_break is None and t_rej is None:
            # unresolved
            continue
        elif t_break is None:
            is_reject = 1
        elif t_rej is None:
            is_break = 1
        else:
            if t_break < t_rej:
                is_break = 1
            else:
                is_reject = 1

        # Features
        strength = tf_strength.get(best["tf"], {}).get(float(lvl), 0.0)
        rows.append({
            "time": idx[i],
            "close": c,
            "level": lvl,
            "tf": best["tf"],
            "dist": float(abs(c - lvl)),
            "dist_norm": float(abs(c - lvl) / max(1e-9, tol)),
            "side": 1 if side == "below" else -1,
            "vol": v,
            "ret1": float(np.log(closes[i]) - np.log(closes[i-1])),
            "vol_30": float(vol_30[i]),
            "level_strength": float(strength),
            "is_break": is_break,
            "is_reject": is_reject,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["time"] = pd.to_datetime(out["time"])
    return out
