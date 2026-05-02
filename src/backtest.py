import numpy as np
import pandas as pd


def backtest_from_predictions(preds: pd.DataFrame, df1: pd.DataFrame,
                              p_enter: float = 0.60,
                              horizon_min: int = 60,
                              fee_bps: float = 4.0):
    # Minimal illustrative strategy:
    # - If p_reject >= p_enter: trade for rejection (mean reversion away from level)
    # - Else if p_reject <= 1 - p_enter: trade for breakout
    # Uses close-to-close execution (conservative assumptions recommended in real work).

    if preds is None or len(preds) == 0:
        return pd.DataFrame(), pd.DataFrame()

    preds = preds.copy()
    preds["time"] = pd.to_datetime(preds["time"])

    close = df1["close"].copy()
    close.index = pd.to_datetime(close.index)

    fee = fee_bps / 10000.0

    trades = []
    eq = []
    equity = 1.0

    # Build quick lookup for future close
    close_map = close

    for _, r in preds.iterrows():
        t0 = r["time"]
        if t0 not in close_map.index:
            continue
        i0 = close_map.index.get_loc(t0)
        i1 = i0 + horizon_min
        if i1 >= len(close_map):
            continue

        p0 = float(close_map.iloc[i0])
        p1 = float(close_map.iloc[i1])
        lvl = float(r["level"])

        p_rej = float(r["p_reject"])
        side = "below" if p0 < lvl else "above"

        direction = None
        style = None

        if p_rej >= p_enter:
            style = "reject"
            # if below level and expecting reject, go short near level; if above, go long
            direction = -1 if side == "below" else +1
        elif p_rej <= (1 - p_enter):
            style = "break"
            # breakout: if below, go long; if above, go short
            direction = +1 if side == "below" else -1
        else:
            continue

        ret = direction * (p1 / p0 - 1.0) - fee
        equity *= (1.0 + ret)

        trades.append({
            "time": t0,
            "style": style,
            "direction": int(direction),
            "p0": p0,
            "p1": p1,
            "ret": float(ret),
            "equity": float(equity),
            "p_reject": p_rej,
        })
        eq.append({"time": t0, "equity": float(equity)})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(eq)
    return trades_df, eq_df
