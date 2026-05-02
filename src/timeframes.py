import pandas as pd


def make_timeframes(df1: pd.DataFrame):
    # df1 is 1-minute indexed
    # We will build higher TF OHLCV with standard resampling
    rules = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

    tfs = {"1m": df1}
    for name, rule in rules.items():
        ohlc = df1[["open", "high", "low", "close"]].resample(rule, label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        vol = df1[["volume"]].resample(rule, label="right", closed="right").sum()
        tf = ohlc.join(vol, how="left")
        tf = tf.dropna(subset=["open", "high", "low", "close"])
        tfs[name] = tf
    return tfs
