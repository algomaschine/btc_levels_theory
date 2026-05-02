import pandas as pd


def load_ohlcv(path: str) -> pd.DataFrame:
    # Flexible loader: supports an extra unnamed first column
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Handle possible unnamed index col
    if df.columns[0] in {"", "Unnamed: 0", "Unnamed: 0.1"}:
        df = df.drop(columns=[df.columns[0]])

    # Required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Date column name variants
    date_col = None
    for c in ["Date", "date", "timestamp", "time", "datetime"]:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        raise ValueError("No Date/time column found (expected one of Date/date/timestamp/time/datetime)")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    # Cast numerics
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]) 

    return df
