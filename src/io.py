import pandas as pd


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # drop unnamed leading column if present
    if df.columns[0].lower().startswith('unnamed') or df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])

    required = {'open','high','low','close','volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    date_col = None
    for c in ['Date','date','timestamp','time','datetime']:
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        raise ValueError('No Date/time column found')

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])

    return df
