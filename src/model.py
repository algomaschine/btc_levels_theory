import numpy as np
import pandas as pd


def _sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(X, y, l2=1.0, steps=400, lr=0.05):
    # Simple L2-regularized logistic regression via gradient descent
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        # gradients
        gw = (X.T @ (p - y)) / n + l2 * w / n
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb

    return w, b


def predict_logit(X, w, b):
    return _sigmoid(X @ w + b)


def walk_forward_logit(touch_df: pd.DataFrame):
    # Predict probability of rejection vs breakout.
    # We'll model y=1 for reject, 0 for break, and drop other cases.
    df = touch_df.copy()
    df = df[(df["is_reject"] == 1) | (df["is_break"] == 1)].copy()
    if len(df) < 500:
        return pd.DataFrame(), {"error": "Not enough labeled touches"}

    df["y"] = df["is_reject"].astype(int)

    feats = ["dist_norm", "side", "ret1", "vol_30", "level_strength"]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats + ["y"]).reset_index(drop=True)

    # Standardize features on train windows only
    n = len(df)
    train_size = int(n * 0.6)
    step = int(n * 0.1)

    preds = []
    metrics = {"windows": []}

    start = 0
    while start + train_size + 100 < n:
        tr = df.iloc[start:start+train_size]
        te = df.iloc[start+train_size:start+train_size+step]
        if len(te) < 50:
            break

        Xtr = tr[feats].values.astype(float)
        ytr = tr["y"].values.astype(float)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0

        Xtrz = (Xtr - mu) / sd
        w, b = fit_logit(Xtrz, ytr, l2=2.0, steps=500, lr=0.05)

        Xtez = (te[feats].values.astype(float) - mu) / sd
        p = predict_logit(Xtez, w, b)

        out = te[["time", "close", "level", "tf", "is_break", "is_reject", "y"]].copy()
        out["p_reject"] = p
        preds.append(out)

        # simple metrics
        yte = te["y"].values.astype(int)
        # Brier
        brier = float(np.mean((p - yte)**2))
        # logloss
        eps = 1e-9
        ll = float(-np.mean(yte*np.log(p+eps) + (1-yte)*np.log(1-p+eps)))
        metrics["windows"].append({
            "start_idx": int(start),
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "brier": brier,
            "logloss": ll,
            "p_reject_mean": float(np.mean(p)),
            "y_mean": float(np.mean(yte)),
        })

        start += step

    if not preds:
        return pd.DataFrame(), {"error": "No walk-forward windows produced"}

    all_preds = pd.concat(preds, ignore_index=True)

    # aggregate metrics
    briers = [w["brier"] for w in metrics["windows"]]
    lls = [w["logloss"] for w in metrics["windows"]]
    metrics["brier_mean"] = float(np.mean(briers)) if briers else None
    metrics["logloss_mean"] = float(np.mean(lls)) if lls else None
    metrics["n_predictions"] = int(len(all_preds))
    metrics["features"] = feats

    return all_preds, metrics
