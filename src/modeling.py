import numpy as np
import pandas as pd


def _sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0/(1.0+np.exp(-z))


def fit_logit(X, y, l2=2.0, steps=500, lr=0.05):
    n,d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X@w + b)
        gw = (X.T@(p-y))/n + (l2*w)/n
        gb = float(np.mean(p-y))
        w -= lr*gw
        b -= lr*gb
    return w,b


def predict_logit(X, w, b):
    return _sigmoid(X@w + b)


def fit_xgb_classifier(Xtr, ytr, seed=7):
    import xgboost as xgb  # optional
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        eval_metric='logloss'
    )
    clf.fit(Xtr, ytr)
    return clf


def walk_forward_predict(df: pd.DataFrame, feats, model='logit', embargo_min: int = 60, seed: int = 7):
    from .purging import purged_train_indices

    d = df.copy()
    d = d[(d['is_reject']==1)|(d['is_break']==1)].copy()
    d['y'] = d['is_reject'].astype(int)

    for c in feats:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=feats+['y','time','t1']).reset_index(drop=True)

    if len(d) < 5000:
        return pd.DataFrame(), {'error':'Not enough samples'}

    n = len(d)
    train_size = int(n*0.6)
    step = int(n*0.1)

    preds=[]
    metrics={'windows':[], 'features': feats, 'model': model}
    embargo = pd.Timedelta(minutes=embargo_min)

    start = 0
    while start + train_size + 200 < n:
        tr_idx = np.arange(start, start+train_size)
        te_idx = np.arange(start+train_size, min(start+train_size+step, n))
        if len(te_idx) < 200:
            break

        # purging + embargo
        keep_tr = purged_train_indices(d, tr_idx, te_idx, embargo=embargo)
        tr = d.loc[keep_tr]
        te = d.loc[te_idx]

        Xtr = tr[feats].values.astype(float)
        ytr = tr['y'].values.astype(int)
        Xte = te[feats].values.astype(float)
        yte = te['y'].values.astype(int)

        # standardize for logit (not for xgb)
        if model=='logit':
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0); sd[sd==0]=1.0
            Xtrz=(Xtr-mu)/sd
            Xtez=(Xte-mu)/sd
            w,b = fit_logit(Xtrz, ytr.astype(float))
            p = predict_logit(Xtez, w, b)
        else:
            try:
                clf = fit_xgb_classifier(Xtr, ytr, seed=seed)
                p = clf.predict_proba(Xte)[:,1]
            except Exception as e:
                return pd.DataFrame(), {'error': f'xgb not available/failed: {e}'}

        eps=1e-12
        brier=float(np.mean((p-yte)**2))
        ll=float(-np.mean(yte*np.log(p+eps) + (1-yte)*np.log(1-p+eps)))

        out = te[['time','t1','close','level','tf','is_break','is_reject','y']].copy()
        out['p_reject']=p
        preds.append(out)

        metrics['windows'].append({
            'start_idx': int(start),
            'train_n': int(len(tr)),
            'test_n': int(len(te)),
            'brier': brier,
            'logloss': ll,
            'p_reject_mean': float(np.mean(p)),
            'y_mean': float(np.mean(yte)),
        })

        start += step

    if not preds:
        return pd.DataFrame(), {'error':'No windows'}

    pr = pd.concat(preds, ignore_index=True)
    metrics['brier_mean'] = float(np.mean([w['brier'] for w in metrics['windows']]))
    metrics['logloss_mean'] = float(np.mean([w['logloss'] for w in metrics['windows']]))
    metrics['n_predictions'] = int(len(pr))
    return pr, metrics
