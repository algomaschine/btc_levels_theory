import numpy as np
import pandas as pd

from .modeling import fit_logit, predict_logit, fit_xgb_classifier


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feats, model='logit', seed: int = 7):
    # Train once on train_df; predict once on test_df.
    tr = train_df.copy()
    te = test_df.copy()

    tr = tr[(tr['is_reject']==1)|(tr['is_break']==1)].copy()
    te = te[(te['is_reject']==1)|(te['is_break']==1)].copy()

    tr['y'] = tr['is_reject'].astype(int)
    te['y'] = te['is_reject'].astype(int)

    for c in feats:
        tr[c] = pd.to_numeric(tr[c], errors='coerce')
        te[c] = pd.to_numeric(te[c], errors='coerce')

    tr = tr.dropna(subset=feats+['y'])
    te = te.dropna(subset=feats+['y'])

    if len(tr) < 5000 or len(te) < 500:
        return pd.DataFrame(), {'error': f'Not enough data (train={len(tr)}, test={len(te)})'}

    Xtr = tr[feats].values.astype(float)
    ytr = tr['y'].values.astype(int)
    Xte = te[feats].values.astype(float)
    yte = te['y'].values.astype(int)

    if model=='logit':
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd==0]=1.0
        Xtrz=(Xtr-mu)/sd
        Xtez=(Xte-mu)/sd
        w,b = fit_logit(Xtrz, ytr.astype(float))
        p = predict_logit(Xtez, w, b)
    else:
        clf = fit_xgb_classifier(Xtr, ytr, seed=seed)
        p = clf.predict_proba(Xte)[:,1]

    eps=1e-12
    brier=float(np.mean((p-yte)**2))
    ll=float(-np.mean(yte*np.log(p+eps) + (1-yte)*np.log(1-p+eps)))

    out = te[['time','t1','close','level','tf','is_break','is_reject','y']].copy()
    out['p_reject']=p

    return out, {
        'train_n': int(len(tr)),
        'test_n': int(len(te)),
        'brier': brier,
        'logloss': ll,
        'p_reject_mean': float(np.mean(p)),
        'y_mean': float(np.mean(yte)),
        'feats': feats,
        'model': model,
    }
