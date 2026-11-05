"""Cross-validation that every model goes through identically."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

N_FOLDS, SEED = 5, 42

METRICS = {
    "ROC-AUC": roc_auc_score,            # ranking: P(score of a defaulter > score of a payer)
    "PR-AUC": average_precision_score,   # ranking, focused on the rare positive class
    "Brier": brier_score_loss,           # mean squared error of the probabilities (lower = better)
    "LogLoss": log_loss,                 # penalises confident wrong probabilities (lower = better)
}


def folds(y: pd.Series):
    """The same 5 stratified splits for every model."""
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def run_cv(make_model, X: pd.DataFrame, y: pd.Series):
    """Train on 4 folds, predict the 5th, repeat. Returns per-fold metrics and the
    out-of-fold (OOF) predictions: one honest prediction for every training row."""
    oof = np.zeros(len(y))
    rows = []
    for k, (tr, va) in enumerate(folds(y)):
        t = time.time()
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append({"fold": k, **{m: f(y.iloc[va], p) for m, f in METRICS.items()},
                     "seconds": time.time() - t})
    return pd.DataFrame(rows), oof
