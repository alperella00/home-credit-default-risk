"""Baseline model definitions. Fixed, reasonable settings; no tuning (see docs).

Each model gets the preprocessing that is natural for its family:
* Logistic regression needs numbers everywhere -> impute, scale, one-hot.
* Random forest (sklearn >= 1.4) accepts NaN natively -> only one-hot categories.
* LightGBM accepts NaN and pandas categories natively -> no preprocessing.
Everything lives inside a Pipeline, so imputation medians / scaling are learned
on the training folds only and never see the validation fold (no leakage).
"""
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
numeric = make_column_selector(dtype_include="number")
categorical = make_column_selector(dtype_include="category")


def _onehot():
    # Rare levels (< 1% of rows) are merged into one "infrequent" column.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                         sparse_output=False)


def logreg(missing_indicators: bool):
    """(a) median imputation only, or (b) median + one 0/1 'was missing' column per feature."""
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=missing_indicators),
                        StandardScaler())
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"), _onehot())
    pre = ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))


def random_forest():
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"), _onehot())
    pre = ColumnTransformer([("num", "passthrough", numeric), ("cat", cat, categorical)])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt",
                                n_jobs=-1, random_state=SEED)
    return make_pipeline(pre, rf)


def lightgbm():
    return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                          min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1)


MODELS = {
    "LogReg (median)": lambda: logreg(missing_indicators=False),
    "LogReg (median + missing flags)": lambda: logreg(missing_indicators=True),
    "Random Forest": random_forest,
    "LightGBM": lightgbm,
}
