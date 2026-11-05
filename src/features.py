"""Feature engineering for the main application table."""
import numpy as np
import pandas as pd

SENTINEL_DAYS_EMPLOYED = 365243  # EDA: code for pensioners / unemployed, not a real duration


def application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, model-agnostic cleaning of application_{train,test}.

    Kept deliberately small for the baseline: the point is to compare model
    families on (almost) raw data, not to win with feature engineering.
    """
    out = df.copy()

    # 1) Sentinel -> NaN + explicit flag (EDA section 3)
    out["IS_PENSIONER"] = (out["DAYS_EMPLOYED"] == SENTINEL_DAYS_EMPLOYED).astype(np.int8)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(SENTINEL_DAYS_EMPLOYED, np.nan)

    # 2) Affordability ratios (EDA section 4)
    out["CREDIT_TO_INCOME"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_TO_INCOME"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]

    return out.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
