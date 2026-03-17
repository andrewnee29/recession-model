# ============================================================
# src/data.py
# FRED data fetching and feature engineering
# ============================================================

import pandas as pd
import numpy as np
from fredapi import Fred

START_DATE = "1990-01-01"

SERIES = {
    "USREC":        "NBER Recession Indicator",
    "T10Y2Y":       "Yield Curve (10Y-2Y)",
    "UNRATE":       "Unemployment Rate",
    "SAHMREALTIME": "Sahm Rule Indicator",
    "PAYEMS":       "Nonfarm Payrolls",
    "PCEPILFE":     "Core PCE Inflation",
    "UMCSENT":      "Consumer Sentiment",
    "DRCCLACBS":    "Credit Card Delinquency Rate",
    "FEDFUNDS":     "Federal Funds Rate",
}


def fetch_fred_data(api_key: str) -> pd.DataFrame:
    """
    Pull all FRED series and align to a monthly DatetimeIndex.

    Args:
        api_key: FRED API key (free at fred.stlouisfed.org)

    Returns:
        DataFrame with one column per series, monthly frequency
    """
    fred = Fred(api_key=api_key)
    frames = {}

    for sid, name in SERIES.items():
        print(f"  Fetching {name}...")
        s = fred.get_series(sid, observation_start=START_DATE)
        frames[sid] = s.resample("MS").last()   # align to month start

    df = pd.DataFrame(frames)

    # diagnostic — show earliest available date per series
    print("\nEarliest available date per series:")
    for col in df.columns:
        first = df[col].first_valid_index()
        print(f"  {col:15s} {first}")

    df = df.dropna(subset=["USREC"])
    print(f"\nRaw data: {df.shape[0]} months, {df.shape[1]} series")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged and derived features from raw FRED data.

    All features are lagged >= 1 month to prevent data leakage.
    Target variable: recession onset within next 3 months.

    Args:
        df: Raw DataFrame from fetch_fred_data()

    Returns:
        Feature DataFrame including 'recession' target column
    """
    feat = pd.DataFrame(index=df.index)

    # --- Yield curve ---
    feat["yield_curve"]        = df["T10Y2Y"].shift(1)
    feat["yield_curve_3m_chg"] = df["T10Y2Y"].diff(3).shift(1)
    feat["yield_inverted"]     = (df["T10Y2Y"] < 0).astype(int).shift(1)

    # --- Labor market ---
    feat["unemployment"]       = df["UNRATE"].shift(1)
    feat["unemp_3m_chg"]       = df["UNRATE"].diff(3).shift(1)
    feat["sahm_rule"]          = df["SAHMREALTIME"].shift(1)

    # Payroll growth YoY (%)
    payroll_yoy                = df["PAYEMS"].pct_change(12) * 100
    feat["payroll_yoy"]        = payroll_yoy.shift(1)
    feat["payroll_negative"]   = (payroll_yoy < 0).astype(int).shift(1)

    # --- Inflation ---
    feat["core_pce_yoy"]       = df["PCEPILFE"].pct_change(12, fill_method=None).shift(1) * 100

    # --- Consumer sentiment ---
    feat["consumer_sentiment"] = df["UMCSENT"].shift(1)
    feat["sentiment_6m_chg"]   = df["UMCSENT"].diff(6).shift(1)

    # --- Credit stress ---
    feat["cc_delinquency"]     = df["DRCCLACBS"].shift(1)
    feat["cc_delinq_chg"]      = df["DRCCLACBS"].diff(3).shift(1)

    # --- Fed policy ---
    feat["fed_funds"]          = df["FEDFUNDS"].shift(1)
    feat["fed_funds_chg"]      = df["FEDFUNDS"].diff(6).shift(1)

    # Target: does a recession begin within the next 6 months?
    # 6-month window gives the model more lead time to learn pre-recession patterns
    feat["recession"]          = df["USREC"].rolling(6).max().shift(-6)

    feat = feat.dropna()
    print(f"Feature matrix: {feat.shape[0]} months, {feat.shape[1]-1} features")
    return feat


def get_feature_cols(feat: pd.DataFrame) -> list:
    """Return list of feature column names, excluding target."""
    return [c for c in feat.columns if c != "recession"]