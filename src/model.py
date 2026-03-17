# ============================================================
# src/model.py
# Model training, evaluation, and prediction
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight


def train_model(feat: pd.DataFrame, feature_cols: list):
    """
    Train a Gradient Boosting classifier using TimeSeriesSplit CV.

    Why Gradient Boosting over Random Forest:
        Recession prediction is a weak signal problem — no single indicator
        dominates. GB's sequential error correction learns subtle interactions
        between indicators better than averaging independent trees (RF).
        Class imbalance (~10% recession months) also favors boosting, as
        sequential weighting naturally focuses on hard-to-classify periods.

    Args:
        feat:         Feature DataFrame including 'recession' target column
        feature_cols: List of feature column names (excludes target)

    Returns:
        model:        Fitted GradientBoostingClassifier
        scaler:       Fitted StandardScaler
        mean_auc:     Mean cross-validation AUC score
        auc_scores:   AUC score per fold
    """
    X = feat[feature_cols].values
    y = feat["recession"].values

    tscv   = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    # Logistic Regression generalizes better than GB with only 3 recessions
    # in the training data. GB memorizes the specific recession signatures
    # and fails to generalize to new pre-recession environments.
    model  = LogisticRegression(
        class_weight="balanced",  # handles class imbalance
        C=0.1,                    # strong regularization to prevent overfitting
        max_iter=1000,
        random_state=42
    )

    auc_scores = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  Fold {fold} skipped (single class)")
            continue
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        model.fit(X_tr, y[tr])
        prob = model.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y[te], prob)
        auc_scores.append(auc)

        # diagnostic — show fold composition
        n_rec_tr = y[tr].sum()
        n_rec_te = y[te].sum()
        print(f"  Fold {fold} | train: {len(tr)} months, {int(n_rec_tr)} recession | "
              f"test: {len(te)} months, {int(n_rec_te)} recession | AUC: {auc:.3f}")

    mean_auc  = float(np.mean(auc_scores))
    print(f"\nMean CV AUC: {mean_auc:.3f} ± {np.std(auc_scores):.3f}")

    # final fit on all data
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    return model, scaler, mean_auc, auc_scores


def predict_proba_series(feat: pd.DataFrame, feature_cols: list,
                          model, scaler) -> np.ndarray:
    """
    Generate recession probability for every month in the feature DataFrame.

    Args:
        feat:         Feature DataFrame
        feature_cols: Feature column names
        model:        Fitted model
        scaler:       Fitted scaler

    Returns:
        Array of probabilities aligned to feat.index
    """
    X = scaler.transform(feat[feature_cols].values)
    return model.predict_proba(X)[:, 1]


def current_probability(feat: pd.DataFrame, feature_cols: list,
                         model, scaler) -> float:
    """
    Print and return the model's current recession probability.

    Args:
        feat:         Feature DataFrame (uses last row)
        feature_cols: Feature column names
        model:        Fitted model
        scaler:       Fitted scaler

    Returns:
        Recession probability as a float between 0 and 1
    """
    latest       = feat[feature_cols].iloc[-1]
    date_str     = feat.index[-1].strftime("%B %Y")

    # diagnostic — compare current values to historical pre-recession averages
    pre_rec = feat[feat["recession"] == 1][feature_cols]
    print("\n── Current vs Pre-Recession Averages ──")
    for col in feature_cols:
        print(f"  {col:25s} now={latest[col]:7.2f}  pre-rec avg={pre_rec[col].mean():7.2f}")

    latest_scaled = scaler.transform(latest.values.reshape(1, -1))
    prob          = model.predict_proba(latest_scaled)[0, 1]

    print("\n" + "=" * 45)
    print(f"  CURRENT RECESSION PROBABILITY: {prob:.1%}")
    print(f"  As of: {date_str}")
    print("=" * 45)

    if prob > 0.60:
        print("  Signal: HIGH RISK ⚠️")
    elif prob > 0.35:
        print("  Signal: ELEVATED RISK 🟡")
    else:
        print("  Signal: LOW RISK 🟢")

    return prob


def feature_importance_df(model, feature_cols: list) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances.

    Args:
        model:        Fitted GradientBoostingClassifier
        feature_cols: Feature column names

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending
    """
    # LogisticRegression uses coef_, GradientBoosting uses feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])  # abs value since coef can be negative

    return (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )