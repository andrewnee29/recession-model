# ============================================================
# main.py
# CLI entry point — runs the full pipeline
# Usage: python main.py --api-key YOUR_KEY
# ============================================================

import argparse
import os
from dotenv import load_dotenv
load_dotenv()  # loads FRED_API_KEY from .env into os.environ

from src.data  import fetch_fred_data, engineer_features, get_feature_cols
from src.model import (train_model, predict_proba_series,
                       current_probability, feature_importance_df)
from src.viz   import (plot_recession_probability, plot_indicators,
                       plot_feature_importance)


def parse_args():
    parser = argparse.ArgumentParser(
        description="US Recession Probability Model"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("FRED_API_KEY"),
        help="FRED API key (or set FRED_API_KEY env variable)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save charts as PNG files"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "No FRED API key found. Pass --api-key or set "
            "the FRED_API_KEY environment variable.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # ── 1. Data ───────────────────────────────────────────────
    print("── Fetching FRED data ──")
    df_raw = fetch_fred_data(args.api_key)

    print("\n── Engineering features ──")
    feat         = engineer_features(df_raw)
    feature_cols = get_feature_cols(feat)

    # ── 2. Model ──────────────────────────────────────────────
    print("\n── Training model ──")
    model, scaler, mean_auc, _ = train_model(feat, feature_cols)

    # ── 3. Current signal ─────────────────────────────────────
    print("\n── Current signal ──")
    current_prob = current_probability(feat, feature_cols, model, scaler)

    # ── 4. Charts ─────────────────────────────────────────────
    if not args.no_plots:
        probs = predict_proba_series(feat, feature_cols, model, scaler)
        fi_df = feature_importance_df(model, feature_cols)

        prob_path = "output/recession_probability.png" if args.save_plots else None
        ind_path  = "output/indicators.png"            if args.save_plots else None
        fi_path   = "output/feature_importance.png"    if args.save_plots else None

        if args.save_plots:
            os.makedirs("output", exist_ok=True)

        print("\n── Generating charts ──")
        plot_recession_probability(feat, probs, current_prob, save_path=prob_path)
        plot_indicators(feat, save_path=ind_path)
        plot_feature_importance(fi_df, save_path=fi_path)


if __name__ == "__main__":
    main()