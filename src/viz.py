# ============================================================
# src/viz.py
# Matplotlib visualizations for CLI/static output
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _shade_recessions(ax, dates, recession_series):
    """
    Shade NBER recession periods on a matplotlib axis.

    Args:
        ax:                Matplotlib axis
        dates:             DatetimeIndex aligned to recession_series
        recession_series:  Binary series (1 = recession)
    """
    in_rec, start = False, None
    for d, r in zip(dates, recession_series):
        if r == 1 and not in_rec:
            start = d
            in_rec = True
        elif r == 0 and in_rec:
            ax.axvspan(start, d, color="lightcoral", alpha=0.3)
            in_rec = False


def plot_recession_probability(feat: pd.DataFrame, probs: np.ndarray,
                                current_prob: float, save_path: str = None):
    """
    Plot model recession probability over time with NBER shading.

    Args:
        feat:         Feature DataFrame (index = dates)
        probs:        Probability array from predict_proba_series()
        current_prob: Current probability scalar
        save_path:    Optional file path to save PNG
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    _shade_recessions(ax, feat.index, feat["recession"])

    ax.fill_between(feat.index, probs, alpha=0.3, color="crimson")
    ax.plot(feat.index, probs, color="crimson", linewidth=1.5,
            label="Recession Probability")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               label="50% threshold")
    ax.axhline(current_prob, color="darkred", linestyle=":",
               linewidth=0.8, label=f"Current: {current_prob:.1%}")

    ax.set_title("US Recession Probability — 3-Month Forward Estimate",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.01, 0.92, "Pink shading = NBER recessions",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_indicators(feat: pd.DataFrame, save_path: str = None):
    """
    Three-panel plot of key recession indicators.

    Args:
        feat:      Feature DataFrame
        save_path: Optional file path to save PNG
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Key Recession Indicators", fontsize=14, fontweight="bold")

    panels = [
        ("yield_curve",        "Yield Curve (10Y-2Y %)",      "steelblue",   0),
        ("unemployment",       "Unemployment Rate (%)",        "darkorange",  0),
        ("sahm_rule",          "Sahm Rule Indicator",          "purple",      0),
    ]

    for ax, (col, title, color, hline) in zip(axes, panels):
        _shade_recessions(ax, feat.index, feat["recession"])
        ax.plot(feat.index, feat[col], color=color, linewidth=1.2)
        if hline is not None:
            ax.axhline(hline, color=color, linestyle=":", linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(title.split("(")[-1].replace(")", "") if "(" in title else "")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 10,
                             save_path: str = None):
    """
    Horizontal bar chart of top feature importances.

    Args:
        fi_df:     DataFrame from feature_importance_df()
        top_n:     Number of features to display
        save_path: Optional file path to save PNG
    """
    df = fi_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["feature"], df["importance"], color="steelblue", alpha=0.8)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()