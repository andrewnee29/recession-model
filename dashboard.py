# ============================================================
# dashboard.py
# Streamlit interactive dashboard
# Run with: streamlit run dashboard.py
# ============================================================

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data import fetch_fred_data, engineer_features, get_feature_cols, SERIES
from src.model import (train_model, predict_proba_series,
                       current_probability, feature_importance_df)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="US Recession Probability Model",
    page_icon="📉",
    layout="wide"
)


# ── CACHED PIPELINE ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train(api_key: str):
    """Fetch data, engineer features, train model. Cached after first run."""
    df_raw = fetch_fred_data(api_key)
    feat = engineer_features(df_raw)
    feature_cols = get_feature_cols(feat)
    model, scaler, mean_auc, _ = train_model(feat, feature_cols)
    probs = predict_proba_series(feat, feature_cols, model, scaler)
    feat["prob"] = probs
    fi_df = feature_importance_df(model, feature_cols)
    return feat, model, scaler, feature_cols, mean_auc, fi_df


# ── RECESSION SHADING ────────────────────────────────────────
def add_recession_shading(fig, dates, rec_series, row=1):
    in_rec, start = False, None
    for d, r in zip(dates, rec_series):
        if r == 1 and not in_rec:
            start = d;
            in_rec = True
        elif r == 0 and in_rec:
            fig.add_vrect(
                x0=start, x1=d,
                fillcolor="salmon", opacity=0.15,
                line_width=0, row=row, col=1
            )
            in_rec = False


# ── APP ───────────────────────────────────────────────────────
def main():
    st.title("📉 US Recession Probability Model")
    st.caption(
        "ML model trained on FRED economic indicators · "
        "Predicts recession risk over next 3 months"
    )

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📉 Recession Model")
        st.caption("Predicts recession risk over the next 6 months using FRED economic data.")

        st.markdown("---")

        # API key — prefer environment variable, fallback to text input
        api_key = os.getenv("FRED_API_KEY") or st.text_input(
            "FRED API Key",
            type="password",
            help="Free at fred.stlouisfed.org/docs/api/api_key.html"
        )

        st.markdown("---")

        with st.expander("📊 Indicators used"):
            for name in list(SERIES.values())[1:]:
                st.markdown(f"- {name}")

        with st.expander("🧠 Model summary"):
            st.markdown("""
            **Algorithm:** Logistic Regression  
            **Validation:** TimeSeriesSplit (5 folds)  
            **Features lagged:** ≥1 month  
            **Target:** Recession onset within 6 months  
            **Class balancing:** Weighted to prioritize recall  
            """)

        st.markdown("---")
        st.caption("Data sourced from FRED · Not financial advice")

    if not api_key:
        st.info("👈 Enter your FRED API key in the sidebar to load data.")
        st.stop()

    # ── Load & train ─────────────────────────────────────────
    with st.spinner("Fetching FRED data and training model..."):
        feat, model, scaler, feature_cols, mean_auc, fi_df = load_and_train(api_key)

    # ── Current signal metrics ────────────────────────────────
    current_prob = feat["prob"].iloc[-1]
    current_date = feat.index[-1].strftime("%B %Y")

    if current_prob > 0.60:
        signal = "HIGH RISK ⚠️"
    elif current_prob > 0.35:
        signal = "ELEVATED RISK 🟡"
    else:
        signal = "LOW RISK 🟢"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recession Probability", f"{current_prob:.1%}", delta=signal)
    col2.metric("As of", current_date)
    col3.metric("Model CV AUC", f"{mean_auc:.3f}")
    col4.metric("Training Period",
                f"{feat.index[0].year}–{feat.index[-1].year}")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Probability", "📊 Indicators", "📐 Coefficients", "📖 Methodology"]
    )

    # Tab 1 — Recession probability
    with tab1:
        fig = go.Figure()
        add_recession_shading(fig, feat.index, feat["recession"])
        fig.add_trace(go.Scatter(
            x=feat.index, y=feat["prob"],
            fill="tozeroy", fillcolor="rgba(220,50,50,0.2)",
            line=dict(color="crimson", width=2),
            name="Recession Probability"
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                      annotation_text="50% threshold")
        fig.add_hline(y=current_prob, line_dash="dot", line_color="darkred",
                      annotation_text=f"Current: {current_prob:.1%}",
                      annotation_position="bottom right")
        fig.update_layout(
            title="Recession Probability (6-Month Forward)",
            yaxis_title="Probability", yaxis_range=[0, 1],
            height=450, hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Pink shading = NBER-dated recessions")

    # Tab 2 — Key indicators
    with tab2:
        indicator_map = {
            "Yield Curve (10Y-2Y)": ("yield_curve", "steelblue"),
            "Unemployment Rate": ("unemployment", "darkorange"),
            "Sahm Rule Indicator": ("sahm_rule", "purple"),
            "Payroll Growth YoY %": ("payroll_yoy", "green"),
            "Core PCE Inflation YoY %": ("core_pce_yoy", "red"),
            "Consumer Sentiment": ("consumer_sentiment", "teal"),
            "Credit Card Delinquency Rate": ("cc_delinquency", "brown"),
        }

        selected = st.multiselect(
            "Select indicators to display",
            list(indicator_map.keys()),
            default=["Yield Curve (10Y-2Y)", "Unemployment Rate",
                     "Sahm Rule Indicator"]
        )

        if selected:
            fig2 = make_subplots(
                rows=len(selected), cols=1,
                shared_xaxes=True,
                subplot_titles=selected,
                vertical_spacing=0.06
            )
            for i, name in enumerate(selected, 1):
                col_name, clr = indicator_map[name]
                if col_name in feat.columns:
                    add_recession_shading(
                        fig2, feat.index, feat["recession"], row=i
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=feat.index, y=feat[col_name],
                            line=dict(color=clr, width=1.5),
                            name=name, showlegend=False
                        ),
                        row=i, col=1
                    )
            fig2.update_layout(
                height=250 * len(selected), hovermode="x unified"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Tab 3 — Feature importance
    with tab3:
        plot_df = fi_df.head(12).sort_values("importance")
        fig3 = px.bar(
            plot_df, x="importance", y="feature",
            orientation="h", color="importance",
            color_continuous_scale="Reds",
            title="Feature Coefficients (Absolute Value)"
        )
        fig3.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        top_feature = fi_df.iloc[0]["feature"]
        st.markdown(
            f"The strongest predictor is **{top_feature}**. "
            "Bars show the absolute value of each standardized Logistic Regression "
            "coefficient — how strongly each feature pushes the predicted probability "
            "up or down. Features are standardized before training so coefficients "
            "are directly comparable. Note that correlated features (e.g. unemployment "
            "and unemp_3m_chg) may split their influence, so interpret individual "
            "coefficients alongside the methodology tab."
        )

    # Tab 4 — Methodology
    with tab4:
        st.markdown("## How the Model Works")
        st.markdown("""
        This model estimates the probability that a US recession will begin within the 
        next 6 months, using real-time economic data from the Federal Reserve (FRED).
        """)

        st.markdown("### The Machine Learning Model")
        st.markdown("""
        The model is a **Logistic Regression classifier** trained on monthly economic data 
        from 1990 to present. Logistic Regression was chosen over more complex models like 
        Gradient Boosting because:

        - **Few training examples** — only 3–4 recessions exist in the dataset. Complex 
          tree-based models tend to memorize those specific events rather than learning 
          generalizable patterns.
        - **Interpretability** — logistic regression coefficients directly show how each 
          indicator pushes the probability up or down, which is important for understanding 
          *why* the model is signaling risk.
        - **Regularization (L2, C=0.1)** — L2 regularization penalizes large coefficients 
          by adding the sum of squared weights to the loss function. This shrinks all 
          coefficients toward zero without eliminating any feature entirely — appropriate 
          here because every indicator was chosen for economic reasons and should contribute. 
          C=0.1 applies strong regularization, preventing the model from over-relying on 
          whichever indicator happened to spike before any single recession.
        - **Balanced class weights** — recessions occur in only ~10% of months. Without 
          correction the model would just predict "no recession" always. Balanced weights 
          force the model to treat missed recessions as costly errors, prioritizing **recall** 
          over precision — appropriate when the cost of missing a recession far exceeds the 
          cost of a false alarm.
        """)

        st.markdown("### Preventing Data Leakage")
        st.markdown("""
        A critical challenge in time series modeling is **data leakage** — accidentally 
        allowing the model to learn from future information when predicting the past.

        Two safeguards are in place:

        1. **All features are lagged by at least 1 month** — the model only sees data that 
           would have been available at the time of prediction.
        2. **TimeSeriesSplit cross-validation** — instead of randomly shuffling folds 
           (which would let future data leak into training), each validation fold only 
           contains data *after* the training fold. This gives an honest estimate of how 
           the model would perform in real time.
        """)

        st.markdown("### Target Variable")
        st.markdown("""
        The label is `1` if an **NBER-defined recession begins within the next 6 months**, 
        `0` otherwise. This forward-looking label makes the model practically useful — 
        it predicts *onset*, not whether a recession is already underway.

        NBER recession dates are the gold standard for US recession classification, 
        though they are published with a significant lag — meaning the most recent labels 
        may shift as new data arrives.
        """)

        st.markdown("### Features & Economic Intuition")

        features = {
            "Yield Curve (10Y–2Y)": (
                "When short-term rates exceed long-term rates, banks stop lending "
                "profitably and credit tightens economy-wide. Yield curve inversion "
                "has preceded every recession since 1970, typically 12–18 months in advance."
            ),
            "Unemployment Rate & 3M Change": (
                "Rising unemployment signals businesses are contracting. The 3-month "
                "change is particularly important — it captures the *acceleration* of "
                "labor market deterioration, not just the level."
            ),
            "Sahm Rule Indicator": (
                "Triggers when the 3-month average unemployment rate rises 0.5 percentage "
                "points above its prior 12-month low. Designed specifically as an early "
                "recession indicator with very few historical false positives."
            ),
            "Nonfarm Payroll Growth YoY": (
                "Year-over-year payroll growth turning negative has coincided with every "
                "recession in the dataset. It captures broad labor market weakness across "
                "all sectors simultaneously."
            ),
            "Core PCE Inflation YoY": (
                "The Fed's preferred inflation measure. Elevated inflation constrains the "
                "Fed's ability to cut rates in response to weakness, potentially trapping "
                "the economy in a high-rate, slowing-growth environment — the stagflation scenario."
            ),
            "Consumer Sentiment & 6M Change": (
                "Consumers drive ~70% of US GDP. Falling sentiment is a forward-looking "
                "signal that spending is about to slow before it appears in hard economic "
                "data. The 6-month change captures trend deterioration."
            ),
            "Credit Card Delinquency Rate & Change": (
                "Rising delinquencies signal consumers are financially stressed and "
                "over-leveraged. Banks respond by tightening credit standards, which "
                "amplifies the economic slowdown through reduced lending."
            ),
            "Federal Funds Rate & 6M Change": (
                "The trajectory of rate changes shapes borrowing costs economy-wide. "
                "Historically, the Fed begins cutting rates before recessions — so a "
                "declining fed funds rate is often a pre-recession signal. The current "
                "environment (rates on hold) is structurally different from prior recessions."
            ),
        }

        for feature, explanation in features.items():
            with st.expander(feature):
                st.markdown(explanation)

        st.markdown("### Model Limitations")
        st.markdown("""
        - **Few training examples** — only 3–4 recessions since 1990 limits the model's 
          ability to generalize to novel economic environments.
        - **Structural differences** — the current environment (high rates, sticky inflation, 
          Fed on hold) differs from 2001, 2008, and 2020 recessions which all featured 
          aggressive pre-recession rate cuts. The model may underestimate risk in this regime.
        - **Black swan events** — sudden shocks like COVID-19 or geopolitical crises are 
          not predictable from lagged indicators.
        - **NBER lag** — recession dates are officially confirmed months after onset, 
          meaning recent training labels may be revised.
        - This model is a research tool. It is not financial advice.
        """)


if __name__ == "__main__":
    main()