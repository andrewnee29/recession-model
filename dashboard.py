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

from src.data  import fetch_fred_data, engineer_features, get_feature_cols, SERIES
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
    df_raw       = fetch_fred_data(api_key)
    feat         = engineer_features(df_raw)
    feature_cols = get_feature_cols(feat)
    model, scaler, mean_auc, _ = train_model(feat, feature_cols)
    probs        = predict_proba_series(feat, feature_cols, model, scaler)
    feat["prob"] = probs
    fi_df        = feature_importance_df(model, feature_cols)
    return feat, model, scaler, feature_cols, mean_auc, fi_df


# ── RECESSION SHADING ────────────────────────────────────────
def add_recession_shading(fig, dates, rec_series, row=1):
    in_rec, start = False, None
    for d, r in zip(dates, rec_series):
        if r == 1 and not in_rec:
            start = d; in_rec = True
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
        st.header("⚙️ Settings")

        # API key — prefer environment variable, fallback to text input
        api_key = os.getenv("FRED_API_KEY") or st.text_input(
            "FRED API Key",
            type="password",
            help="Free at fred.stlouisfed.org/docs/api/api_key.html"
        )

        st.markdown("---")
        st.markdown("**Indicators used:**")
        for name in list(SERIES.values())[1:]:
            st.markdown(f"• {name}")

        st.markdown("---")
        st.markdown("**Methodology**")
        st.markdown(
            "Gradient Boosting classifier with TimeSeriesSplit CV. "
            "All features lagged ≥1 month to prevent data leakage. "
            "Label: recession onset within 3 months."
        )

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
    tab1, tab2, tab3 = st.tabs(
        ["📈 Probability", "📊 Indicators", "🔍 Feature Importance"]
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
            title="Recession Probability (3-Month Forward)",
            yaxis_title="Probability", yaxis_range=[0, 1],
            height=450, hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Pink shading = NBER-dated recessions")

    # Tab 2 — Key indicators
    with tab2:
        indicator_map = {
            "Yield Curve (10Y-2Y)":        ("yield_curve",        "steelblue"),
            "Unemployment Rate":            ("unemployment",       "darkorange"),
            "Sahm Rule Indicator":          ("sahm_rule",          "purple"),
            "Payroll Growth YoY %":         ("payroll_yoy",        "green"),
            "Core PCE Inflation YoY %":     ("core_pce_yoy",       "red"),
            "Consumer Sentiment":           ("consumer_sentiment", "teal"),
            "Credit Card Delinquency Rate": ("cc_delinquency",     "brown"),
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
            title="Top Feature Importances"
        )
        fig3.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        top_feature = fi_df.iloc[0]["feature"]
        st.markdown(
            f"The most predictive feature is **{top_feature}**. "
            "Features are ranked by how much they reduce impurity "
            "across all decision trees in the model."
        )


if __name__ == "__main__":
    main()