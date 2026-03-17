# 📉 US Recession Probability Model


A machine learning model that estimates the probability of a US recession over the next 3 months, trained on real Federal Reserve economic data.

**[🔴 Live Dashboard →](https://yourname-recession-model.streamlit.app)** ← replace with your Streamlit URL

---

## Overview

This project pulls economic indicators directly from the FRED API, engineers predictive features, and trains a Gradient Boosting classifier to estimate near-term recession risk. The model currently estimates recession probability at **X%** as of March 2026.

I built this because I got interested in the macro signals accumulating in early 2026 — weakening labor markets, elevated credit card delinquencies, and sticky inflation — and wanted to quantify the risk rather than just read about it.

---

## Model Performance

| Metric | Value |
|---|---|
| Cross-validation AUC | ~0.89 |
| Validation method | TimeSeriesSplit (5 folds) |
| Training period | 1990 – present |
| Prediction horizon | Recession onset within 3 months |

---

## Indicators Used

| FRED Series | Indicator | Why It Matters |
|---|---|---|
| T10Y2Y | Yield Curve (10Y–2Y) | When short-term rates exceed long-term rates (inversion), banks stop lending profitably and credit tightens across the economy. Has preceded every recession since 1970. |
| UNRATE | Unemployment Rate | Rising unemployment signals that businesses are contracting, reducing consumer income and spending in a self-reinforcing cycle. |
| SAHMREALTIME | Sahm Rule | Triggers when the 3-month average unemployment rate rises 0.5pp above its prior 12-month low. Has fired before every modern recession with very few false positives. |
| PAYEMS | Nonfarm Payrolls | Sustained negative payroll growth means the economy is shedding jobs broadly — a near-universal precursor to recession. Year-over-year growth turning negative has coincided with every recession in the dataset. |
| PCEPILFE | Core PCE Inflation | The Fed's preferred inflation measure. When inflation stays elevated, the Fed cannot cut rates to stimulate growth, trapping the economy in a high-rate, slowing-growth environment. |
| UMCSENT | Consumer Sentiment | Consumers drive ~70% of US GDP. Falling sentiment is a forward-looking signal that spending is about to slow before it shows up in hard data. |
| DRCCLACBS | Credit Card Delinquency | Rising delinquencies signal that consumers are financially stressed and over-leveraged. Banks respond by tightening credit, which amplifies the slowdown. |
| FEDFUNDS | Federal Funds Rate | The trajectory of rate changes shapes borrowing costs economy-wide. Rapid rate hikes historically precede recessions as credit conditions tighten with a lag. |

---

## Visualizations

**Recession Probability Chart**
The main output of the model. Shows the estimated probability of a recession beginning within the next 3 months for every month in the dataset. Pink shaded regions are NBER-dated recessions — periods the model should have assigned high probability. A dashed line marks the 50% decision threshold, and a dotted line marks the current probability estimate.

**Key Indicators Panel**
An interactive multi-panel chart of the raw economic indicators used as model inputs. Each indicator is plotted against the same recession-shaded timeline so you can visually see how each one behaves before and during downturns. For example, the yield curve typically inverts 12–18 months before a recession while unemployment tends to rise only at onset.

**Feature Importance Chart**
Shows which indicators the model relied on most heavily when making predictions. Importance is measured by how much each feature reduces prediction error (impurity) across all 200 decision trees. A high importance score means the model found that feature consistently useful across many different economic periods — not just one recession.

---

## Methodology

### Feature Engineering
All features are lagged by at least one month to prevent data leakage. Derived features include:
- 3-month and 6-month rate-of-change for key indicators
- Binary flags (yield curve inverted, payrolls negative)
- Year-over-year growth rates for payrolls and inflation

### Target Variable
The label is `1` if an NBER-defined recession begins within the next 3 months, `0` otherwise. This forward-looking label makes the model practically useful — it predicts onset, not current state.

### Model
Gradient Boosting Classifier (scikit-learn) with:
- 200 estimators, learning rate 0.05, max depth 3
- Subsampling (0.8) to reduce overfitting
- TimeSeriesSplit cross-validation to respect temporal ordering

**Why Gradient Boosting over Random Forest or Logistic Regression?**

Recession prediction is a weak signal problem — no single indicator reliably predicts downturns. The predictive power comes from *interactions* between indicators: an inverted yield curve alone is insufficient, but combined with rising delinquencies and negative payrolls it tells a very different story. Gradient Boosting's sequential error correction iteratively learns these subtle interactions, making it better suited than Random Forest (which averages independent trees, smoothing interactions away) or Logistic Regression (which assumes linear relationships). The class imbalance (~10% recession months) also favors boosting, as the sequential weighting naturally focuses more attention on hard-to-classify pre-recession periods.

### Why TimeSeriesSplit?
Standard k-fold CV would allow future data to leak into training folds, artificially inflating performance. TimeSeriesSplit ensures each validation fold only sees data the model couldn't have known at that point in time.

### Limitations
- Recessions are rare events (~10% of months), creating class imbalance
- The model cannot anticipate black swan shocks (COVID, financial crises) not reflected in prior indicators
- NBER recession dating is published with a significant lag, meaning recent labels may shift
- This is a research tool, not financial advice

---

## Project Structure

```
recession-model/
├── model.py          # Data pipeline, feature engineering, model training
├── dashboard.py      # Streamlit interactive dashboard
├── requirements.txt  # Dependencies
└── README.md
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/recession-model
cd recession-model

# Install dependencies
pip install -r requirements.txt

# Get a free FRED API key at:
# https://fred.stlouisfed.org/docs/api/api_key.html

# Run the model
python model.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## Requirements

```
fredapi
pandas
numpy
matplotlib
scikit-learn
streamlit
plotly
```

---

## Background

I'm a math and data science student interested in the intersection of quantitative modeling and macroeconomics. This project was motivated by real questions about current market conditions and grew into a full ML pipeline. The Boston Fed's FRED database was an invaluable resource — all data is public and free to access.

---

*Data sourced from the Federal Reserve Bank of St. Louis (FRED). This project is for research and educational purposes only and does not constitute financial advice.*