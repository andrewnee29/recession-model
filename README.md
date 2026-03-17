# US Recession Probability Model

A machine learning model that estimates the probability of a US recession beginning within the next 6 months, trained on real Federal Reserve economic data.

**[Live Dashboard](https://recession-model.streamlit.app)**

---

## Overview

This project pulls economic indicators directly from the FRED API, engineers predictive features, and trains a Logistic Regression classifier to estimate near-term recession risk. The model currently estimates recession probability at **19.8%** as of March 2026.

---

## Background

I got genuinely interested in the macro signals accumulating in early 2026 — weakening labor markets, elevated credit card delinquencies, and sticky inflation — and wanted to quantify the risk rather than just read about it. The Boston Fed's FRED database made that possible with entirely free, real-time data. What started as a simple data pull grew into a full ML pipeline.

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Mean CV AUC | 0.932 |
| Most recent fold AUC | 0.727 |
| Valid folds | 3 of 5 (2 skipped, single class) |
| Validation method | TimeSeriesSplit |
| Training period | 1990 – present |
| Prediction horizon | Recession onset within 6 months |

Note: early folds contain as few as 1–4 recession months in the test set, making perfect AUC achievable by chance. The most recent fold (AUC 0.727) is the most meaningful estimate of real-world performance.

---

## Indicators Used

| FRED Series | Indicator | Why It Matters |
|---|---|---|
| T10Y2Y | Yield Curve (10Y–2Y) | When short-term rates exceed long-term rates, banks stop lending profitably and credit tightens economy-wide. Banks borrow short and lend long — an inverted curve eliminates their profit margin and signals that bond markets expect economic weakness ahead. Has preceded every recession since 1970, typically 12–18 months in advance. |
| UNRATE | Unemployment Rate | Rising unemployment signals businesses are contracting. The 3-month change captures the acceleration of deterioration, not just the level — a quickly rising rate is more informative than the rate itself. |
| SAHMREALTIME | Sahm Rule | Triggers when the 3-month average unemployment rate rises 0.5 percentage points above its prior 12-month low. Designed specifically as an early recession indicator with very few historical false positives. |
| PAYEMS | Nonfarm Payrolls | Year-over-year payroll growth turning negative has coincided with every recession in the dataset, capturing broad labor market weakness across all sectors simultaneously. |
| PCEPILFE | Core PCE Inflation | The Fed's preferred inflation measure. Elevated inflation constrains the Fed's ability to cut rates in response to weakness, potentially trapping the economy in a high-rate, slowing-growth stagflation environment. |
| UMCSENT | Consumer Sentiment | Consumers drive approximately 70% of US GDP. Falling sentiment is a forward-looking signal that spending is about to slow before it appears in hard economic data. The 6-month change captures trend deterioration. |
| DRCCLACBS | Credit Card Delinquency | Rising delinquencies signal consumers are financially stressed and over-leveraged. Banks respond by tightening credit standards, amplifying the economic slowdown through reduced lending. |
| FEDFUNDS | Federal Funds Rate | Rate trajectory shapes borrowing costs economy-wide. Historically the Fed begins cutting before recessions — so a declining fed funds rate is often a pre-recession signal. The current environment of rates on hold is structurally different from prior recessions. |

---

## Visualizations

**Recession Probability Chart**
The main model output. Shows the estimated probability of recession onset within 6 months for every month in the dataset. Pink shaded regions are NBER-dated recessions. A dashed line marks the 50% decision threshold and a dotted line marks the current probability estimate.

**Key Indicators Panel**
Interactive multi-panel chart of the raw economic indicators used as model inputs. Each is plotted against the same recession-shaded timeline so you can see how each behaves before and during downturns. The yield curve typically inverts 12–18 months before a recession while unemployment tends to rise only at onset.

**Feature Coefficients Chart**
Shows the absolute value of each standardized Logistic Regression coefficient — how strongly each feature pushes the predicted probability up or down. Since all features are standardized before training, coefficients are directly comparable. Note that correlated features may split their influence and should be interpreted alongside the methodology tab.

**Methodology Tab**
Full explanation of the model, regularization choices, data leakage prevention, and feature economic intuition available in the live dashboard.

---

## Methodology

### Feature Engineering

All features are lagged by at least one month to prevent data leakage. Derived features include 3-month and 6-month rates of change for key indicators, binary flags for yield curve inversion and negative payrolls, and year-over-year growth rates for payrolls and inflation.

### Preventing Data Leakage

Data leakage occurs when a model trains on information it couldn't have known at prediction time, producing artificially good backtest performance that fails in production. Two safeguards are in place:

**Feature lagging** — every feature is shifted back by at least 1 month. When predicting January's recession risk the model only sees December's indicators. This mirrors what an investor actually knows in real time — economic data is published with a delay, and using the current month's data to predict the current month's label would be cheating.

**TimeSeriesSplit cross-validation** — standard k-fold CV randomly shuffles data, allowing future observations to leak into training folds. TimeSeriesSplit ensures each validation fold only contains data after the training fold, giving an honest estimate of real-time performance.

### Target Variable

The label is 1 if an NBER-defined recession begins within the next 6 months, 0 otherwise. This forward-looking label makes the model practically useful — it predicts onset, not whether a recession is already underway. NBER recession dates are the gold standard for US recession classification, though published with a significant lag meaning recent labels may be revised.

### Model: Logistic Regression

**Why Logistic Regression over Gradient Boosting or Random Forest?**

Gradient Boosting was initially tested but with only 3–4 recessions in the dataset it memorized the specific signatures of those events rather than learning generalizable patterns, producing near-zero probability outside of known recession periods. Logistic Regression generalizes better under data scarcity and its coefficients are directly interpretable — each one shows how strongly a feature pushes the probability up or down.

**Regularization: L2 (Ridge), C=0.1**

L2 regularization penalizes large coefficients by adding the sum of squared weights to the loss function. This shrinks all coefficients toward zero without eliminating any feature entirely — appropriate here because every indicator was chosen for economic reasons and should contribute. C=0.1 applies strong regularization, preventing the model from over-relying on whichever indicator happened to spike before any single recession. L1 (Lasso) was not used because it zeros out features entirely, which is undesirable when all features carry economic meaning.

**Class Imbalance: Balanced Weights**

Recessions occur in only approximately 10% of months. Without correction the model would predict no recession everywhere and still achieve 90% accuracy. Balanced class weights upweight recession months during training, forcing the model to prioritize recall over precision — appropriate when the cost of missing a recession far exceeds the cost of a false alarm.

### Limitations

- Only 3–4 recessions since 1990 limits generalization to novel economic environments
- The current high-rate, sticky-inflation environment differs structurally from 2001, 2008, and 2020 recessions which all featured aggressive pre-recession rate cuts — the model may underestimate risk in this regime
- Black swan shocks such as COVID-19 or geopolitical crises are not predictable from lagged indicators
- NBER recession dates are confirmed months after onset, meaning recent training labels may be revised
- This is a research tool, not financial advice

---

## Project Structure

```
recession-model/
├── src/
│   ├── __init__.py       # makes src/ a Python package
│   ├── data.py           # FRED fetching and feature engineering
│   ├── model.py          # training, evaluation, prediction
│   └── viz.py            # matplotlib static charts
├── main.py               # CLI entry point
├── dashboard.py          # Streamlit interactive dashboard
├── requirements.txt      # pip dependencies
└── README.md
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/andrewnee29/recession-model
cd recession-model

# Install dependencies
pip install -r requirements.txt

# Get a free FRED API key at:
# https://fred.stlouisfed.org/docs/api/api_key.html

# Add to .env file
echo 'FRED_API_KEY=your_key_here' > .env

# Run the model
python main.py

# Launch the dashboard
streamlit run dashboard.py
```

---

*Data sourced from the Federal Reserve Bank of St. Louis (FRED). This project is for educational and research purposes only and does not constitute financial advice.*