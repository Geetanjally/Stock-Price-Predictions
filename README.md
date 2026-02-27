# Trader Performance vs Market Sentiment

## Objective

This project analyzes how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and performance on Hyperliquid.

The goal is to uncover measurable behavioral patterns and derive actionable strategy insights that can inform smarter trading decisions.

---

## Datasets Used

### 1. Bitcoin Market Sentiment (Fear/Greed Index)

* Columns: `date`, `classification`
* Daily sentiment label (Fear / Greed)

### 2. Hyperliquid Historical Trader Data

* Trade-level execution data
* Key fields:

  * `Account`
  * `Execution Price`
  * `Size USD`
  * `Side`
  * `Timestamp IST`
  * `Closed PnL`
  * `Leverage` (if applicable)

---

## Part A — Data Preparation

### Data Cleaning

* Removed duplicate trade records
* Checked and documented missing values
* Standardized timestamp formats
* Converted timestamps to `datetime64[ns]`
* Normalized trades to daily granularity
* Merged trade data with sentiment data on date

### Feature Engineering

Constructed key analytical metrics:

* Daily PnL per trader
* Win rate (Closed PnL > 0)
* Average trade size
* Trade frequency per day
* Long/Short ratio
* Volatility proxy (standard deviation of daily PnL)
* Trader segmentation (behavior-based grouping)

---

## Part B — Analysis

### 1. Performance Across Sentiment Regimes

* Average PnL tends to increase during Greed regimes.
* Win rate improves moderately in Greed periods.
* However, PnL volatility also increases significantly.

**Interpretation:**
Greed increases capital deployment and risk-taking behavior, leading to higher dispersion in outcomes rather than consistently improved profitability.

---

### 2. Behavioral Changes Under Sentiment Shifts

Observed changes during Greed regimes:

* Increased trade frequency
* Larger average position sizes
* Stronger long bias
* Expanded risk exposure

**Conclusion:**
Market sentiment influences trader behavior more strongly than it influences pure profitability.

---

### 3. Trader Segmentation Insights

Traders were segmented into:

* High-frequency vs Low-frequency traders
* High-risk vs Low-risk traders
* Consistent vs Inconsistent performers

Key observation:
High-frequency traders exhibit more stable performance across regimes, while low-frequency traders display higher volatility during sentiment extremes.

---

## Bonus — Predictive Modeling

A logistic regression model was implemented to predict trade profitability (Win/Loss) using:

* Sentiment classification
* Position size
* Behavioral features

Findings indicate:

* Sentiment contributes to probability shifts in trade outcomes.
* Position sizing plays a stronger role in determining variance of returns.

This confirms that sentiment impacts risk behavior more than deterministic profitability.

---

## Strategy Recommendations

### Strategy 1: Controlled Expansion During Greed

Increase trade frequency during Greed regimes, but cap position size growth to control volatility amplification.

### Strategy 2: Systematic Execution During High Sentiment

Adopt rule-based execution rather than discretionary leverage expansion during sentiment extremes.

---

## Key Takeaways

* Sentiment affects trader behavior more than raw profitability.
* Greed regimes increase both returns and volatility.
* Risk-adjusted thinking is critical during sentiment-driven expansions.
* Behavioral segmentation provides stronger insight than aggregate metrics.

---

## How to Run

### Option 1 — Jupyter Notebook

Open:

```
assignment.ipynb
```

### Option 2 — Streamlit Dashboard

Run:

```
streamlit run main.py
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

---

## Methodology Summary

1. Data cleaning and validation
2. Daily time alignment and merging
3. Feature engineering
4. Regime-based comparison
5. Trader segmentation
6. Predictive modeling
7. Strategy formulation

---
