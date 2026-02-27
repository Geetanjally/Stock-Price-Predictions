# 📊 Trader Performance vs Market Sentiment Analysis

## 🚀 Live Interactive Dashboard

🔗 **[https://geetanjally-stock-price-predictions-main-barkiy.streamlit.app/](https://geetanjally-stock-price-predictions-main-barkiy.streamlit.app/)**

---

## 🎯 Project Objective

This project analyzes how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and performance.

The goal is to identify measurable behavioral patterns and translate them into actionable trading strategy insights.

---

## 📂 Datasets Used

### 1️⃣ Bitcoin Market Sentiment (Fear/Greed Index)

* `date`
* `classification` (Fear / Greed)

### 2️⃣ Historical Trader Data

Trade-level execution records including:

* Account
* Execution Price
* Size USD
* Side (Long/Short)
* Timestamp
* Closed PnL
* Leverage

---

## 🧹 Data Preparation

### ✔ Data Cleaning

* Removed duplicates
* Handled missing values
* Standardized datetime formats
* Converted timestamps to daily level
* Aligned both datasets on `date`

### ✔ Feature Engineering

Created analytical features:

* 📈 Daily PnL per trader
* ✅ Win rate
* 💰 Average trade size
* 🔁 Trade frequency
* ⚖️ Long/Short ratio
* 📊 PnL volatility (std deviation)
* 🎯 Behavioral segments

---

## 📊 Exploratory Analysis

### 🔹 Sentiment vs Performance

* Greed days show higher average PnL
* Win rate slightly improves during Greed
* Volatility increases significantly

👉 Sentiment amplifies risk-taking behavior rather than guaranteeing profitability.

---

### 🔹 Behavioral Shifts

During Greed regimes:

* Trade frequency increases
* Position sizes increase
* Long bias strengthens
* Risk exposure expands

---

### 🔹 Trader Segmentation

Traders segmented into:

* High vs Low leverage
* High vs Low frequency
* Consistent vs Inconsistent performers

High-frequency traders display more stable returns across sentiment regimes.

---

## 🤖 Predictive Modeling

A **Random Forest Classifier** was used to predict trade profitability (Win = 1, Loss = 0).

### 🔎 Features Used:

* Sentiment classification
* Trade size
* Trade frequency
* Leverage
* Long/Short indicator

### 📌 Why Random Forest?

* Captures nonlinear relationships
* Handles mixed feature types
* Robust to noise
* Provides feature importance

### 📊 Model Insights:

* Trade size and leverage were strong predictors
* Sentiment influenced behavior-based features
* Model performance indicates profitability is multi-factor driven

---

## 💡 Strategy Recommendations

### 📌 Strategy 1 — Controlled Risk in Greed Regimes

Increase trade participation during Greed but cap leverage expansion.

### 📌 Strategy 2 — Volatility-Aware Position Sizing

Reduce position sizes during Fear to stabilize drawdowns.

---

## ⚙️ How to Run

### Notebook:

```bash
jupyter notebook assignment.ipynb
```

### Streamlit App:

```bash
streamlit run main.py
```

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
