# 📈 Economic Policy Uncertainty Forecasting — India

A hybrid ML-econometric framework for forecasting India's Economic Policy 
Uncertainty (EPU) Index using ARIMA and XGBoost — with a weighted ensemble model.

## 🔍 What This Does
- Forecasts EPU Index (2003–2024) using three models: ARIMA, XGBoost, and Hybrid
- Compares model performance across MAE, RMSE, R², and MAPE
- Time series decomposition (Trend + Seasonal + Residual)
- XGBoost feature importance analysis
- Residual distribution analysis
- Policy event timeline (Demonetization, COVID, GST, etc.)
- Interactive train/test split control

## 🧠 Methodology
- **ARIMA(2,1,2):** Classical econometric baseline capturing autocorrelation structure
- **XGBoost:** Gradient boosted trees with lagged EPU, rolling averages, and macro features
- **Hybrid Ensemble:** Weighted combination (40% ARIMA + 60% XGBoost)
- Inspired by Baker, Bloom & Davis (2016) EPU Index methodology

## 📊 Features Used in XGBoost
`EPU_lag1` `EPU_lag2` `EPU_lag3` `EPU_rolling3` `EPU_rolling6`  
`Inflation` `Exchange_Rate` `Market_Volatility` `Fiscal_Deficit` `IIP_Growth`

## 🛠️ Tech Stack
`Python` `Streamlit` `XGBoost` `Statsmodels` `Plotly` `Scikit-learn` `Pandas`

## 🚀 Run Locally
```bash
git clone https://github.com/aarushiksharmads/epu-forecasting-india.git
cd epu-forecasting-india
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💻 Author
**Aarushi K Sharma** | Research Associate | MSc Business Statistics, VIT  
Thesis: EPU Forecasting using Hybrid ML-Econometric Ensemble  
