import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPU Forecasting India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f9844a;
        margin-bottom: 8px;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #f9844a;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_epu_data():
    np.random.seed(42)
    dates = pd.date_range(start='2003-01-01', end='2024-12-01', freq='MS')
    n = len(dates)

    # Base EPU with realistic India policy events
    epu = 100 + np.cumsum(np.random.normal(0, 3, n))
    epu = np.abs(epu)

    # Key policy shock events
    shocks = {
        '2008-09': 80,   # Global Financial Crisis
        '2011-06': 40,   # Eurozone crisis spillover
        '2013-06': 35,   # Taper tantrum
        '2016-11': 90,   # Demonetization
        '2017-07': 30,   # GST rollout
        '2019-08': 25,   # Article 370
        '2020-03': 150,  # COVID-19
        '2020-06': -60,  # Recovery
        '2022-02': 45,   # Russia-Ukraine
        '2023-06': -20,  # Stability
    }

    for date_str, shock in shocks.items():
        idx = dates.get_loc(pd.Timestamp(date_str))
        epu[idx:idx+3] += shock

    epu = np.clip(epu, 50, 400)

    # Macro features correlated with EPU
    inflation = 5.5 + 0.015 * epu + np.random.normal(0, 0.5, n)
    exchange_rate = 45 + 0.08 * epu + np.random.normal(0, 1, n)
    market_vol = 15 + 0.05 * epu + np.random.normal(0, 2, n)
    fiscal_deficit = 4.5 + 0.008 * epu + np.random.normal(0, 0.3, n)
    iip = 4 - 0.01 * epu + np.random.normal(0, 1, n)

    df = pd.DataFrame({
        'Date': dates,
        'EPU': np.round(epu, 2),
        'Inflation': np.round(inflation, 2),
        'Exchange_Rate': np.round(exchange_rate, 2),
        'Market_Volatility': np.round(market_vol, 2),
        'Fiscal_Deficit': np.round(fiscal_deficit, 2),
        'IIP_Growth': np.round(iip, 2),
        'EPU_lag1': pd.Series(epu).shift(1).values,
        'EPU_lag2': pd.Series(epu).shift(2).values,
        'EPU_lag3': pd.Series(epu).shift(3).values,
        'EPU_rolling3': pd.Series(epu).rolling(3).mean().values,
        'EPU_rolling6': pd.Series(epu).rolling(6).mean().values,
    })
    return df.dropna().reset_index(drop=True)

# ── Modelling ──────────────────────────────────────────────────────────────────
@st.cache_data
def run_models(df, train_size=0.8):
    split = int(len(df) * train_size)
    train = df.iloc[:split]
    test = df.iloc[split:]

    # ARIMA
    arima_model = ARIMA(train['EPU'], order=(2, 1, 2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(test))

    # XGBoost
    features = ['EPU_lag1', 'EPU_lag2', 'EPU_lag3',
                 'EPU_rolling3', 'EPU_rolling6',
                 'Inflation', 'Exchange_Rate',
                 'Market_Volatility', 'Fiscal_Deficit', 'IIP_Growth']

    X_train = train[features]
    X_test = test[features]
    y_train = train['EPU']
    y_test = test['EPU']

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train_sc, y_train)
    xgb_forecast = xgb.predict(X_test_sc)

    # Hybrid (weighted average)
    hybrid_forecast = 0.4 * arima_forecast.values + 0.6 * xgb_forecast

    actual = test['EPU'].values

    def metrics(pred):
        return {
            'MAE': round(mean_absolute_error(actual, pred), 3),
            'RMSE': round(np.sqrt(mean_squared_error(actual, pred)), 3),
            'R²': round(r2_score(actual, pred), 3),
            'MAPE (%)': round(np.mean(np.abs((actual - pred) / actual)) * 100, 3)
        }

    results = {
        'dates': test['Date'].values,
        'actual': actual,
        'arima': arima_forecast.values,
        'xgboost': xgb_forecast,
        'hybrid': hybrid_forecast,
        'metrics': {
            'ARIMA': metrics(arima_forecast.values),
            'XGBoost': metrics(xgb_forecast),
            'Hybrid': metrics(hybrid_forecast)
        },
        'feature_importance': dict(zip(features, xgb.feature_importances_)),
        'train_dates': train['Date'].values,
        'train_epu': train['EPU'].values
    }
    return results

# ── Load ───────────────────────────────────────────────────────────────────────
df = generate_epu_data()
results = run_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
train_split = st.sidebar.slider("Train/Test Split (%)", 60, 85, 80)
show_decomp = st.sidebar.checkbox("Show Time Series Decomposition", True)
show_feature = st.sidebar.checkbox("Show Feature Importance", True)
show_residuals = st.sidebar.checkbox("Show Residual Analysis", True)
model_focus = st.sidebar.selectbox("Highlight Model", ["All", "ARIMA", "XGBoost", "Hybrid"])

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Economic Policy Uncertainty — India")
st.markdown("**Hybrid ML-Econometric Forecasting Framework** | ARIMA × XGBoost Ensemble")
st.caption("Inspired by Baker, Bloom & Davis (2016) EPU Index methodology | Data: Simulated (2003–2024)")

st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
m = results['metrics']
k1, k2, k3, k4 = st.columns(4)
k1.metric("ARIMA RMSE", m['ARIMA']['RMSE'], f"R² {m['ARIMA']['R²']}")
k2.metric("XGBoost RMSE", m['XGBoost']['RMSE'], f"R² {m['XGBoost']['R²']}")
k3.metric("Hybrid RMSE", m['Hybrid']['RMSE'], f"R² {m['Hybrid']['R²']}")
k4.metric("Best Model", min(m, key=lambda x: m[x]['RMSE']), "by RMSE")

st.markdown("---")

# ── Main Forecast Chart ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Forecast vs Actual EPU</div>', unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=results['train_dates'], y=results['train_epu'],
    name='Training Data', line=dict(color='#888', width=1.5, dash='dot')
))
fig.add_trace(go.Scatter(
    x=results['dates'], y=results['actual'],
    name='Actual EPU', line=dict(color='white', width=2)
))

colors = {'ARIMA': '#4f8bf9', 'XGBoost': '#f9844a', 'Hybrid': '#90e0ef'}
model_data = {
    'ARIMA': results['arima'],
    'XGBoost': results['xgboost'],
    'Hybrid': results['hybrid']
}

for model, values in model_data.items():
    if model_focus == "All" or model_focus == model:
        fig.add_trace(go.Scatter(
            x=results['dates'], y=values,
            name=f'{model} Forecast',
            line=dict(color=colors[model], width=2, dash='dash')
        ))

fig.update_layout(
    template='plotly_dark', height=420,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    xaxis_title='Date', yaxis_title='EPU Index'
)
st.plotly_chart(fig, use_container_width=True)

# ── Metrics Table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
metrics_df = pd.DataFrame(results['metrics']).T.reset_index()
metrics_df.rename(columns={'index': 'Model'}, inplace=True)
st.dataframe(metrics_df.style.highlight_min(subset=['MAE','RMSE','MAPE (%)'], color='#1a472a')
                              .highlight_max(subset=['R²'], color='#1a472a'),
             use_container_width=True)

# ── Decomposition ──────────────────────────────────────────────────────────────
if show_decomp:
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Time Series Decomposition</div>', unsafe_allow_html=True)
    decomp = seasonal_decompose(df['EPU'], model='additive', period=12)
    fig2 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          subplot_titles=['Observed','Trend','Seasonal','Residual'])
    components = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]
    colors2 = ['white', '#4f8bf9', '#f9844a', '#90e0ef']
    for i, (comp, col) in enumerate(zip(components, colors2), 1):
        fig2.add_trace(go.Scatter(x=df['Date'], y=comp,
                                   line=dict(color=col, width=1.5), showlegend=False), row=i, col=1)
    fig2.update_layout(template='plotly_dark', height=500,
                        margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Feature Importance ─────────────────────────────────────────────────────────
if show_feature:
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 XGBoost Feature Importance</div>', unsafe_allow_html=True)
    fi = pd.DataFrame(list(results['feature_importance'].items()),
                       columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
    fig3 = px.bar(fi, x='Importance', y='Feature', orientation='h',
                   color='Importance', color_continuous_scale='Oranges',
                   template='plotly_dark')
    fig3.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ── Residual Analysis ──────────────────────────────────────────────────────────
if show_residuals:
    st.markdown("---")
    st.markdown('<div class="section-header">📉 Residual Analysis</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    for col, (model, values) in zip([r1, r2], list(model_data.items())[:2]):
        residuals = results['actual'] - values
        with col:
            fig4 = px.histogram(x=residuals, nbins=30, template='plotly_dark',
                                 color_discrete_sequence=[colors[model]],
                                 title=f'{model} Residuals', marginal='box')
            fig4.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

# ── Policy Event Timeline ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🗓️ EPU Spikes & Policy Events</div>', unsafe_allow_html=True)

events = {
    '2008-10-01': 'Global Financial Crisis',
    '2016-11-01': 'Demonetization',
    '2017-07-01': 'GST Rollout',
    '2020-03-01': 'COVID-19 Lockdown',
    '2022-02-01': 'Russia-Ukraine War',
}

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df['Date'], y=df['EPU'],
                            line=dict(color='#f9844a', width=2),
                            fill='tozeroy', fillcolor='rgba(249,132,74,0.1)',
                            name='EPU Index'))

for date, label in events.items():
    fig5.add_vline(x=date, line_dash='dash', line_color='white', line_width=1)
    fig5.add_annotation(x=date, y=df['EPU'].max() * 0.9,
                         text=label, showarrow=False,
                         textangle=-90, font=dict(size=9, color='white'))

fig5.update_layout(template='plotly_dark', height=350,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title='Date', yaxis_title='EPU Index')
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.caption("📚 Methodology: Baker, Bloom & Davis (2016) | Built by Aarushi K Sharma | MSc Business Statistics, VIT | Research Associate")