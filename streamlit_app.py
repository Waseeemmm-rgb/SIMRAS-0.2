# ==========================================================
#  OQBI SMART PLANT DASHBOARD
#  Developed by: Mohammed Waseem Attar
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==========================================================
#  DATA SIMULATION (Realistic Sensor Data)
# ==========================================================
def generate_industrial_data(days=30):
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    data = {
        'Date': dates,
        'Temperature (°C)': np.random.uniform(70, 120, days),
        'Pressure (bar)': np.random.uniform(10, 30, days),
        'Flow Rate (m³/h)': np.random.uniform(200, 500, days),
        'Vibration (Hz)': np.random.uniform(5, 20, days),
        'Energy (GJ)': np.random.uniform(500, 1200, days),
        'Emissions (%)': np.random.uniform(2, 10, days),
    }

    df = pd.DataFrame(data)
    return df


# ==========================================================
#  RISK CALCULATION SYSTEM
# ==========================================================
def calculate_risk(df):
    df['Risk Score'] = (
        0.3 * (df['Temperature (°C)'] / 120) +
        0.25 * (df['Pressure (bar)'] / 30) +
        0.15 * (df['Flow Rate (m³/h)'] / 500) +
        0.15 * (df['Vibration (Hz)'] / 20) +
        0.15 * (df['Emissions (%)'] / 10)
    )

    df['Risk Level'] = pd.cut(
        df['Risk Score'],
        bins=[0, 0.5, 0.8, 1],
        labels=['Safe', 'Warning', 'High Risk']
    )
    return df


# ==========================================================
#  FORECASTING (Predict Next 3 Days)
# ==========================================================
def forecast_trend(df, column, days_ahead=3):
    model = LinearRegression()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[column].values
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
    predictions = model.predict(future_X)
    return predictions


# ==========================================================
#  STREAMLIT DASHBOARD LAYOUT
# ==========================================================
st.set_page_config(page_title="OQBI Smart Plant Dashboard", layout="wide")

st.title("🏭 OQBI Smart Plant Dashboard")
st.markdown("Real-time Industrial Monitoring with Predictive Insights")
st.markdown("Developed by **Mohammed Waseem Attar**")

# Generate data and calculate risk
df = generate_industrial_data(30)
df = calculate_risk(df)

# ==========================================================
#  KPI METRICS
# ==========================================================
col1, col2, col3 = st.columns(3)
col1.metric("Average Temperature", f"{df['Temperature (°C)'].mean():.1f} °C")
col2.metric("Average Pressure", f"{df['Pressure (bar)'].mean():.1f} bar")
col3.metric("Current Risk Level", df['Risk Level'].iloc[-1])

st.divider()

# ==========================================================
#  TEMPERATURE TREND WITH FORECAST
# ==========================================================
st.subheader("🌡 Temperature Trend (Past 30 Days + Forecast)")

pred_temp = forecast_trend(df, 'Temperature (°C)')
future_dates = pd.date_range(df['Date'].iloc[-1] + timedelta(days=1), periods=3)

fig = px.line(df, x='Date', y='Temperature (°C)', title="Temperature Over Time")
fig.add_scatter(x=future_dates, y=pred_temp, mode='lines+markers', name='Forecast', line=dict(dash='dash'))
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
#  RISK OVERVIEW
# ==========================================================
st.subheader("⚠️ Risk Overview (Based on All Parameters)")

risk_fig = px.scatter(
    df, x='Date', y='Risk Score', color='Risk Level',
    color_discrete_map={'Safe': 'green', 'Warning': 'orange', 'High Risk': 'red'},
    title="Daily Risk Score Levels"
)
st.plotly_chart(risk_fig, use_container_width=True)

st.divider()

# ==========================================================
#  WHAT-IF SCENARIO ANALYSIS
# ==========================================================
st.subheader("🔮 What-If Analysis — Simulate Future Scenarios")

col1, col2, col3 = st.columns(3)
temp_input = col1.slider("Set Desired Temperature (°C)", 70, 120, 90)
pressure_input = col2.slider("Set Desired Pressure (bar)", 10, 30, 20)
flow_input = col3.slider("Set Desired Flow (m³/h)", 200, 500, 300)

sim_risk = (0.3 * (temp_input / 120) +
            0.25 * (pressure_input / 30) +
            0.15 * (flow_input / 500))

st.write(f"**Predicted Risk Score:** {sim_risk:.2f}")

if sim_risk < 0.5:
    st.success("✅ Plant Status: SAFE")
elif sim_risk < 0.8:
    st.warning("⚠️ Plant Status: WARNING")
else:
    st.error("🔥 Plant Status: HIGH RISK")

st.divider()

# ==========================================================
#  EXPORT REPORT
# ==========================================================
st.subheader("📤 Export Plant Performance Report")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV Report", csv, "plant_report.csv", "text/csv")

st.caption("© 2025 OQBI Prototype — Developed by Mohammed Waseem Attar")
