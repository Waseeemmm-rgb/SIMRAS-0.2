# simras_advanced.py
# SIMRAS Advanced Prototype — Week 1 integrated deliverable
# Developed for demo: Multi-param sim data, risk score, forecasting, anomaly detection, what-if optimizer, basic assistant.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import io

st.set_page_config(page_title="SIMRAS Advanced Prototype", layout="wide")

# -------------------------
# Helper: simulate industrial data
# -------------------------
def generate_simulated_data(days=30, seed=42):
    np.random.seed(seed)
    start = datetime.now() - timedelta(days=days-1)
    dates = [start + timedelta(days=i) for i in range(days)]
    df = pd.DataFrame({
        "date": dates,
        "temperature": np.round(np.random.normal(loc=95, scale=6, size=days), 2),    # °C
        "pressure": np.round(np.random.normal(loc=22, scale=2.5, size=days), 2),     # bar
        "flow": np.round(np.random.normal(loc=350, scale=40, size=days), 2),        # m3/h
        "vibration": np.round(np.random.normal(loc=10, scale=2, size=days), 2),     # Hz
        "energy": np.round(np.random.normal(loc=850, scale=80, size=days), 2),      # GJ
        "emissions": np.round(np.random.normal(loc=5.5, scale=1.5, size=days), 2),  # %
        "oil_condition": np.round(np.random.uniform(30, 80, size=days), 2)
    })
    # keep realistic bounds
    df['temperature'] = df['temperature'].clip(60, 140)
    df['pressure'] = df['pressure'].clip(8, 40)
    df['flow'] = df['flow'].clip(150, 700)
    df['vibration'] = df['vibration'].clip(1, 40)
    df['energy'] = df['energy'].clip(200, 2000)
    df['emissions'] = df['emissions'].clip(0.5, 30)
    return df

# -------------------------
# Risk scoring
# -------------------------
def compute_risk_score(df_row):
    # weights tuned for demo (you can change)
    # normalize each param by expected safe range max
    t = df_row['temperature'] / 140    # normalizing by 140°C as high bound
    p = df_row['pressure'] / 40        # pressure normalized
    f = abs(df_row['flow'] - 350) / 350 # deviation from nominal flow
    v = df_row['vibration'] / 40
    e = df_row['emissions'] / 30
    oil = (100 - df_row['oil_condition']) / 100  # lower oil_condition -> higher risk

    # weighted sum (customizable)
    score = 0.28*t + 0.24*p + 0.18*f + 0.12*v + 0.12*e + 0.06*oil
    # clip 0..1.5 (we'll map >1 => high risk)
    return float(np.clip(score, 0, 1.5))

def map_risk_level(score):
    if score < 0.45:
        return "Safe"
    if score < 0.8:
        return "Warning"
    return "High Risk"

# -------------------------
# Forecast helper (linear)
# -------------------------
def forecast_linear(series, days_ahead=3):
    # series: pandas Series of numbers (oldest -> newest)
    y = np.array(series, dtype=float)
    X = np.arange(len(y)).reshape(-1,1)
    if len(y) < 2:
        return [float(y[-1])] * days_ahead
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(y), len(y)+days_ahead).reshape(-1,1)
    preds = model.predict(future_X)
    return list(preds)

# -------------------------
# Anomaly detection helper
# -------------------------
def detect_anomalies(df, features=None):
    if features is None:
        features = ["temperature", "pressure", "flow", "vibration", "energy", "emissions", "oil_condition"]
    X = df[features].values
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)  # -1 anomaly, 1 normal
    df['anomaly'] = preds
    df['anomaly_flag'] = df['anomaly'].apply(lambda x: True if x == -1 else False)
    return df

# -------------------------
# App UI / Flow
# -------------------------
st.title("SIMRAS — Advanced Industrial Prototype")
st.caption("Integrated demo: simulated live data, forecasting, anomaly detection, risk scoring, what-if, and assistant.")
st.markdown("---")

# Controls
colA, colB, colC = st.columns([1,1,1])
with colA:
    days = st.number_input("Simulated history days", min_value=10, max_value=120, value=30, step=1)
with colB:
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
with colC:
    simulate = st.button("(Re)Generate simulated history")

# Initialize session state for data
if 'df' not in st.session_state or simulate:
    st.session_state.df = generate_simulated_data(days=days, seed=seed)
    # compute risk for all rows
    st.session_state.df['risk_score'] = st.session_state.df.apply(compute_risk_score, axis=1)
    st.session_state.df['risk_level'] = st.session_state.df['risk_score'].apply(map_risk_level)
    st.session_state.df = detect_anomalies(st.session_state.df)

# Option: simulate live tick by appending a new row (next day)
st.sidebar.header("Live simulation")
if st.sidebar.button("Append new live reading (next day)"):
    last = st.session_state.df.iloc[-1].to_dict()
    new_date = st.session_state.df['date'].iloc[-1] + timedelta(days=1)
    # small random walk changes
    new_row = {
        "date": new_date,
        "temperature": float(np.clip(last['temperature'] + np.random.normal(0,2), 60, 140)),
        "pressure": float(np.clip(last['pressure'] + np.random.normal(0,0.8), 8, 40)),
        "flow": float(np.clip(last['flow'] + np.random.normal(0,10), 150, 700)),
        "vibration": float(np.clip(last['vibration'] + np.random.normal(0,0.5), 1, 40)),
        "energy": float(np.clip(last['energy'] + np.random.normal(0,15), 200, 2000)),
        "emissions": float(np.clip(last['emissions'] + np.random.normal(0,0.2), 0.5, 30)),
        "oil_condition": float(np.clip(last['oil_condition'] + np.random.normal(0,1), 0, 100))
    }
    new_row['risk_score'] = compute_risk_score(new_row)
    new_row['risk_level'] = map_risk_level(new_row['risk_score'])
    # append
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
    st.session_state.df = detect_anomalies(st.session_state.df)
    st.experimental_rerun()

# Main view: latest snapshot
st.subheader("Live Snapshot — Latest Reading")
latest = st.session_state.df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature (°C)", f"{latest['temperature']:.2f}")
col2.metric("Pressure (bar)", f"{latest['pressure']:.2f}")
col3.metric("Flow (m³/h)", f"{latest['flow']:.2f}")
col4.metric("Risk Level", f"{latest['risk_level']}", help=f"Risk score: {latest['risk_score']:.2f}")

st.markdown("**Anomaly flag:** " + ("🔴 ANOMALY DETECTED" if latest['anomaly_flag'] else "✅ Normal"))

st.markdown("---")

# Time-series with forecasts
st.subheader("Trends & Forecasts (next 3 days)")

# choose parameter to view
param = st.selectbox("Select parameter to view", options=["temperature","pressure","flow","vibration","energy","emissions","oil_condition"], index=0)
label_map = {
    "temperature":"Temperature (°C)", "pressure":"Pressure (bar)", "flow":"Flow Rate (m³/h)",
    "vibration":"Vibration (Hz)", "energy":"Energy (GJ)", "emissions":"Emissions (%)", "oil_condition":"Oil Condition Index"
}
display_label = label_map[param]

series = st.session_state.df[param]
preds = forecast_linear(series, days_ahead=3)
# build plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=st.session_state.df['date'], y=series, mode='lines+markers', name='Historical'))
future_dates = [st.session_state.df['date'].iloc[-1] + timedelta(days=i) for i in range(1,4)]
fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines+markers', name='Forecast', line=dict(dash='dash')))
fig.update_layout(title=f"{display_label} — Historical + Forecast", xaxis_title="Date", yaxis_title=display_label)
st.plotly_chart(fig, use_container_width=True)

# show table of last N days with risk and anomaly
st.subheader("Recent Data Table (last 15 rows)")
st.dataframe(st.session_state.df.tail(15).reset_index(drop=True))

# aggregated risk chart
st.subheader("Risk Score Over Time")
fig2 = px.line(st.session_state.df, x='date', y='risk_score', title="Risk Score")
fig2.add_scatter(x=st.session_state.df['date'], y=st.session_state.df['risk_score'], mode='markers',
                 marker=dict(color=st.session_state.df['anomaly_flag'].map({True:'red', False:'blue'})), name='points')
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# what-if optimizer (digital twin-lite)
st.subheader("What-If / Digital-Twin (Quick) — Adjust setpoints and see risk impact")

colA, colB, colC = st.columns(3)
with colA:
    wtemp = st.slider("Simulated Temperature (°C)", 70.0, 140.0, float(latest['temperature']), step=0.5)
with colB:
    wpress = st.slider("Simulated Pressure (bar)", 8.0, 40.0, float(latest['pressure']), step=0.1)
with colC:
    wflow = st.slider("Simulated Flow (m³/h)", 150.0, 700.0, float(latest['flow']), step=1.0)

sim_row = {"temperature": wtemp, "pressure": wpress, "flow": wflow, "vibration": latest['vibration'],
           "energy": latest['energy'], "emissions": latest['emissions'], "oil_condition": latest['oil_condition']}
sim_score = compute_risk_score(sim_row)
st.metric("Simulated Risk Score", f"{sim_score:.3f}")
st.write("Resulting Risk Level:", map_risk_level(sim_score))
st.markdown("**Advisor suggestion (prototype):**")
if sim_score >= 0.8:
    st.write("- High risk predicted. Suggest reducing throughput or initiating controlled shutdown. Check vibration and oil condition immediately.")
elif sim_score >= 0.5:
    st.write("- Warning: Monitor closely. Inspect critical valves and cooling loops.")
else:
    st.write("- Conditions within acceptable range.")

st.markdown("---")

# anomaly insight
st.subheader("Anomaly Insights")
num_anoms = int(st.session_state.df['anomaly_flag'].sum())
st.write(f"Detected anomalies in history: **{num_anoms}** (IsolationForest prototype)")
if num_anoms > 0:
    st.dataframe(st.session_state.df[st.session_state.df['anomaly_flag']].sort_values('date').reset_index(drop=True))

st.markdown("---")

# Basic assistant (rule-based, uses actual numbers)
st.subheader("Assistant — Ask the system (basic)")
if 'assistant_history' not in st.session_state:
    st.session_state.assistant_history = []

query = st.text_input("Ask SIMRAS (e.g., 'What is the most risky parameter?', 'Predict tomorrow pressure')", key="assistant_input")
if st.button("Ask"):
    q = query.lower()
    answer = "I didn't understand — try asking about 'risk', 'predict', 'trend', or 'anomaly'."
    if 'risky' in q or 'most risky' in q or 'which parameter' in q:
        # compute parameter impact by comparing normalized contribution
        latest_row = st.session_state.df.iloc[-1]
        contributions = {
            'temperature': 0.28*(latest_row['temperature']/140),
            'pressure': 0.24*(latest_row['pressure']/40),
            'flow_deviation': 0.18*(abs(latest_row['flow']-350)/350),
            'vibration': 0.12*(latest_row['vibration']/40),
            'emissions': 0.12*(latest_row['emissions']/30),
            'oil_condition': 0.06*((100-latest_row['oil_condition'])/100)
        }
        top = max(contributions, key=contributions.get)
        answer = f"Most contributing factor right now: **{top}** (see latest contribution values)."
    elif 'predict' in q or 'tomorrow' in q or 'next' in q:
        # give basic predictions for next day using linear forecast
        preds = {p: round(forecast_linear(st.session_state.df[p], days_ahead=1)[0], 2) for p in ['temperature','pressure','flow']}
        answer = f"Predicted next-day values (approx): Temperature {preds['temperature']} °C, Pressure {preds['pressure']} bar, Flow {preds['flow']} m³/h."
    elif 'trend' in q:
        answer = "Trends: temperature and pressure show gentle upward trend (linear forecast). Use 'predict' for numeric values."
    elif 'anomaly' in q:
        answer = f"There are {int(num_anoms)} anomalies in the current history. Latest reading anomaly flag: {bool(latest['anomaly_flag'])}."
    else:
        answer = "Try: 'What is the most risky parameter?', 'Predict next day', or 'How many anomalies?'"

    st.session_state.assistant_history.append({"query": query, "answer": answer})
    st.experimental_rerun()

if st.session_state.assistant_history:
    st.subheader("Assistant Log")
    for h in reversed(st.session_state.assistant_history[-6:]):
        st.markdown(f"**Q:** {h['query']}  \n**A:** {h['answer']}")

# -------------------------
# Download CSV report
# -------------------------
st.markdown("---")
st.subheader("Export / Report")
buf = io.StringIO()
st.session_state.df.to_csv(buf, index=False)
st.download_button("Download CSV of history", buf.getvalue(), file_name="simras_history.csv", mime="text/csv")

st.info("Prototype notes: This system uses simulated data. For production, connect to plant historian (OPC UA/MQTT), validate models with real labelled data, and secure all endpoints.")
