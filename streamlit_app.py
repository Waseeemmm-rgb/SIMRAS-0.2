import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simras_logic import load_data, predict_next_value, detect_risk

st.set_page_config(page_title="SIMRAS Industrial Dashboard", layout="wide")

st.title("🏭 SIMRAS - Smart Industrial Monitoring & Risk Alert System")

st.write("""
This prototype helps automate monitoring of industrial parameters like temperature, pressure, 
and flow rate — predicting future values and detecting risks automatically.
""")

# ---------------------------------------------
# Load and Display Data
# ---------------------------------------------
df = load_data()
st.subheader("📊 Live Sensor Data")
st.dataframe(df)

# ---------------------------------------------
# Data Visualization
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.write("### Temperature Trend")
    plt.figure()
    plt.plot(df["Date"], df["Temperature"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Over Time")
    st.pyplot(plt)

with col2:
    st.write("### Pressure Trend")
    plt.figure()
    plt.plot(df["Date"], df["Pressure"], marker="o", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure Over Time")
    st.pyplot(plt)

# ---------------------------------------------
# Predictions & Risk Detection
# ---------------------------------------------
st.subheader("🔮 Predictions")

pred_temp = predict_next_value(df, "Temperature")
pred_press = predict_next_value(df, "Pressure")
pred_flow = predict_next_value(df, "FlowRate")

st.write(f"🌡️ **Predicted Next Temperature:** {pred_temp} °C")
st.write(f"🧯 **Predicted Next Pressure:** {pred_press} bar")
st.write(f"💧 **Predicted Next Flow Rate:** {pred_flow} m³/hr")

risk = detect_risk(df)
st.warning(risk if "⚠️" in risk else risk)

st.success("✅ Analysis complete. Use this dashboard to monitor and predict industrial behavior.")

# ---------------------------------------------
# 💬 AI Chatbot Section
# ---------------------------------------------
st.divider()
st.header("💬 SIMRAS AI Assistant")
st.write("Ask me anything about industrial data, risks, or predictions.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display messages
for msg in st.session_state["messages"]:
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['text']}")

# Input box
user_input = st.text_input("Type your message here:")

if st.button("Send") and user_input:
    st.session_state["messages"].append({"role": "user", "text": user_input})

    # Basic AI responses
    if "risk" in user_input.lower():
        reply = "Based on current readings, the system detects high pressure and temperature risk."
    elif "trend" in user_input.lower():
        reply = "Temperature and pressure show an upward trend this week."
    elif "flow" in user_input.lower():
        reply = "Flow rate fluctuations may indicate valve irregularities."
    elif "predict" in user_input.lower():
        reply = f"The next temperature is expected around {pred_temp}°C, pressure {pred_press} bar, and flow {pred_flow} m³/hr."
    else:
        reply = "I'm SIMRAS — your AI assistant for industrial insights. Try asking about risk, flow, or predictions."

    st.session_state["messages"].append({"role": "assistant", "text": reply})
    st.rerun()
