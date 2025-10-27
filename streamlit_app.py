import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ----------------------------
# HUGGING FACE CHATBOT SETUP
# ----------------------------
MODEL_NAME = "facebook/blenderbot-400M-distill"
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # must be set in Streamlit Secrets

if hf_token is None:
    st.error("⚠️ Hugging Face token not found! Set it in Streamlit Secrets.")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="SIMRAS Industrial AI Dashboard", layout="wide")
st.title("🧠 SIMRAS Industrial AI Dashboard")

# ----------------------------
# INDUSTRIAL PARAMETERS
# ----------------------------
params = {
    "Pressure (bar)": (20, 50),
    "Temperature (°C)": (200, 300),
    "Flow (Kg/hr)": (1, 100),
    "Vibration (Hz)": (10, 80),
    "Oil Condition Index": (20, 80),
    "Chemical Concentration (%)": (30, 90),
    "Energy Consumption (GJ)": (25, 50),
    "Emissions (%)": (1, 30),
    "Production Unit (ton/day)": (50, 500)
}

# ----------------------------
# INITIALIZE SIMULATED DATABASE
# ----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp"] + list(params.keys()))

# ----------------------------
# SIMULATE LIVE SENSOR READINGS
# ----------------------------
def simulate_live_data():
    new_entry = {"Timestamp": pd.Timestamp.now().strftime("%H:%M:%S")}
    for param, (low, high) in params.items():
        value = np.random.uniform(low * 0.9, high * 1.1)
        new_entry[param] = round(value, 2)
    return new_entry

if st.button("📈 Generate New Live Reading"):
    new_data = simulate_live_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)
    st.success("✅ New sensor readings added.")
    st.rerun()

# ----------------------------
# DISPLAY DATA TABLE
# ----------------------------
st.subheader("📊 Real-Time Sensor Data")
st.dataframe(st.session_state.data.tail(10), use_container_width=True)

# ----------------------------
# AI INSIGHTS & PREDICTIVE ALERTS
# ----------------------------
st.subheader("💡 AI Insights & Alerts")
if len(st.session_state.data) > 0:
    latest = st.session_state.data.iloc[-1]
    for param, (low, high) in params.items():
        val = latest[param]
        # Predictive alert: if value > 95% of high or < 105% of low
        if val < low * 1.05 or val > high * 0.95:
            st.error(f"⚠️ {param} near unsafe range! Current Value: {val}")
        else:
            st.info(f"{param} within safe range: {val}")
else:
    st.info("No readings yet. Click 'Generate New Live Reading'.")

# ----------------------------
# TREND VISUALIZATION
# ----------------------------
st.subheader("📈 Parameter Trends")
param_choice = st.selectbox("Select Parameter to View Trend:", list(params.keys()))
if len(st.session_state.data) > 0:
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(st.session_state.data["Timestamp"], st.session_state.data[param_choice], marker='o')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(param_choice)
    ax.set_title(f"{param_choice} Trend")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------------
# CSV REPORT EXPORT
# ----------------------------
if st.button("🧾 Export CSV Report"):
    st.download_button(
        label="⬇️ Download CSV",
        data=st.session_state.data.to_csv(index=False),
        file_name="simras_report.csv",
        mime="text/csv"
    )
    st.success("✅ CSV Report Ready!")

# ----------------------------
# CHATBOT SECTION
# ----------------------------
st.subheader("💬 SIMRAS AI Assistant (Chatbot)")

if hf_token is not None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask me anything:")

    if st.button("Send"):
        if user_input:
            # Encode input
            inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors="pt")
            reply_ids = model.generate(**inputs, max_new_tokens=100)
            reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

            # Save chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", reply))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI:** {message}")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("""
---
**SIMRAS v1.0** — Developed by *Mohammed Waseem Attar*  
Demo Prototype for **OQBI Oman** — Industrial monitoring + AI chatbot.
""")
