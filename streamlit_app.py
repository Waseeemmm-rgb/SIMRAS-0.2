import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ----------------------------
# Hugging Face chatbot setup
# ----------------------------
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
MODEL_NAME = "google/flan-t5-small"  # lightweight model for cloud

if hf_token is None:
    st.error("⚠️ Hugging Face token not found! Set it in Streamlit Secrets.")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="SIMRAS Industrial AI Dashboard", layout="wide")
st.title("🧠 SIMRAS Industrial AI Dashboard")

# ----------------------------
# Industrial Parameters
# ----------------------------
params = {
    "Pressure (bar)": (20,50),
    "Temperature (°C)": (200,300),
    "Flow (Kg/hr)": (1,100),
    "Vibration (Hz)": (10,80),
    "Oil Condition Index": (20,80),
    "Chemical Concentration (%)": (30,90),
    "Energy Consumption (GJ)": (25,50),
    "Emissions (%)": (1,30),
    "Production Unit (ton/day)": (50,500)
}

# ----------------------------
# Initialize simulated database
# ----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Timestamp"] + list(params.keys()))

# ----------------------------
# Simulate live sensor readings
# ----------------------------
def simulate_data():
    new_row = {"Timestamp": pd.Timestamp.now().strftime("%H:%M:%S")}
    for p, (low, high) in params.items():
        new_row[p] = round(np.random.uniform(low*0.9, high*1.1), 2)
    return new_row

if st.button("📈 Generate New Reading"):
    new_data = simulate_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)
    st.success("✅ New sensor readings added.")
    st.experimental_rerun()

# ----------------------------
# Display data table
# ----------------------------
st.subheader("📊 Latest Sensor Data")
st.dataframe(st.session_state.data.tail(10))

# ----------------------------
# AI Insights / Predictive Alerts
# ----------------------------
st.subheader("💡 AI Insights & Alerts")
if len(st.session_state.data) > 0:
    latest = st.session_state.data.iloc[-1]
    for p, (low, high) in params.items():
        val = latest[p]
        if val < low*1.05 or val > high*0.95:
            st.error(f"⚠️ {p} near unsafe range! Current Value: {val}")
        else:
            st.info(f"{p} within safe range: {val}")
else:
    st.info("No readings yet. Click 'Generate New Reading'.")

# ----------------------------
# Trend Visualization
# ----------------------------
st.subheader("📈 Parameter Trends")
param_choice = st.selectbox("Select Parameter:", list(params.keys()))
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
# CSV Export
# ----------------------------
if st.button("🧾 Export CSV"):
    st.download_button(
        "⬇️ Download CSV",
        data=st.session_state.data.to_csv(index=False),
        file_name="simras_report.csv",
        mime="text/csv"
    )

# ----------------------------
# Chatbot Section
# ----------------------------
st.subheader("💬 SIMRAS AI Assistant")
if hf_token is not None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask me anything:")
    if st.button("Send"):
        if user_input:
            inputs = tokenizer(user_input, return_tensors="pt")
            reply_ids = model.generate(**inputs, max_new_tokens=50)
            reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", reply))

    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
**SIMRAS v1.0** — Developed by Mohammed Waseem Attar  
Demo Prototype for OQBI Oman — Industrial monitoring + AI chatbot
""")
