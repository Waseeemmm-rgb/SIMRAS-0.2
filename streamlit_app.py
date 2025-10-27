import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login

# ----------------------------
# HUGGING FACE LOGIN
# ----------------------------
login("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")  # your token

# Load chatbot model
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="SIMRAS Industrial AI Dashboard", layout="wide")
st.title("🧠 SIMRAS Industrial AI Dashboard")

# ----------------------------
# PARAMETERS AND LIMITS
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

# Initialize simulated database
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

# Add new reading
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
# AI INSIGHT SIMULATION
# ----------------------------
st.subheader("💡 AI Insights")
st.info("Pressure is trending up. Risk of overheating in 2 hours. Recommend adjusting valves.")
st.info("Flow rates are within limits. Energy consumption optimal.")

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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about industrial processes:")

if st.button("Send"):
    if user_input:
        # Encode input
        inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors="pt")
        reply_ids = model.generate(**inputs, max_new_tokens=100)
        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Save history
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
Demo Prototype for **OQBI Oman** — Combining industrial monitoring with AI chatbot assistance.
""")
