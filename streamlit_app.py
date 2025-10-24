import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simras_logic import load_data, predict_next, analyze_risk

st.set_page_config(page_title="SIMRAS - Smart Industrial Monitoring", layout="wide")

st.title("🧠 SIMRAS - Smart Industrial Monitoring & Risk Advisory System")
st.write("Monitor industrial parameters, predict trends, and detect risks automatically.")

df = load_data()
st.subheader("📊 Industrial Data Overview")
st.dataframe(df)

st.subheader("⚠️ Risk Analysis")
risks = analyze_risk(df)
for key, value in risks.items():
    st.write(f"**{key}:** {value}")

st.subheader("🤖 Predicted Next Day Values (AI-based)")
predictions = predict_next(df)
st.write(predictions)

st.subheader("📈 Trend Graphs")
fig, ax = plt.subplots(figsize=(10, 5))
for col in ['Temperature', 'Pressure', 'Flow']:
    ax.plot(df['Day'], df[col], marker='o', label=col)
ax.legend()
ax.set_xlabel("Day")
ax.set_ylabel("Values")
ax.grid(True)
st.pyplot(fig)

st.success("✅ Analysis complete. Use this dashboard to monitor and predict industrial behavior.")
import requests

st.divider()
st.header("💬 SIMRAS AI Assistant")
st.write("Ask me anything about industrial data, risks, or parameter predictions.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for msg in st.session_state["messages"]:
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['text']}")

# Input box
user_input = st.text_input("Type your message here:")

if st.button("Send") and user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "text": user_input})

    # --- Simple logic for now ---
    if "risk" in user_input.lower():
        reply = "Based on current readings, the system detects high pressure and elevated temperature risks."
    elif "trend" in user_input.lower():
        reply = "Temperature and pressure are increasing steadily across the last 5 days."
    elif "flow" in user_input.lower():
        reply = "Flow rates show variability, possibly due to valve control issues."
    else:
        # If nothing matches, use AI model from Hugging Face (free)
        API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        headers = {"Authorization": "Bearer hf_xxxxxxxxx"}  # optional if you create a free account
        payload = {"inputs": user_input}
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        reply = data[0]["generated_text"] if isinstance(data, list) else "I'm not sure, could you rephrase?"

    st.session_state["messages"].append({"role": "assistant", "text": reply})
    st.rerun()

