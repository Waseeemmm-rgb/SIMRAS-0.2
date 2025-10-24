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
