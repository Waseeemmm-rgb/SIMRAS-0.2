import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_data():
    # Simulated dataset (replace with real industrial data later)
    data = {
        "Date": pd.date_range(start="2025-01-01", periods=10, freq="D"),
        "Temperature": [85, 88, 90, 87, 86, 92, 93, 91, 90, 89],
        "Pressure": [30, 31, 32, 33, 32, 34, 35, 34, 33, 32],
        "FlowRate": [120, 125, 130, 128, 127, 132, 135, 133, 131, 129],
    }
    df = pd.DataFrame(data)
    return df

def predict_next_value(df, column):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[column].values
    model = LinearRegression()
    model.fit(X, y)
    next_value = model.predict([[len(df)]])[0]
    return round(next_value, 2)

def detect_risk(df):
    latest_temp = df["Temperature"].iloc[-1]
    latest_pressure = df["Pressure"].iloc[-1]

    if latest_temp > 90 and latest_pressure > 33:
        return "⚠️ High Risk: Temperature and pressure are both elevated."
    elif latest_temp > 90:
        return "⚠️ Moderate Risk: Temperature approaching limit."
    elif latest_pressure > 33:
        return "⚠️ Moderate Risk: Pressure approaching limit."
    else:
        return "✅ Safe: All readings within limits."
