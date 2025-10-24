import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("sample_data.csv")
    return df

def predict_next(df):
    preds = {}
    for col in ['Temperature', 'Pressure', 'Flow']:
        y = df[col].values.astype(float)
        x = np.arange(1, len(y) + 1)
        if len(y) < 2:
            preds[col] = float(y[-1])
            continue
        coeffs = np.polyfit(x, y, 1)
        next_x = len(y) + 1
        pred = float(np.polyval(coeffs, next_x))
        preds[col] = round(pred, 3)
    return preds

def analyze_risk(df):
    latest = df.iloc[-1]
    risks = {}
    if latest['Temperature'] > 320:
        risks['Temperature'] = "High - overheating risk."
    if latest['Pressure'] > 17:
        risks['Pressure'] = "Critical - pressure too high."
    if latest['Flow'] < 90:
        risks['Flow'] = "Low - possible blockage."
    return risks or {"Status": "All parameters are safe."}
