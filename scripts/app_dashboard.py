import streamlit as st
import pandas as pd
import requests
import time

st.title("🛡️ IoT Security Monitor")
col1, col2 = st.columns(2)
status = col1.empty()
chart = col2.empty()

if st.button("Start Live Sim"):
    df = pd.read_csv('data/clean_mixed_test.csv').sample(50)
    X = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore')
    history = []
    
    for row in X.values:
        try:
            resp = requests.post("http://localhost:8000/predict", json={"features": row.tolist()})
            score = resp.json()['anomaly_score']
            history.append(score)
            chart.line_chart(history[-50:])
            if resp.json()['is_attack']:
                status.error(f"ATTACK DETECTED! Score: {score:.4f}")
            else:
                status.success(f"Normal. Score: {score:.4f}")
            time.sleep(0.1)
        except:
            st.error("Is the API running?")
            break