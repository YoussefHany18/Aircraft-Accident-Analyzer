import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and label map
model = joblib.load("accident_predictor_model.pkl")
label_map = joblib.load("label_map.pkl")
reverse_label_map = {v: k for k, v in label_map.items()}

st.set_page_config(page_title="Flight Risk Predictor Console", layout="wide")

# styles markup css
st.markdown("""
    <style>
    .main {
        background-color: #0c1b2a;
        color: white;
        font-family: 'Courier New', monospace;
    }
    div.stButton > button:first-child {
        background-color: #004466;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

#----------------------------------------------------------------

st.title("🛫 Flight Data Recorder - Smart Accident Analyzer")
st.markdown("### Cockpit AI Console")

col1, col2 = st.columns(2)

with col1:
    altitude = st.slider("Altitude (ft)", 0, 40000, 10000)
    airspeed = st.slider("Airspeed (kt)", 50, 500, 250)
    aoa = st.slider("Angle of Attack (°)", -5, 25, 5)
    pitch = st.slider("Pitch (°)", -10, 30, 2)

with col2:
    vspeed = st.slider("Vertical Speed (fpm)", -6000, 3000, 0)
    rpm = st.slider("Engine RPM (%)", 0, 100, 85)
    thrust = st.slider("Thrust (%)", 0, 100, 80)
    flap = st.slider("Flap Setting (°)", 0, 40, 5)

input_data = pd.DataFrame([[altitude, airspeed, aoa, pitch, vspeed, rpm, thrust, flap]],
                          columns=['Altitude_ft', 'Airspeed_kt', 'AoA_deg', 'Pitch_deg',
                                   'VerticalSpeed_fpm', 'EngineRPM_pct', 'Thrust_pct', 'Flap_deg'])

if st.button("🔍 Analyze Flight Data"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    predicted_class = reverse_label_map[prediction]
    top_3 = np.argsort(probs)[-3:][::-1]

    st.subheader(f"✈️ Predicted Cause: **{predicted_class}**")

    # Danger Radar
    st.markdown("#### 🔥 Danger Radar")
    st.progress(min(int(max(probs) * 100), 100))
    st.write(f"**Risk Confidence:** {max(probs)*100:.2f}%")
    st.write("**Top 3 Predictions:**")
    for idx in top_3:
        st.write(f"- {reverse_label_map[idx]}: {probs[idx]*100:.2f}%")

    # confidence chart
    st.markdown("#### 🧠 Model Confidence (All Accident Types)")
    fig2, ax2 = plt.subplots(figsize=(7, 3))

    accident_names = [reverse_label_map[i] for i in range(len(probs))]
    probs_series = pd.Series(probs, index=accident_names).sort_values(ascending=False)

    sns.barplot(x=probs_series.index, y=probs_series.values, ax=ax2, palette='crest')
    ax2.set_ylabel("Confidence", fontsize=10)
    ax2.set_xlabel("")
    ax2.set_title("Model Confidence Distribution", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=30, labelsize=9)

    st.pyplot(fig2)

    st.success("Analysis complete")
