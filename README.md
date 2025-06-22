# ✈️ Aircraft Accident Cause Classifier  
AI-powered accident prediction using Flight Data Recorder (FDR) parameters.

---

## 🧠 Project Overview
This project uses machine learning to **predict possible aircraft accident causes** from real-time flight data inputs such as altitude, airspeed, engine RPM, and more.

- Built using **Python, scikit-learn, and Streamlit**
- Trained on **30,000+ synthetic yet realistic FDR records**
- Achieved **99.6% accuracy** using a Random Forest model
- Features a **live cockpit-style interface** for dispatchers and safety teams

---

## 🎯 Purpose
To simulate a **smart flight operations console** that helps:
- Aviation safety teams
- Aircraft dispatchers
- Training and simulation engineers  
in identifying and analyzing possible causes behind dangerous flight conditions.

---

## 📊 Input Parameters
The model takes 8 standard flight parameters:

- `Altitude_ft`
- `Airspeed_kt`
- `AoA_deg` (Angle of Attack)
- `Pitch_deg`
- `VerticalSpeed_fpm`
- `EngineRPM_pct`
- `Thrust_pct`
- `Flap_deg`

---

## 🧠 Output: Predicted Accident Type

Model classifies into 10 accident categories:

1. Stall  
2. Engine Failure  
3. Pilot Error  
4. Loss of Control  
5. Hard Landing  
6. Runway Overrun  
7. Mid-Air Collision Risk  
8. Fire/Smoke Onboard  
9. Uncommanded Pitch Event  
10. Landing Gear Failure  

---

## 💡 Features
- Streamlit GUI with **sliders for each input**
- Real-time prediction with **model confidence radar**
- Visual bar chart for **full accident risk distribution**
- Exportable and portable model (`.pkl` format)

---

## 🚀 How to Run the App

1. Clone or download this repo
2. Make sure you have Python 3.8+ installed
3. Install dependencies:
   in cmd:
  - pip install -r requirements.txt
  - streamlit run accident_gui.py
  
