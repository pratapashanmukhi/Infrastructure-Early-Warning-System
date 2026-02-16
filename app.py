import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page settings
st.set_page_config(page_title="Infrastructure Early Warning System", layout="centered")

st.title("Infrastructure Early Warning System")
st.write("Predict failure risk for Bridges and Water Pipelines using Machine Learning.")

# Load datasets
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# -------------------------------
# BRIDGE MODEL
# -------------------------------

# Encode bridge categorical columns
bridge["Material_Type"] = bridge["Material_Type"].map({
    "Concrete": 0,
    "Steel": 1
})

bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map({
    "No-Maintenance": 0,
    "Bi-Annual": 1,
    "Annual": 2
})

X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# -------------------------------
# WATER MODEL
# -------------------------------

X_water = water.drop(["failure", "infrastructure_type", "Sensor_ID"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# -------------------------------
# BRIDGE PREDICTION UI
# -------------------------------
st.header("Bridge Failure Prediction")

age = st.number_input("Bridge Age", 0, 200, 50)
traffic = st.number_input("Traffic Volume", 0, 10000, 2000)
material = st.selectbox("Material Type", ["Concrete", "Steel"])
maintenance = st.selectbox("Maintenance Level", ["No-Maintenance", "Bi-Annual", "Annual"])

# Encode inputs
material_val = 0 if material == "Concrete" else 1
maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
maintenance_val = maintenance_map[maintenance]

if st.button("Predict Bridge Risk"):
    sample = pd.DataFrame(
        [[age, traffic, material_val, maintenance_val]],
        columns=X_bridge.columns
    )
    prediction = bridge_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")

# -------------------------------
# WATER PREDICTION UI
# -------------------------------
st.header("Water Pipeline Failure Prediction")

pressure = st.number_input("Pressure (bar)", 0.0, 10.0, 3.5)
flow = st.number_input("Flow Rate (L/s)", 0.0, 500.0, 180.0)
temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
burst = st.selectbox("Burst Status", [0, 1])

if st.button("Predict Water Risk"):
    sample = pd.DataFrame(
        [[pressure, flow, temperature, burst]],
        columns=X_water.columns
    )
    prediction = water_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")
