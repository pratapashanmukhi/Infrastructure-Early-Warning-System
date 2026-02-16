import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.markdown(
    "Predict failure risks in bridges and water pipelines using **Machine Learning**."
)

# -----------------------------
# Load datasets
# -----------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# -----------------------------
# Clean column names
# -----------------------------
bridge.columns = bridge.columns.str.strip()
water.columns = water.columns.str.strip()

# -----------------------------
# Encode bridge dataset
# -----------------------------
bridge["Material_Type"] = bridge["Material_Type"].map({"Concrete": 0, "Steel": 1})
bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map(
    {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
)

# Drop non-numeric column
X_bridge = bridge[
    ["Age_of_Bridge", "Traffic_Volume", "Material_Type", "Maintenance_Level"]
]
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# -----------------------------
# Prepare water dataset
# Use only numeric columns
# -----------------------------
X_water = water[
    ["Pressure (bar)", "Flow Rate (L/s)", "Temperature (°C)", "Burst Status"]
]
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# -----------------------------
# UI Layout
# -----------------------------
st.markdown("---")

col1, col2 = st.columns(2)

# =============================
# Bridge Prediction Section
# =============================
with col1:
    st.subheader("Bridge Failure Prediction")

    age = st.slider("Bridge Age (years)", 1, 100, 50)
    traffic = st.slider("Traffic Volume", 100, 5000, 2000)

    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox(
        "Maintenance Level", ["No-Maintenance", "Bi-Annual", "Annual"]
    )

    if st.button("Predict Bridge Risk"):
        material_val = 0 if material == "Concrete" else 1
        maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
        maintenance_val = maintenance_map[maintenance]

        pred = bridge_model.predict(
            [[age, traffic, material_val, maintenance_val]]
        )[0]

        if pred == 1:
            st.error("⚠️ High Failure Risk Detected")
        else:
            st.success("✅ Bridge is Safe")

# =============================
# Water Prediction Section
# =============================
with col2:
    st.subheader("Water Pipeline Failure Prediction")

    pressure = st.slider("Pressure (bar)", 1, 20, 8)
    flow = st.slider("Flow Rate (L/s)", 10, 200, 80)
    temp = st.slider("Temperature (°C)", 0, 60, 25)
    burst = st.selectbox("Burst Status", [0, 1])

    if st.button("Predict Pipeline Risk"):
        pred = water_model.predict([[pressure, flow, temp, burst]])[0]

        if pred == 1:
            st.error("⚠️ High Failure Risk Detected")
        else:
            st.success("✅ Pipeline is Safe")
