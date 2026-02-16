import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.markdown(
    "Predict failure risks in bridges and water pipelines using **Machine Learning**."
)

# --------------------------
# Load datasets
# --------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# --------------------------
# Encode bridge dataset
# (using correct column names from your CSV)
# --------------------------
bridge["Material_1"] = bridge["Material_1"].map({"Concrete": 0, "Steel": 1})
bridge["Maintenance_1"] = bridge["Maintenance_1"].map(
    {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
)

# Prepare bridge model
X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# --------------------------
# Prepare water model
# --------------------------
X_water = water.drop(["failure", "infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# --------------------------
# UI Layout
# --------------------------
st.markdown("---")
col1, col2 = st.columns(2)

# --------------------------
# Bridge Section
# --------------------------
with col1:
    st.subheader("Bridge Failure Prediction")

    age = st.number_input("Bridge Age", 0, 200, 50)
    traffic = st.number_input("Traffic Volume", 0, 10000, 2000)
    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox(
        "Maintenance Level",
        ["No-Maintenance", "Bi-Annual", "Annual"]
    )

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

# --------------------------
# Water Section
# --------------------------
with col2:
    st.subheader("Water Pipeline Failure Prediction")

    pressure = st.number_input("Pressure (bar)", 0.0, 20.0, 3.5)
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
