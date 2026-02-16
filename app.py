import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page settings
st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

# Title
st.title("🏗️ Infrastructure Early Warning System")
st.markdown(
    "This application predicts failure risks in bridges and water pipelines "
    "using Machine Learning to support **preventive maintenance**."
)

# -------------------------
# Load datasets
# -------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# -------------------------
# Prepare Bridge model
# -------------------------
X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# -------------------------
# Prepare Water model
# -------------------------
X_water = water.drop(["failure", "infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# -------------------------
# Bridge Prediction UI
# -------------------------
st.markdown("---")
st.header("Bridge Failure Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Bridge Age", 0, 200, 50)
    traffic = st.number_input("Traffic Volume", 0, 10000, 2000)

with col2:
    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox(
        "Maintenance Level", ["No-Maintenance", "Bi-Annual", "Annual"]
    )

# Encode inputs
material_val = 0 if material == "Concrete" else 1
maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
maintenance_val = maintenance_map[maintenance]

if st.button("Predict Bridge Risk"):
    sample = pd.DataFrame(
        [[age, traffic, material_val, maintenance_val]],
        columns=X_bridge.columns,
    )

    prediction = bridge_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")

# -------------------------
# Water Prediction UI
# -------------------------
st.markdown("---")
st.header("Water Pipeline Failure Prediction")

col3, col4 = st.columns(2)

with col3:
    pressure = st.number_input("Pressure (bar)", 0.0, 10.0, 3.5)
    flow = st.number_input("Flow Rate (L/s)", 0.0, 500.0, 180.0)

with col4:
    temp = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
    burst = st.selectbox("Burst Status", [0, 1])

if st.button("Predict Water Risk"):
    sample = pd.DataFrame(
        [[pressure, flow, temp, burst]],
        columns=X_water.columns,
    )

    prediction = water_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")

