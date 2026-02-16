import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗 Infrastructure Early Warning System")
st.markdown("Predict failure risks in bridges and water pipelines using Machine Learning.")

# -------------------------------
# Load datasets
# -------------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# Clean column names
bridge.columns = bridge.columns.str.strip()
water.columns = water.columns.str.strip()

# -------------------------------
# Rename bridge columns
# -------------------------------
bridge = bridge.rename(columns={
    "Age_of_Bridge": "age",
    "Traffic_Volume": "traffic",
    "Material_Type": "material",
    "Maintenance_Level": "maintenance"
})

# Encode bridge data
bridge["material"] = bridge["material"].map({"Concrete": 0, "Steel": 1})
bridge["maintenance"] = bridge["maintenance"].map({
    "No-Maintenance": 0,
    "Bi-Annual": 1,
    "Annual": 2
})

X_bridge = bridge[["age", "traffic", "material", "maintenance"]]
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# -------------------------------
# Rename water columns
# -------------------------------
water = water.rename(columns={
    "Pressure (bar)": "pressure",
    "Flow Rate (L/s)": "flow",
    "Temperature (°C)": "temperature",
    "Burst Status": "burst"
})

X_water = water[["pressure", "flow", "temperature", "burst"]]
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# -------------------------------
# UI Layout
# -------------------------------
col1, col2 = st.columns(2)

# ---- Bridge Prediction ----
with col1:
    st.subheader("Bridge Failure Prediction")

    age = st.slider("Bridge Age", 1, 100, 50)
    traffic = st.slider("Traffic Volume", 100, 5000, 2000)
    material = st.selectbox("Material", ["Concrete", "Steel"])
    maintenance = st.selectbox("Maintenance", ["No-Maintenance", "Bi-Annual", "Annual"])

    if st.button("Predict Bridge Risk"):
        material_val = 0 if material == "Concrete" else 1
        maintenance_map = {
            "No-Maintenance": 0,
            "Bi-Annual": 1,
            "Annual": 2
        }
        maintenance_val = maintenance_map[maintenance]

        pred = bridge_model.predict([[age, traffic, material_val, maintenance_val]])
        prob = bridge_model.predict_proba([[age, traffic, material_val, maintenance_val]])[0][1]

        st.metric("Risk Probability", f"{prob*100:.2f}%")

        if pred[0] == 1:
            st.error("High Failure Risk")
        else:
            st.success("Low Failure Risk")

# ---- Water Prediction ----
with col2:
    st.subheader("Water Pipeline Failure Prediction")

    pressure = st.slider("Pressure (bar)", 1.0, 10.0, 5.0)
    flow = st.slider("Flow Rate (L/s)", 10, 300, 100)
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    burst = st.selectbox("Burst Status", [0, 1])

    if st.button("Predict Pipeline Risk"):
        pred = water_model.predict([[pressure, flow, temperature, burst]])
        prob = water_model.predict_proba([[pressure, flow, temperature, burst]])[0][1]

        st.metric("Risk Probability", f"{prob*100:.2f}%")

        if pred[0] == 1:
            st.error("High Failure Risk")
        else:
            st.success("Low Failure Risk")
