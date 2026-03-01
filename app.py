import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.write("Predict failure risks in bridges and water pipelines using Machine Learning.")

# ---------------- LOAD DATA SAFELY ----------------
@st.cache_data
def load_data():

    # -------- BRIDGE DATA --------
    bridge = pd.read_csv("bridge.csv")
    bridge = bridge.dropna()

    bridge.columns = bridge.columns.str.strip()

    # encode text columns safely
    if "Material_Type" in bridge.columns:
        bridge["Material_Type"] = bridge["Material_Type"].map(
            {"Concrete": 0, "Steel": 1}
        )

    if "Maintenance_Level" in bridge.columns:
        bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map(
            {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
        )

    bridge = bridge.dropna()

    # -------- WATER DATA --------
    water = pd.read_csv("water.csv")
    water = water.dropna()
    water.columns = water.columns.str.strip()

    # convert everything numeric safely
    for col in water.columns:
        if col != "failure":
            water[col] = pd.to_numeric(water[col], errors="coerce")

    water = water.dropna()

    return bridge, water


bridge, water = load_data()

# ---------------- TRAIN MODELS ----------------

# Bridge model
X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# Water model
X_water = water.drop(["failure", "infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# ---------------- UI LAYOUT ----------------
col1, col2 = st.columns(2)

# ================= BRIDGE =================
with col1:

    st.header("Bridge Failure Prediction")
    st.image("bridge.jpg", use_container_width=True)

    age = st.slider("Bridge Age (years)", 1, 100, 50)
    traffic = st.slider("Traffic Volume", 100, 5000, 2000)

    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox("Maintenance Level",
                               ["No-Maintenance", "Bi-Annual", "Annual"])

    material_val = 0 if material == "Concrete" else 1
    maint_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
    maintenance_val = maint_map[maintenance]

    if st.button("Predict Bridge Risk"):
        pred = bridge_model.predict(
            [[age, traffic, material_val, maintenance_val]]
        )[0]

        if pred == 1:
            st.error("⚠️ HIGH RISK of Failure")
        else:
            st.success("✅ LOW RISK")

# ================= WATER =================
with col2:

    st.header("Water Pipeline Failure Prediction")
    st.image("water.jpg", use_container_width=True)

    pressure = st.slider("Pressure (bar)", 1, 20, 8)
    flow = st.slider("Flow Rate (L/s)", 10, 200, 80)
    temp = st.slider("Temperature (°C)", 0, 60, 25)
    burst = st.selectbox("Burst Status", [0, 1])

    if st.button("Predict Pipeline Risk"):
        pred2 = water_model.predict(
            [[pressure, flow, temp, burst]]
        )[0]

        if pred2 == 1:
            st.error("⚠️ HIGH RISK of Failure")
        else:
            st.success("✅ LOW RISK")
