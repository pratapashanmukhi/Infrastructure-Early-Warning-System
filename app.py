import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.write("Predict failure risks in bridges and water pipelines using Machine Learning.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():

    bridge = pd.read_csv("bridge.csv")
    water = pd.read_csv("water.csv")

    # clean column names
    bridge.columns = bridge.columns.str.strip()
    water.columns = water.columns.str.strip()

    # remove empty rows
    bridge = bridge.dropna()
    water = water.dropna()

    # encode bridge text data
    bridge["Material_Type"] = bridge["Material_Type"].map(
        {"Concrete": 0, "Steel": 1}
    )

    bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map(
        {"No-Maintenance": 0, "Annual": 1, "Bi-Annual": 2}
    )

    # convert numeric safely (water dataset)
    for col in water.columns:
        if col not in ["failure", "infrastructure_type"]:
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
    st.subheader("Bridge Failure Prediction")

    age = st.slider("Bridge Age (years)", 1, 100, 40)
    traffic = st.slider("Traffic Volume", 100, 5000, 2000)

    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox(
        "Maintenance Level",
        ["No-Maintenance", "Annual", "Bi-Annual"]
    )

    material_val = 0 if material == "Concrete" else 1
    maintenance_val = {"No-Maintenance":0, "Annual":1, "Bi-Annual":2}[maintenance]

    if st.button("Predict Bridge Risk"):

        pred = bridge_model.predict(
            [[age, traffic, material_val, maintenance_val]]
        )[0]

        if pred == 1:
            st.error("⚠️ High Failure Risk!")
        else:
            st.success("✅ Low Failure Risk")


# ================= WATER =================
with col2:
    st.subheader("Water Pipeline Failure Prediction")

    pressure = st.slider("Pressure (bar)", 1, 20, 8)
    flow = st.slider("Flow Rate (L/s)", 10, 200, 80)
    temp = st.slider("Temperature (°C)", 0, 60, 25)
    burst = st.selectbox("Burst Status", [0,1])

    if st.button("Predict Pipeline Risk"):

        pred = water_model.predict(
            [[pressure, flow, temp, burst]]
        )[0]

        if pred == 1:
            st.error("⚠️ High Failure Risk!")
        else:
            st.success("✅ Low Failure Risk")
