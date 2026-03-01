import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Infrastructure Failure Detection", layout="centered")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.main {
    background-color: #eaf1f8;
}
.block-container {
    max-width: 850px;
    margin: auto;
    padding-top: 2rem;
}
h1 {
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Infrastructure Failure Detection System")

# ---------------- LOAD DATA ----------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# ---------------- FIX COLUMN NAMES ----------------
bridge.columns = bridge.columns.str.strip()
water.columns = water.columns.str.strip()

# ---------------- ENCODE BRIDGE DATA ----------------
bridge["Material_Type"] = bridge["Material_Type"].map(
    {"Concrete": 0, "Steel": 1}
)

bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map(
    {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
)

X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# ---------------- ENCODE WATER DATA ----------------
# (adjust column names if needed based on your dataset)

water.columns = water.columns.str.strip()

if "Material_Type" in water.columns:
    water["Material_Type"] = water["Material_Type"].map(
        {"Concrete": 0, "Steel": 1}
    )

X_water = water.drop(["failure", "infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# ---------------- SELECT INFRASTRUCTURE ----------------
infra = st.selectbox(
    "Select Infrastructure",
    ["Bridge", "Water Pipeline"]
)

# ---------------- BRIDGE SECTION ----------------
if infra == "Bridge":

    st.image("bridge.jpg", use_container_width=True)

    st.subheader("Bridge Inputs")

    age = st.slider("Bridge Age (years)", 0, 100, 50)
    traffic = st.number_input("Traffic Volume", value=2000)

    material = st.selectbox("Material Type", ["Concrete", "Steel"])
    maintenance = st.selectbox(
        "Maintenance Level",
        ["No-Maintenance", "Bi-Annual", "Annual"]
    )

    if st.button("Predict Bridge Risk"):

        material_val = 0 if material == "Concrete" else 1

        maintenance_map = {
            "No-Maintenance": 0,
            "Bi-Annual": 1,
            "Annual": 2
        }

        input_data = pd.DataFrame([[
            age,
            traffic,
            material_val,
            maintenance_map[maintenance]
        ]], columns=X_bridge.columns)

        prediction = bridge_model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Failure Risk Detected")
        else:
            st.success("✅ Low Failure Risk")

# ---------------- WATER PIPELINE SECTION ----------------
else:

    st.image("water.jpg", use_container_width=True)

    st.subheader("Water Pipeline Inputs")

    pressure = st.slider("Pressure (bar)", 0, 20, 8)
    flow = st.number_input("Flow Rate (L/s)", value=80)
    temperature = st.slider("Temperature (°C)", 0, 60, 25)

    if st.button("Predict Pipeline Risk"):

        input_values = [pressure, flow, temperature]

        # if dataset has extra feature add here
        input_df = pd.DataFrame(
            [input_values],
            columns=X_water.columns[:len(input_values)]
        )

        prediction = water_model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ High Pipeline Failure Risk")
        else:
            st.success("✅ Low Pipeline Failure Risk")
