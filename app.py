import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.markdown(
    "Predict failure risks in bridges and water pipelines using **Machine Learning**."
)

# -------------------------
# Load datasets
# -------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# -------------------------
# Clean Bridge dataset
# -------------------------
bridge["Material_Type"] = bridge["Material_Type"].map({"Concrete": 0, "Steel": 1})
bridge["Maintenance"] = bridge["Maintenance"].map({
    "No-Maintenance": 0,
    "Bi-Annual": 1,
    "Annual": 2
})

X_bridge = bridge[[
    "Age_of_Bridge",
    "Traffic_Volume",
    "Material_Type",
    "Maintenance"
]]
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# -------------------------
# Clean Water dataset
# -------------------------
for col in water.columns:
    if water[col].dtype == "object":
        water[col] = water[col].astype("category").cat.codes

X_water = water.drop("failure", axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# -------------------------
# Bridge UI
# -------------------------
st.markdown("---")
st.header("Bridge Failure Prediction")

age = st.number_input("Bridge Age", 0, 200, 50)
traffic = st.number_input("Traffic Volume", 0, 10000, 2000)
material = st.selectbox("Material Type", ["Concrete", "Steel"])
maintenance = st.selectbox("Maintenance Level", ["No-Maintenance", "Bi-Annual", "Annual"])

material_val = 0 if material == "Concrete" else 1
maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
maintenance_val = maintenance_map[maintenance]

if st.button("Predict Bridge Risk"):
    sample = pd.DataFrame([[age, traffic, material_val, maintenance_val]],
                          columns=X_bridge.columns)
    prediction = bridge_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")

# -------------------------
# Water UI
# -------------------------
st.markdown("---")
st.header("Water Pipeline Failure Prediction")

inputs = []
for col in X_water.columns:
    val = st.number_input(col, 0.0, 10000.0, 10.0)
    inputs.append(val)

if st.button("Predict Water Risk"):
    sample = pd.DataFrame([inputs], columns=X_water.columns)
    prediction = water_model.predict(sample)[0]

    if prediction == 1:
        st.error("High Failure Risk")
    else:
        st.success("Low Failure Risk")

