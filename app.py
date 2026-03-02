import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Infrastructure Early Warning System",
    layout="wide"
)

st.title("🏗️ Infrastructure Early Warning System")
st.markdown("Predict failure risks in **bridges** and **water pipelines** using Machine Learning.")

# ---------------- LOAD IMAGES ----------------
bridge_img = Image.open("bridge.jpg")
water_img = Image.open("water.jpg")

# ---------------- BRIDGE DATA ----------------
bridge = pd.read_csv("bridge.csv")
bridge.columns = bridge.columns.str.strip()

if "Material_Type" in bridge.columns:
    bridge["Material_Type"] = bridge["Material_Type"].map({
        "Concrete":0,
        "Steel":1
    })

if "Maintenance_Level" in bridge.columns:
    bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map({
        "No-Maintenance":0,
        "Bi-Annual":1,
        "Annual":2
    })

bridge = bridge.dropna()

X_bridge = bridge.drop(["failure","infrastructure_type"], axis=1)
y_bridge = bridge["failure"]

bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

# ---------------- WATER DATA ----------------
water = pd.read_csv("water.csv")
water.columns = water.columns.str.strip()

# convert numeric safely
for col in water.columns:
    if col not in ["failure","infrastructure_type"]:
        water[col] = pd.to_numeric(water[col], errors="coerce")

water = water.dropna()

X_water = water.drop(["failure","infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# ---------------- UI ----------------
col1, col2 = st.columns(2)

# ========= BRIDGE =========
with col1:

    st.subheader("🌉 Bridge Failure Prediction")
    st.image(bridge_img, use_container_width=True)

    age = st.slider("Bridge Age", 10, 100, 50)
    traffic = st.slider("Traffic Volume", 100, 5000, 2000)
    material = st.selectbox("Material", ["Concrete","Steel"])
    maint = st.selectbox("Maintenance", ["No-Maintenance","Bi-Annual","Annual"])

    if st.button("Predict Bridge Risk"):

        mat = 0 if material=="Concrete" else 1
        m = {"No-Maintenance":0,"Bi-Annual":1,"Annual":2}[maint]

        pred = bridge_model.predict([[age,traffic,mat,m]])

        if pred[0] == 1:
            st.error("⚠️ High Bridge Failure Risk")
        else:
            st.success("✅ Low Bridge Failure Risk")

# ========= WATER =========
with col2:

    st.subheader("💧 Water Pipeline Prediction")
    st.image(water_img, use_container_width=True)

    pressure = st.slider("Pressure", 1, 20, 8)
    flow = st.slider("Flow Rate", 10, 200, 80)
    temp = st.slider("Temperature", 0, 50, 25)

    if st.button("Predict Pipeline Risk"):

        pred = water_model.predict([[pressure, flow, temp]])

        if pred[0] == 1:
            st.error("⚠️ High Pipeline Failure Risk")
        else:
            st.success("✅ Low Pipeline Failure Risk")



