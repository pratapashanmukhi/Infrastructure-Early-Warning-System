import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

st.title("🏗️ Infrastructure Early Warning System")
st.write("Predict failure risks in bridges and water pipelines using Machine Learning.")

# ---------- LOAD BRIDGE DATA ----------
bridge = pd.read_csv("bridge.csv")
bridge.columns = bridge.columns.str.strip()

# encode safely
if "Material_Type" in bridge.columns:
    bridge["Material_Type"] = bridge["Material_Type"].map({"Concrete":0,"Steel":1})

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

# ---------- LOAD WATER DATA (SUPER SAFE FIX) ----------
water = pd.read_csv("water.csv")
water.columns = water.columns.str.strip()

# convert ALL columns except target to numbers
for col in water.columns:
    if col != "failure" and col != "infrastructure_type":
        water[col] = pd.to_numeric(water[col], errors="coerce")

water = water.dropna()

X_water = water.drop(["failure","infrastructure_type"], axis=1)
y_water = water["failure"]

water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# ---------- UI ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bridge Failure Prediction")
    age = st.slider("Bridge Age",10,100,50)
    traffic = st.slider("Traffic Volume",100,5000,2000)
    material = st.selectbox("Material",["Concrete","Steel"])
    maint = st.selectbox("Maintenance",["No-Maintenance","Bi-Annual","Annual"])

    if st.button("Predict Bridge Risk"):
        mat = 0 if material=="Concrete" else 1
        m = {"No-Maintenance":0,"Bi-Annual":1,"Annual":2}[maint]
        pred = bridge_model.predict([[age,traffic,mat,m]])
        st.success("High Risk" if pred[0]==1 else "Low Risk")

with col2:
    st.subheader("Water Pipeline Prediction")
    pressure = st.slider("Pressure",1,20,8)
    flow = st.slider("Flow Rate",10,200,80)
    temp = st.slider("Temperature",0,50,25)

    if st.button("Predict Pipeline Risk"):
        pred = water_model.predict([[pressure,flow,temp]])
        st.success("High Risk" if pred[0]==1 else "Low Risk")
