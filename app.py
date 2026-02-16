import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="Infrastructure Early Warning", layout="wide")

# --------------------------
# Simple Login System
# --------------------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# --------------------------
# Load datasets
# --------------------------
bridge = pd.read_csv("bridge.csv")
water = pd.read_csv("water.csv")

# Encode bridge
bridge["Material_Type"] = bridge["Material_Type"].map({"Concrete": 0, "Steel": 1})
bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map({
    "No-Maintenance": 0,
    "Bi-Annual": 1,
    "Annual": 2
})

# Encode water (if needed)
if "Burst_Status" in water.columns:
    water["Burst_Status"] = water["Burst_Status"].astype(int)

# --------------------------
# Train models
# --------------------------
X_bridge = bridge.drop(columns=["failure", "infrastructure_type"])
y_bridge = bridge["failure"]
bridge_model = RandomForestClassifier()
bridge_model.fit(X_bridge, y_bridge)

X_water = water.drop(columns=["failure", "infrastructure_type"])
y_water = water["failure"]
water_model = RandomForestClassifier()
water_model.fit(X_water, y_water)

# --------------------------
# UI
# --------------------------
st.title("🏗 Infrastructure Early Warning System")
st.markdown("Predict failure risks using Machine Learning")

tab1, tab2, tab3 = st.tabs(["Prediction", "Analytics", "Map Monitoring"])

# --------------------------
# TAB 1: Prediction
# --------------------------
with tab1:
    col1, col2 = st.columns(2)

    # Bridge
    with col1:
        st.subheader("Bridge Failure Prediction")

        age = st.slider("Bridge Age", 0, 100, 40)
        traffic = st.slider("Traffic Volume", 0, 5000, 2000)
        material = st.selectbox("Material Type", ["Concrete", "Steel"])
        maintenance = st.selectbox("Maintenance Level",
                                   ["No-Maintenance", "Bi-Annual", "Annual"])

        if st.button("Predict Bridge Risk"):
            material_val = 0 if material == "Concrete" else 1
            maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
            maintenance_val = maintenance_map[maintenance]

            input_data = [[age, traffic, material_val, maintenance_val]]
            prob = bridge_model.predict_proba(input_data)[0][1]
            risk = prob * 100

            st.metric("Risk Percentage", f"{risk:.2f}%")

            if risk > 60:
                st.error("High Failure Risk")
            else:
                st.success("Low Failure Risk")

    # Water
    with col2:
        st.subheader("Water Pipeline Failure Prediction")

        pressure = st.slider("Pressure", 0, 10, 5)
        flow = st.slider("Flow Rate", 0, 200, 100)
        temp = st.slider("Temperature", 0, 100, 25)
        burst = st.selectbox("Burst Status", [0, 1])

        if st.button("Predict Pipeline Risk"):
            input_data = [[pressure, flow, temp, burst]]
            prob = water_model.predict_proba(input_data)[0][1]
            risk = prob * 100

            st.metric("Risk Percentage", f"{risk:.2f}%")

            if risk > 60:
                st.error("High Failure Risk")
            else:
                st.success("Low Failure Risk")

# --------------------------
# TAB 2: Analytics Dashboard
# --------------------------
with tab2:
    st.subheader("Infrastructure Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(bridge, x="Age_of_Bridge",
                            title="Bridge Age Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(water, x="Pressure",
                            title="Water Pressure Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(bridge,
                      x="Age_of_Bridge",
                      y="Traffic_Volume",
                      color="failure",
                      title="Bridge Risk Scatter")
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# TAB 3: Map Monitoring
# --------------------------
with tab3:
    st.subheader("Live Infrastructure Map")

    # Create dummy map data
    map_data = pd.DataFrame({
        "lat": np.random.uniform(17.0, 18.0, 50),
        "lon": np.random.uniform(78.0, 79.0, 50)
    })

    st.map(map_data)
