import streamlit as st
import pandas as pd
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Infrastructure Early Warning System", layout="wide")

# -------------------------------
# Database setup
# -------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# -------------------------------
# Password hashing
# -------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -------------------------------
# Auth functions
# -------------------------------
def create_user(username, password):
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hash_password(password)))
    return c.fetchone()

# -------------------------------
# Session state
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------------
# Login / Signup UI
# -------------------------------
def login_signup():
    st.title("🔐 User Authentication")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # LOGIN
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            result = login_user(username, password)
            if result:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        st.subheader("Create New Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            if create_user(new_user, new_pass):
                st.success("Account created! You can now log in.")
            else:
                st.error("Username already exists")

# -------------------------------
# Main Dashboard
# -------------------------------
def dashboard():
    st.title("🏗 Infrastructure Early Warning System")
    st.write(f"Welcome, **{st.session_state.username}**")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    # Load datasets
    bridge = pd.read_csv("bridge.csv")
    water = pd.read_csv("water.csv")

    # Encode bridge data
    bridge["Material_Type"] = bridge["Material_Type"].map({"Concrete": 0, "Steel": 1})
    bridge["Maintenance_Level"] = bridge["Maintenance_Level"].map(
        {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
    )

    X_bridge = bridge.drop(["failure", "infrastructure_type"], axis=1)
    y_bridge = bridge["failure"]

    bridge_model = RandomForestClassifier()
    bridge_model.fit(X_bridge, y_bridge)

    # Prepare water model
    X_water = water.drop(["failure", "infrastructure_type"], axis=1)
    y_water = water["failure"]

    water_model = RandomForestClassifier()
    water_model.fit(X_water, y_water)

    # Layout
    col1, col2 = st.columns(2)

    # ---------------- Bridge Section ----------------
    with col1:
        st.subheader("Bridge Failure Prediction")

        age = st.slider("Bridge Age (years)", 1, 100, 50)
        traffic = st.slider("Traffic Volume", 100, 5000, 2000)

        material = st.selectbox("Material Type", ["Concrete", "Steel"])
        maintenance = st.selectbox("Maintenance Level",
                                   ["No-Maintenance", "Bi-Annual", "Annual"])

        material_val = 0 if material == "Concrete" else 1
        maintenance_map = {"No-Maintenance": 0, "Bi-Annual": 1, "Annual": 2}
        maintenance_val = maintenance_map[maintenance]

        if st.button("Predict Bridge Risk"):
            pred = bridge_model.predict([[age, traffic, material_val, maintenance_val]])[0]
            prob = bridge_model.predict_proba([[age, traffic, material_val, maintenance_val]])[0][1]

            st.metric("Risk Probability", f"{prob*100:.2f}%")

            if pred == 1:
                st.error("High Risk of Failure")
            else:
                st.success("Low Risk")

    # ---------------- Water Section ----------------
    with col2:
        st.subheader("Water Pipeline Failure Prediction")

        pressure = st.slider("Pressure (bar)", 1, 15, 8)
        flow = st.slider("Flow Rate (L/s)", 10, 200, 80)
        temp = st.slider("Temperature (°C)", 0, 60, 25)
        burst = st.selectbox("Burst Status", [0, 1])

        if st.button("Predict Pipeline Risk"):
            pred = water_model.predict([[pressure, flow, temp, burst]])[0]
            prob = water_model.predict_proba([[pressure, flow, temp, burst]])[0][1]

            st.metric("Risk Probability", f"{prob*100:.2f}%")

            if pred == 1:
                st.error("High Risk of Failure")
            else:
                st.success("Low Risk")

# -------------------------------
# App Flow
# -------------------------------
if not st.session_state.logged_in:
    login_signup()
else:
    dashboard()
