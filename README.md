# Infrastructure-Early-Warning-System
Machine Learning project for predicting infrastructure failures
# Infrastructure Early Warning System: End-to-End Machine Learning Project

This project implements an applied machine learning system to predict failures in critical infrastructure systems such as bridges and water pipelines. The primary goal is to build an early warning system that helps authorities move from reactive maintenance to predictive maintenance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wKwQudpaVrTSfYcyuc2XujN4qWkVjw86?usp=sharing)


---

## 📝 Project Description

Infrastructure failures like bridge collapses and pipeline bursts can cause massive economic damage and risk to human life. Traditional inspection methods are time-based and manual.

This project uses real-world datasets and Machine Learning to automatically predict failure risk based on historical and sensor data. By learning patterns in structural conditions and environmental parameters, the system provides early warnings before failures occur.

---

## 📂 Datasets Used

### 1. Bridge Dataset  
Features:
- Age of Bridge  
- Traffic Volume  
- Material Type  
- Maintenance Level  
- Failure (Target)

### 2. Water Pipeline Dataset  
Features:
- Pressure  
- Flow Rate  
- Temperature  
- Burst Status  
- Failure (Target)

Source: Kaggle / Simulated infrastructure datasets

---

## 🛠️ Technical Skills Demonstrated

- Python (Pandas, NumPy)
- Scikit-learn
- Machine Learning (Random Forest)
- Data Cleaning & Encoding
- Feature Engineering
- Model Evaluation
- Predictive Analytics

---

## 📈 Analytical Workflow


I followed a structured machine learning pipeline:

### 1️⃣ Data Loading  
Imported bridge and water datasets into Pandas.

### 2️⃣ Data Cleaning  
Removed irrelevant columns and handled categorical variables using Label Encoding.

### 3️⃣ Feature Engineering  
Converted material types and maintenance levels into numerical form.

### 4️⃣ Train-Test Split  
Split datasets into 80% training and 20% testing.

### 5️⃣ Model Training  
Trained two separate Random Forest models:
- Bridge Failure Model  
- Water Failure Model  

### 6️⃣ Model Evaluation  
Evaluated accuracy on unseen test data.

### 7️⃣ Prediction  
Tested system using real sample inputs.

---

## 🔍 Algorithms Used

- Random Forest Classifier

---

## 📊 Output

The system predicts:

- `1` → High failure risk  
- `0` → Low failure risk  

Example:
```python
Bridge prediction: [1]
Water prediction: [0]
  ```
---
📧 Email: pratapashanumukhi@gmail.com  
🔗 LinkedIn: [Shanmukhi Pratapa](https://www.linkedin.com/in/shanmukhi-pratapa-6a4484336)

Feel free to connect with me for collaborations, internships, and machine learning projects.



