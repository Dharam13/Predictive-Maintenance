# 🛠️ Predictive Maintenance System using XGBoost

![Maintenance Banner](https://img.shields.io/badge/Machine_Learning-XGBoost-blue)  
![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)  
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview

Predictive Maintenance is a proactive approach that uses data and advanced machine learning techniques to predict equipment failures before they occur. This project builds a robust Time-To-Failure (TTF) prediction pipeline using an **Enhanced XGBoost Multi-Output Regressor**.

The model predicts the number of weeks remaining before four different machine components are likely to fail. The solution is optimized for **short-term accuracy**, making it suitable for real-world industrial applications where quick preventive action is critical.

---

## 📊 Dataset Description

The dataset (`TTF_Dataset_Weeks.csv`) contains telemetry, error, and maintenance records from manufacturing machines, including:

- **Sensor Data**: `volt`, `rotate`, `pressure`, `vibration`
- **Operational Info**: `model`, `age`, `error_count`
- **Maintenance Logs**: `days_since_comp1_maint` to `comp4`
- **Target Variables**: `ttf_comp1_weeks` to `ttf_comp4_weeks` (Time To Failure)

---

## 🧠 Machine Learning Approach

### ⚙️ Key Steps:

1. **Stratified Sampling**  
   - 200,000 representative samples selected based on multiple criteria (`model`, `age`, error level, etc.)

2. **Feature Engineering**  
   - Excludes `machineID` and `failure_within_48h` for cleaner predictions  
   - Label encoding and scaling applied

3. **Model Architecture**  
   - `XGBoostRegressor` wrapped in `MultiOutputRegressor`  
   - Custom tuned hyperparameters for better performance under 10-week predictions

4. **Evaluation Metrics**  
   - RMSE, MAE, R², MAPE (with special attention to short-term MAPE)  
   - Short-term predictions are emphasized (e.g., `ttf < 10 weeks`)

5. **Model Packaging**  
   - Saves full pipeline (`scaler`, `label encoders`, trained model) using `joblib`

---

## 🔍 Results Summary

| Metric                      | Average Value |
|----------------------------|---------------|
| Test R²                    | ~0.95         |
| Test MAPE (%)              | ~11.25%       |
| Short-Term MAE (<10 weeks) | ~0.91 weeks   |

✅ The model excels in **short-term TTF prediction**, ideal for maintenance scheduling and avoiding unexpected breakdowns.

---


## 👥 Contributors

- **Dharam Patel**
- **Ayesha Patel**  
- **Vrunda Patel**




