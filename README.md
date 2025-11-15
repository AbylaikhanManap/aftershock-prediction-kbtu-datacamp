# 🇰🇿 KBTU DataCamp 2025 — Aftershock Prediction (1st Place)

This repository contains my solution to the national machine learning competition **KBTU DataCamp 2025**, where I ranked **1st place** among ~70 teams from across Kazakhstan.  
While most teams consisted of up to four participants, I deliberately decided to compete **alone** to test myself under real pressure and gain independent experience.

---

# 🧠 Problem Overview
The task was to build a machine learning model capable of predicting the **first aftershock** following a major earthquake.  
Given historical seismic data describing the main earthquake:

- time (year, month, day, hour, minute, second)
- latitude & longitude  
- depth (km)
- energy class  

the goal was to forecast **ten aftershock parameters**:

- year, month, day  
- hour, minute, second (float with milliseconds)
- latitude, longitude  
- depth  
- energy class  

The competition used a **private leaderboard** on Kaggle with a hidden test set and **MAE** as the evaluation metric.

I placed **Top-10** in the Kaggle online stage → advanced to the onsite 24-hour final → and won **1st place** after presenting my full pipeline to the academic jury.

---

# 📊 ML Pipeline Summary

My approach combined **feature engineering**, **deep learning**, and **gradient boosting**, followed by a **stacking ensemble**.

## **1. Feature Engineering**
I generated additional temporal and cyclical features:

- `day_of_year` — seasonal patterns  
- `main_eq_time_decimal` — continuous time representation  
- `cos_hour`, `sin_hour` — circular encoding for time-of-day  
- combined timestamps into a single `datetime` object  
- removed infinities and missing values  

These features allowed the model to learn periodic and seasonal behaviours of seismic activity.

---

## **2. Base Models**
I trained three diverse models:

### **🔹 TabNetRegressor**
- excellent for tabular data  
- sparse attention  
- handles complicated nonlinear patterns  
- tuned with `n_d=16`, `n_steps=3`, `gamma=1.5`, etc.

### **🔹 CatBoostRegressor (MultiOutputRegressor wrapper)**
- robust to noise  
- strong performance on tabular datasets  
- GPU-accelerated training  

### **🔹 Custom PyTorch Neural Network**

Architecture:
Linear(14 → 128) → ReLU → Dropout(0.2)
Linear(128 → 64) → ReLU
Linear(64 → 10)

- trained 50 epochs  
- Adam optimizer  
- MSE loss  

---

## **3. Cross-Validation & OOF Predictions**
- Used **5-fold KFold**
- Generated out-of-fold predictions for stacking:
  - TabNet  
  - CatBoost  
  - PyTorch  

This provides unbiased meta-features for the final model.

---

## **4. Stacking Ensemble**
The meta-model:
RidgeCV is stable and works well for linear blending across multiple targets.

---

## **5. Final Training & Submission**
- Retrained base models on the full training dataset  
- Predicted on test set  
- Combined predictions through the meta-model  
- Inverse-transformed outputs  
- Rounded results according to competition format  
- Saved final submission: `submission_stacked.csv`

This file ranked **#1** on the private leaderboard.

---

# 📁 Repository Structure
aftershock-prediction-kbtu-datacamp/
│
├── model.ipynb # Full ML pipeline
├── stacking_pipeline.py # Clean version (optional)
├── neural_network.py # PyTorch model (optional)
├── tabnet_train.py # TabNet training (optional)
├── results/
│ ├── submission_stacked.csv
│ ├── feature_importance.png (optional)
│ └── training_curves.png (optional)
│
└── README.md

---

# 🌍 Real-World Impact
This project showed how AI can extract meaningful signal from highly noisy geophysical data.  
Models like this could support:

- early warning systems  
- disaster response  
- evacuation planning  
- seismic risk mitigation  

This competition strongly influenced my decision to study **AI Engineering** and develop robust ML systems for real-world applications.

---

# 📬 Contact
Feel free to reach out if you want to discuss this project or improvements.
