# 🚢 Titanic Survival Prediction

🔗 **Live Streamlit App:**  
https://titanicsurvivalprediction-prasadshetty.streamlit.app/

An **end-to-end machine learning project** that predicts whether a passenger survived the Titanic disaster.  
This project demonstrates the complete data science lifecycle — from **EDA and feature engineering** to **model tuning and deployment** using Streamlit.

---

## 🧠 Project Overview

The goal of this project is to analyze Titanic passenger data and build a robust predictive model to estimate survival probability.  
It showcases practical, real-world data science skills including preprocessing, modeling, evaluation, and deployment.

This repository is supported by **detailed PDF documentation** explaining every stage of development.

📄 **Project Documentation:**  
`Titanic_Survival_Prediction_Model_Development_and_Deployment.pdf`

---

## 📊 Dataset

The project uses the classic **Titanic Dataset**, containing demographic and travel-related passenger information.

**Target Variable**
- `Survived`  
  - 0 → Did Not Survive  
  - 1 → Survived  

---

## 🧪 Key Features Used

| Feature | Description |
|------|-------------|
| `Pclass` | Passenger class (1, 2, 3) |
| `Sex` | Gender of passenger |
| `Embarked` | Port of embarkation |
| `AgeGroup` | Binned age category (Child, Teen, Adult, Senior) |
| `FareBin` | Fare category using quantile-based binning |
| `IsAlone` | Indicates whether the passenger traveled alone |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand distributions, missing values, and feature relationships.

**Highlights:**
- Survival analysis by sex, passenger class, age, fare, and embarkation port
- Identification of class imbalance
- Analysis of skewed fare distribution
- Feature interaction analysis (e.g., Embarked vs Pclass)

---

## 🧹 Data Cleaning & Feature Engineering

- Handled missing values:
  - Age → median strategy
  - Embarked → mode
- Removed irrelevant and high-cardinality features:
  - PassengerId, Name, Ticket, Cabin
- Engineered new features:
  - `AgeGroup`
  - `FareBin` (qcut)
  - `IsAlone`
- Prepared categorical variables for modeling and deployment

---

## 🤖 Model Training

**Models Trained:**
- Logistic Regression (baseline)
- Random Forest Classifier (final model)

**Training Strategy:**
- Stratified train-test split
- Hyperparameter tuning using **GridSearchCV**
- Stratified cross-validation

---

## 📈 Model Evaluation

**Evaluation Metric:** ROC-AUC

| Model | ROC-AUC |
|----|----|
| Logistic Regression | ~0.85 |
| Random Forest | ~0.86 |

The **Random Forest model** achieved the best performance and was selected for deployment.

---

## 🌐 Streamlit Application

The trained model is deployed as an interactive Streamlit application.

**App Features:**
- User-friendly dropdowns and checkboxes
- Real-time survival prediction
- Probability score output
- Manual dummy-column alignment to ensure feature consistency

🔗 **Live App:**  
https://titanicsurvivalprediction-prasadshetty.streamlit.app/

---

## 📁 Complete Project Structure

```
Titanic-Survival-Prediction/
│
├── 📊 TITANIC SURVIVAL PREDICTION.ipynb
│   └── EDA, feature engineering, model training & evaluation
│
├── 📄 Titanic-Dataset.csv
│   └── Original dataset
│
├── 🧠 titanic_model.pkl
│   └── Trained Random Forest model
│
├── 🌐 app.py
│   └── Streamlit application
│
├── 📦 requirements.txt
│   └── Project dependencies
│
├── 📝 README.md
│   └── Project overview and usage instructions 
│
└── 📘 Titanic_Survival_Prediction_Model_Development_and_Deployment.pdf
    └──Detailed project documentation
```

---

## 🧰 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Durga Prasad**  
- GitHub: https://github.com/shettyprasad-git  
- LinkedIn: https://www.linkedin.com/in/durgaprasadshetty  

