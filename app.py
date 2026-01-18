import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger survived the Titanic disaster.")


# Load Trained Model

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()


# User Inputs

st.subheader("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Queenstown", "Southampton"]
)

age_group = st.selectbox(
    "Age Group",
    ["Child", "Teen", "Adult", "Senior"]
)

fare_bin = st.selectbox(
    "Fare Category",
    ["Very Low", "Low", "High", "Very High"]
)

is_alone = st.checkbox("Traveling Alone?")
is_alone = int(is_alone)


# Create Input DataFrame

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Embarked": embarked,
    "AgeGroup": age_group,
    "FareBin": fare_bin,
    "IsAlone": is_alone
}])


# Prediction (MANUAL DUMMY MATCHING)

if st.button("Predict Survival"):

    # Convert categorical inputs to dummies
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns expected by the model
    for col in model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure column order matches training
    input_encoded = input_encoded[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability:.2%})")


# Footer

st.markdown("---")
st.caption("Random Forest model trained with stratified cross-validation.")
