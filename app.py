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
st.write("Predict passenger survival based on travel and personal details.")


# Load Trained Model

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")

model = load_model()


# User Inputs

st.subheader("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    options=["male", "female"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=["Cherbourg", "Queenstown", "Southampton"]
)

age_group = st.selectbox(
    "Age Group",
    options=["Child", "Teen", "Adult", "Senior"]
)

fare_bin = st.selectbox(
    "Fare Category",
    options=["Very Low", "Low", "High", "Very High"]
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


# Prediction

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability:.2%})")


# Footer

st.markdown("---")
st.caption("Model trained using Random Forest with stratified cross-validation.")
