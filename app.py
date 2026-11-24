import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger would survive the Titanic disaster.")

st.subheader("Enter Passenger Details")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Gender", ["Female", "Male"])
age = st.slider("Passenger Age", 1, 80, 30)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=30.0)

if st.button("Predict Survival"):
    try:
        model = load_model()

        sex_value = 1 if sex == "Female" else 0

        data = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=["pclass", "sex", "age", "fare"]
        )

        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]

        survival = probabilities[list(model.classes_).index(1)]
        death = probabilities[list(model.classes_).index(0)]

        if prediction == 1:
            st.success("Passenger is predicted to SURVIVE!")
        else:
            st.error("Passenger is predicted NOT TO SURVIVE.")

        st.metric("Survival Probability", f"{survival:.2%}")
        st.metric("Non-Survival Probability", f"{death:.2%}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
