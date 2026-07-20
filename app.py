import streamlit as st
import pickle
import pandas as pd

# Now, configuring the page
st.set_page_config(
    page_title="Pregnancy Risk Detector",
    page_icon="🤰",
    layout="wide"
)

# Now, loading the model
with open("models/pregnancy_risk_detector.pkl", "rb") as file:     # using with open, automatically closes it after reading it.
    model = pickle.load(file)

st.title("🤰 Pregnancy Risk Detector")
st.write(       # Discription
    """
    Predict whether a pregnancy is **High Risk** or **Low Risk**
    using a trained Random Forest Machine Learning model.
    """
)

st.divider()


