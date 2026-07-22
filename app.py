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

# Load Standard Scaler
with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load Simple Imputer
with open("models/imputer.pkl", "rb") as file:
    imputer = pickle.load(file)

st.title("🤰 Pregnancy Risk Detector")
st.write(       # Discription
    """
    Predict whether a pregnancy is **High Risk** or **Low Risk**
    using a trained Random Forest Machine Learning model.
    """
)

st.divider()

# Now creating the tabs
tab1, tab2, tab3 = st.tabs(
    [
        "🩺 Vitals",
        "📊 Health Metrics",
        "📋 Medical History"
    ]
)

# tab1 -- Vitals
with tab1:

    age = st.slider(
        "Age",
        min_value = 10,
        max_value = 65,
        value = 25
    )

    systolic_bp = st.slider(
        "Systolic BP",
        min_value = 70,
        max_value = 200,
        value = 120
    )

    diastolic_bp = st.slider(
        "Diastolic BP",
        min_value = 40,
        max_value = 140,
        value = 80
    )

    body_temp = st.slider(
        "Body Temperature",
        min_value = 97,
        max_value = 103,
        value = 98
    )

    heart_rate = st.slider(
        "Heart Rate",
        min_value = 58,
        max_value = 92,
        value = 76
    )

    """
    Because these are numerical values with known ranges from dataset, 
    from EDA, got the Minimum and Maximum.
    """

# tab2 -- Health Metrics
with tab2:

    blood_sugar = st.slider(
        "Blood Sugar (BS)",
        min_value = 3.0,
        max_value = 19.0,
        value = 7.0,
        step = 0.1
    )

    bmi = st.slider(
        "BMI",
        min_value = 0.0,
        max_value = 37.0,
        value = 23.0,
        step = 0.1      # Because Blood Sugar and BMI are floating-point values (with it --> 6.0, 6.1, 6.2...).
    )

    mental_health = st.selectbox(
        "Mental Health Issue",
        [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

# tab3 -- Medical History

with tab3:

    previous_complications = st.selectbox(
        "Previous Complications",
        [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

    preexisting_diabetes = st.selectbox(
        "Preexisting Diabetes",
        [0, 1],
        format_func= lambda x: "Yes" if x == 1 else "No"
    )

    gestational_diabetes = st.selectbox(
        "Gestational Diabetes",
        [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

    """
    Sliders and select boxes prevent invalid inputs.

    """

# Now, making the predictions

predict_button = st.button("🔍 Predict Risk Level")

# Now, Creating the Input DataFrame (converting the inputs into the DataFrame via pandas)

if predict_button:

    input_data = pd.DataFrame({
        "Age": [age],
        "Systolic BP": [systolic_bp],
        "Diastolic": [diastolic_bp],
        "BS": [blood_sugar],
        "Body Temp": [body_temp],
        "BMI": [bmi],
        "Previous Complications": [previous_complications],
        "Preexisting Diabetes": [preexisting_diabetes],
        "Gestational Diabetes": [gestational_diabetes],
        "Mental Health": [mental_health],
        "Heart Rate": [heart_rate]
    })

    # Now, before predicting checking the input
    with st.subheader("View Patient Information:"):
        st.dataframe(input_data)

    # Now, Preprocessing
   
    # Fill missing values using the learned medians
    input_data = imputer.transform(input_data)

    # Scale using the learned mean and standard deviation
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)[0]       # It gives now the label as --> Prediction = High or Low

    # Confidence
    confidence = model.predict_proba(input_data).max() * 100

    # Now, showing the result
    st.subheader("Prediction")

    if prediction == "High":
        st.error("⚠️ High Risk Pregnancy")

    else:
        st.success("✅ Low Risk Pregnancy")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )