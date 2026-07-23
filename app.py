import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Now, configuring the page
st.set_page_config(
    page_title="Pregnancy Risk Detector",
    page_icon="🤰",
    layout="wide"
)

# Loading the CSS

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# Sidebar

with st.sidebar:

    st.title("🤰 Pregnancy Risk Detector")

    st.markdown("---")

    st.markdown("### 📌 About")

    st.markdown(
        """
        ### AI-Powered Maternal Health Risk Prediction

        This application uses a **Random Forest Machine Learning model**
        to predict whether a pregnancy is **High Risk** or **Low Risk**.

        Enter the patient's information below and click **Predict Risk Level**.
        """
    )

    st.markdown("---")

    st.markdown("### 🧠 Model")

    st.success("Random Forest Classifier")

    st.markdown("### 📈 Performance")

    st.metric("Accuracy", "99.58%")

    st.metric("Precision", "100%")

    st.metric("Recall", "98.95%")

    st.metric("F1 Score", "99.47%")

    st.markdown("---")

    st.info(
        "💡 This application is intended for educational purposes only and should not replace professional medical advice."
    )

st.divider()

# Now, loading the trained models
@st.cache_resource
def load_models():

    with open("models/pregnancy_risk_detector.pkl", "rb") as file:     # using with open, automatically closes it after reading it.
        model = pickle.load(file)

    # Load Standard Scaler
    with open("models/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Load Simple Imputer
    with open("models/imputer.pkl", "rb") as file:
        imputer = pickle.load(file)

    return model, scaler, imputer

model, scaler, imputer = load_models()

# Now creating the Patient Input Tabs
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

    
    # Because these are numerical values with known ranges from dataset, 
    # from EDA, got the Minimum and Maximum.
    

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

predict_button = st.button("🔍 Predict Risk Level", use_container_width=True)

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
    with st.expander("📋 View Patient Information:"):
        st.dataframe(input_data)

    # Now, Preprocessing
   
    # Fill missing values using the learned medians
    input_data = imputer.transform(input_data)

    # Scale using the learned mean and standard deviation
    input_data = scaler.transform(input_data)

    # Prediction

    with st.spinner("Predicting Risk Level..."):

        prediction = model.predict(input_data)[0]       # It gives now the label as --> Prediction = High or Low

        confidence = model.predict_proba(input_data).max() * 100        # Confidence

    # Now, showing the result
    col1, col2 = st.columns([2,1])

    with col1:

        if prediction == "High":
            st.error("⚠️ High Risk Pregnancy")

        else:
            st.success("✅ Low Risk Pregnancy")

    with col2:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Confidence")

        st.progress(confidence / 100)       # Progress Bar
        st.caption(f"{confidence:.2f}% Confidence")

    st.divider()

    st.subheader("🩺 Clinical Interpretation")

    if prediction == "High":

        st.warning(
        """
        The model predicts a **High Risk Pregnancy**.

        Please consult a qualified healthcare professional for further assessment.
        This prediction should not be considered a medical diagnosis.
        """
    )

    else:

        st.success(
            """
            The model predicts a **Low Risk Pregnancy**.

            Continue regular prenatal care and follow your healthcare provider's advice.
            """
        )

    st.divider()

    st.header("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "99.58%")

    c2.metric("Precision", "100%")

    c3.metric("Recall", "98.95%")

    c4.metric("F1 Score", "99.47%")

    # Feature Importance Chart

    st.divider()

    st.header("📈 Feature Importance")

    importance = pd.DataFrame({

        "Feature":[
            "Preexisting Diabetes",
            "Blood Sugar",
            "BMI",
            "Heart Rate",
            "Gestational Diabetes",
            "Mental Health",
            "Previous Complications",
            "Age",
            "Diastolic BP",
            "Systolic BP",
            "Body Temperature"
        ],

        "Importance":[
            0.223632,
            0.213698,
            0.149055,
            0.132830,
            0.089276,
            0.089119,
            0.036210,
            0.021560,
            0.019857,
            0.018414,
            0.006349
        ]

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Random Forest Feature Importance"
    )

    fig.update_layout(
        template = "plotly_white",
        title_x = 0.25
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Footer

st.divider()

st.markdown(
"""
<center>

### 👨‍💻 Developed by Mohd Sami

Computer Science (AI & ML)

Python • Scikit-Learn • Streamlit • Random Forest

</center>
""",
unsafe_allow_html=True
)