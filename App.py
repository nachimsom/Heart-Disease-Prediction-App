import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("Heart_diseases_model.pkl")

# Title
st.title("Heart Disease Prediction App")

st.write(
    """
    This app predicts the *risk of heart disease* using a trained Machine Learning model.
    
    🩺 *Fill in the details in the sidebar and click Predict.*
    """
)
# Sidebar inputs
def user_input_features():
    st.sidebar.header("Patient Information")
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', (1, 0))
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral (mg/dl)', 100, 600, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar >120 mg/dl (1=True, 0=False)', (1, 0))
    restecg = st.sidebar.slider('Resting ECG Results (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1=Yes, 0=No)', (1, 0))
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.slider('Slope (0-2)', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thal (1=Normal;2=Fixed Defect;3=Reversable Defect)', 1, 3, 2)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Output")
    heart_disease = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"*Result:* {heart_disease}")
    
    st.subheader("Prediction Probability")
    st.write(prediction_proba)

    st.markdown(
    """
    ---
    Created by *Chimsom* as part of the 3MTT Knowledge Showcase 🎓
    """
)

