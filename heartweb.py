import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('D:/trainedmodel/trained_model.sav', 'rb'))

def heart_disease_prediction(input_data):
    """
    Predict heart disease based on input data
    """
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'

def main():
    st.title("Heart Disease Prediction System")

    # Get input data from the user
    age = st.number_input('Age of person')
    sex = st.number_input('Gender')
    cp = st.number_input('Blood Pressure value')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholesterol Level')
    fbs = st.number_input('Fasting Blood Sugar')
    restecg = st.number_input('Resting ECG')
    thalach = st.number_input('Maximum Heart Rate Achieved')
    exang = st.number_input('Exercise Induced Angina')
    oldpeak = st.number_input('Old Peak')
    slope = st.number_input('Slope of the Peak Exercise ST Segment')
    ca = st.number_input('Number of Major Vessels Colored by Flourosopy')
    thal = st.number_input('Thalassemia')

    # Create a button for prediction
    if st.button('Heart Disease Test Result'):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        diagnosis = heart_disease_prediction(input_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()