import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Load areas from text file
try:
    with open('areas.txt', 'r') as file:
        area_list = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    st.error("The 'areas.txt' file is missing. Please add it to the working directory.")
    area_list = []

# Title of the app
st.title("CO2 Emission Prediction App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
Please select an area from the dropdown and enter the year to get the prediction.
""")

# Model Selection (get model names from the dictionary keys)
model_choice = st.selectbox("Select Model:", list(models.keys()))

# Dropdown for Area Selection
area = st.selectbox("Select the Area:", area_list)

# Input for Year
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

# Prediction Button
if st.button("Predict"):
    if area and year:
        try:
            # 1. Create DataFrame
            new_data = pd.DataFrame({'area': [area], 'year': [year]})
            
            # Preprocess the data using the original preprocessor
            new_data_transformed = preprocessor.transform(new_data)
            
            # 2. Get selected model
            model = models[model_choice]

            # 3. Make predictions
            prediction = model.predict(new_data_transformed)[0]  # Use transformed data

            # 4. Display result
            st.success(f"Predicted Total CO2 Emission for {area} in {year}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")
