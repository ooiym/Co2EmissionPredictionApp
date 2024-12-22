import streamlit as st
import pandas as pd
import joblib

# Load the trained LightGBM model
model = joblib.load('sgd_best_model.pkl')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Create encoder

# Title of the app
st.title("CO2 Emission Prediction App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
Please enter the values below to get the prediction.
""")

# Input form for Area and Year
area = st.text_input("Enter the Area (e.g., Country or Region):")
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

# Predict button
if st.button("Predict"):
    if area and year:
        try:
            # Preprocess the input data
            year_ordinal = pd.to_datetime(f"{int(year)}-01-01").toordinal()
            
            # One-Hot Encode the 'area'
            area_encoded = encoder.fit_transform(pd.DataFrame({'area': [area]})) 
            
            # Create input DataFrame with encoded 'area' and 'year'
            input_data = pd.DataFrame({'year': [year_ordinal]}) # DataFrame for year
            input_data = pd.concat([pd.DataFrame(area_encoded), input_data], axis=1)
            
            # Make predictions
            prediction = model.predict(input_data)[0]

            # Display the result
            st.success(f"Predicted Total CO2 Emission: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")

