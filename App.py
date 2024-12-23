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

# Title of the app
st.title("CO2 Emission Prediction App")

# Instructions for the user
st.write("""
This app predicts the **Total CO2 Emission** based on the Area and Year input.
Please enter the values below to get the prediction.
""")

# Model Selection (get model names from the dictionary keys)
model_choice = st.selectbox("Select Model:", list(models.keys()))

# Input form for Area and Year
area = st.text_input("Enter the Area (e.g., Country or Region):")
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)


# Streamlit App ...

if st.button("Predict"):
    if area and year:
        try:
            # 1. Create DataFrame
            new_data = pd.DataFrame({'area': [area], 'year': [year]})
            # convert year to datetime format
            new_data['Year'] = pd.to_datetime(new_data['Year'], format='%Y')
            #standardize Column Names
            new_data.columns = new_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('\W', '')

            # 2. Preprocess using the ORIGINAL preprocessor
            new_data_transformed = preprocessor.transform(new_data)
            # Convert 'year' to ordinal before scaling
            new_data_encoded['year'] = pd.to_datetime(new_data_encoded['year'], format='%Y', errors='coerce').apply(lambda date: date.toordinal() if pd.notnull(date) else 0)
            new_data_encoded['year'] = pd.to_numeric(new_data_encoded['year'], errors='coerce').fillna(0).astype(int)  # Ensure numerical type

            # 3. Get selected model
            model = models[model_choice]

            # 4. Make predictions
            prediction = model.predict(new_data_transformed)[0]  # Use transformed data

            # 5. Display result
            st.success(f"Predicted Total CO2 Emission for {area} in {year}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")
