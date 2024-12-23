import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
# Load all models into a dictionary
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    # Add more models here in the future...
}
preprocessor = joblib.load('preprocessor.pkl')  # Load the preprocessor
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

# Predict button
if st.button("Predict"):
    if area and year:
        try:
            # 1. Create a DataFrame for the new data
            new_data = pd.DataFrame({'area': [area], 'year': [year]})
            st.write("Datatype of 'year':", new_data['year'].dtypes) 
            # 2. Preprocess the new data (similar to your code)
            new_data_encoded = pd.get_dummies(new_data, columns=['area'], drop_first=True)

            feature_names = preprocessor.get_feature_names_out()

            missing_cols = set(feature_names) - set(new_data_encoded.columns)
            missing_data = pd.DataFrame(0, index=new_data_encoded.index, columns=list(missing_cols))
            new_data_encoded = pd.concat([new_data_encoded, missing_data], axis=1)

            new_data_encoded = new_data_encoded[feature_names]
            # Convert 'year' to ordinal before scaling
            new_data_encoded['year'] = pd.to_datetime(new_data_encoded['year'], format='%Y', errors='coerce').apply(lambda date: date.toordinal() if pd.notnull(date) else 0) 
            
            # Scale the data
            new_data_scaled = scaler.transform(new_data_encoded)
            # 3. Get the selected model from the dictionary
            model = models[model_choice]

            # 4. Make predictions
            prediction = model.predict(new_data_scaled)[0]

            # 5. Display the result
            st.success(f"Predicted Total CO2 Emission for {area} in {year}: {prediction:.2f}")
        #except Exception as e:
            #st.error(f"Error during prediction: {e}")
    #else:
        #st.warning("Please provide valid inputs for both Area and Year.")
