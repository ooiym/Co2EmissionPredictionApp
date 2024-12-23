
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
            # Prepare Input Data:
            new_data = pd.DataFrame({'area': [area], 'year': [year]})  # Replace with your desired area and year

            # 1. Encode 'area':
            new_data_encoded = pd.get_dummies(new_data, columns=['area'], drop_first=True)

            # Get feature names from ColumnTransformer
            feature_names = preprocessor.get_feature_names_out()

            # 2. Handle missing columns (if any) to match training data:
            missing_cols = set(feature_names) - set(new_data_encoded.columns)
            # Create a DataFrame for missing columns with 0 values
            missing_data = pd.DataFrame(0, index=new_data_encoded.index, columns=list(missing_cols))

            # Concatenate the missing columns with the existing DataFrame
            new_data_encoded = pd.concat([new_data_encoded, missing_data], axis=1)

            new_data_encoded = new_data_encoded[feature_names]  # Reorder columns to match training data

            # 3. Scale features:
            new_data_scaled = scaler.transform(new_data_encoded)  # Use the same scaler used during training

            # 4. Make Predictions:
            #predicted_emission = lgb_grid.best_estimator_.predict(new_data_scaled)
            #print(f"Predicted Total Emission for {new_data['area'][0]}: {predicted_emission[0]}")

            # 3. Get selected model
            model = models[model_choice]

            # 4. Make predictions
            prediction = model.predict(new_data_scaled)[0]  # Use transformed data

            # 5. Display result
            st.success(f"Predicted Total CO2 Emission for {area} in {year}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")
