import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load Models and Preprocessor
models = {
    "SGD": joblib.load('sgd_best_model.pkl'),
    "LightGBM": joblib.load('lgb_best_model.pkl'),
    "Lasso": joblib.load('best_lasso_model.pkl'),
    "Elastic Net": joblib.load('best_elasticnet_model.pkl'),
}
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler
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
Please enter the values below to get the prediction.
""")

# Model Selection (get model names from the dictionary keys)
model_choice = st.selectbox("Select Model:", ["All Models"] + list(models.keys()))

# Input form for Area and Year
# Dropdown for Area Selection
area = st.selectbox("Select the Area:", area_list)
year = st.number_input("Enter the Year (e.g., 2023):", min_value=1900, max_value=2100, step=1)

if st.button("Predict"):
    if area and year:
        try:
            # Prepare Input Data:
            new_data = pd.DataFrame({'area': [area], 'year': [year]})

            # Encode 'area':
            new_data_encoded = pd.get_dummies(new_data, columns=['area'], drop_first=True)

            # Get feature names from ColumnTransformer
            feature_names = preprocessor.get_feature_names_out()

            # Handle missing columns (if any) to match training data:
            missing_cols = set(feature_names) - set(new_data_encoded.columns)
            missing_data = pd.DataFrame(0, index=new_data_encoded.index, columns=list(missing_cols))
            new_data_encoded = pd.concat([new_data_encoded, missing_data], axis=1)
            new_data_encoded = new_data_encoded[feature_names]  # Reorder columns to match training data

            # Scale features:
            new_data_scaled = scaler.transform(new_data_encoded)

            if model_choice == "All Models":
                # Display predictions for all models
                st.subheader("Predictions from All Models:")
                results = {}
                for model_name, model in models.items():
                    prediction = model.predict(new_data_scaled)[0]
                    results[model_name] = prediction
                    st.write(f"{model_name}: {prediction:.2f}")

                # Optionally, display as a DataFrame
                st.write("Prediction Summary:")
                results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Prediction'])
                results_df['Prediction'] = results_df['Prediction'].map('{:,.2f}'.format)  # Format with commas and 2 decimal places
                st.dataframe(results_df['Prediction'].style.set_properties(**{'text-align': 'right'})) # Align to the right
                st.dataframe(results_df)
            else:
                # Get the selected model and make a prediction
                model = models[model_choice]
                prediction = model.predict(new_data_scaled)[0]
                st.success(f"Predicted Total CO2 Emission for {area} in {year} using {model_choice}: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please provide valid inputs for both Area and Year.")
