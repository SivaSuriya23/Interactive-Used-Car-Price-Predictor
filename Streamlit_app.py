import pandas as pd
import joblib
import numpy as np
import streamlit as st

# Load pre-trained model and necessary transformers
model = joblib.load('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/model.pkl')
scaler = joblib.load('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/scaler.pkl')
poly = joblib.load('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/poly.pkl')
expected_columns = joblib.load('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/expected_columns.pkl')

def preprocess_data(df):
    """
    Preprocess the input DataFrame to match the model's training data format.
    This includes encoding categorical variables and scaling numerical features.
    """
    categorical_features = ['Original Equipment Manufacturer', 'Owner', 'Model', 'Body Type', 'City', 'Seats', 'Color', 'Fuel Type']
    numerical_features = ['Model Year', 'Mileage', 'Registration Year']

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Debugging: Check feature count after encoding
    print(f"Features after encoding: {df_encoded.shape[1]}")

    # Add missing columns with default value 0
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match the training data
    df_encoded = df_encoded[expected_columns]

    # Debugging: Check feature count after reordering
    print(f"Features after reordering: {df_encoded.shape[1]}")

    # Convert all columns expected by the scaler to numeric
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

    # Handle NaN values if any
    df_encoded.fillna(0, inplace=True)

    # Debugging: Verify the number of features before scaling
    print(f"Features before scaling: {df_encoded.shape[1]}")
    if df_encoded.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Feature mismatch: Expected {scaler.n_features_in_} features, but got {df_encoded.shape[1]} features.")

    # Scale numerical features
    X_scaled = scaler.transform(df_encoded)

    # Apply polynomial features if needed
    X_poly = poly.transform(X_scaled)

    return X_poly

def load_data(file):
    """Load data from a CSV file."""
    df = pd.read_csv(file)
    return df

def main():
    st.title("Car Price Prediction App")

    st.sidebar.header("Upload Your CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data loaded successfully:")
        st.write(df.head())

        if st.button("Predict Prices"):
            try:
                # Preprocess the data
                X_transformed = preprocess_data(df)
                
                # Predict prices
                predictions = model.predict(X_transformed)
                
                # Add predictions to the DataFrame
                df['Predicted_Price'] = predictions
                
                st.write("Predicted Prices:")
                st.write(df[['Predicted_Price']])
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Upload a CSV file and click 'Predict Prices' to get predictions.")

if __name__ == "__main__":
    main()
