import pandas as pd
import streamlit as st
import os
import joblib
from PIL import Image
import re

# Streamlit interface setup
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Car Price", "About"])

# Home page
if page == "Home":
    st.title("Welcome to the Car Dheko - Car Price Prediction App")
    st.write("""
        This app allows you to predict car prices based on various features. 
        You can filter the dataset and make predictions.
    """)

    
    # Display car image
    image_path = 'C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/used_cars_image.jpg'
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Predict Car Prices with Car Dheko', use_column_width=True)
    else:
        st.warning("Car image not found. Please ensure the image file is in the specified location.")
    
    # Display car image
    image_path = 'C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/car_image.jpeg'
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Predict Car Prices with Car Dheko', use_column_width=True)
    else:
        st.warning("Car image not found. Please ensure the image file is in the specified location.")



# Prediction page
elif page == "Predict Car Price":
    st.title('Car Price Prediction')

    # Load the preprocessor and model
    preprocessor = joblib.load('preprocessor_Reg.pkl')
    is_regressor = True
    model = joblib.load('model_pipeline_reg.pkl') if is_regressor else joblib.load('model_pipeline_cls.pkl')

    # Define the path to the CSV file
    csv_path = r"C:\Users\Siva\Capstone Projects\Car Dheko\Datasets\Final_data1.csv"

    # Check if the CSV file exists before loading
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error("CSV file not found. Please ensure the file path is correct.")
        df = pd.DataFrame()

    # Define required columns and additional columns
    required_columns = ['model', 'Fuel Type', 'Ownership']
    additional_columns = ['price', 'modelYear', 'Year of Manufacture', 'Mileage', 'Engine', 'Seats', 'Displacement', 'Length', 'Width', 'Height', 'Wheel Base', 'Kilometers driven']

    # Initialize session state for filters
    if 'filters' not in st.session_state:
        st.session_state.filters = {col: None for col in required_columns}

    # Display filter options and update session state
    filter_cols = [col for col in required_columns if col in df.columns]
    for column in filter_cols:
        unique_values = df[column].unique()
        selected_value = st.selectbox(
            f"Filter by {column}",
            options=[None] + list(unique_values),
            index=(list(unique_values).index(st.session_state.filters[column]) + 1
                   if st.session_state.filters[column] in unique_values else 0),
            key=column
        )
        st.session_state.filters[column] = selected_value

    # Apply filters to the data
    filtered_data = df.copy()
    for column, value in st.session_state.filters.items():
        if value is not None and column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]

    # Ensure all required columns and additional columns are present with default values
    all_columns = required_columns + additional_columns
    for col in all_columns:
        if col not in filtered_data.columns:
            filtered_data[col] = 'Unknown' if col in required_columns else 0

    # Clean 'Seats' column to extract numerical values if it exists
    if 'Seats' in filtered_data.columns:
        def extract_numeric(value):
            match = re.search(r'\d+', str(value))
            return float(match.group()) if match else 0.0

        filtered_data['Seats'] = filtered_data['Seats'].apply(extract_numeric)

    # Handle categorical data
    def preprocess_categorical(df, categorical_columns):
        for col in categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
        return df

    categorical_columns = ['model', 'Fuel Type', 'Ownership', 'transmission', 'Color']
    filtered_data = preprocess_categorical(filtered_data, categorical_columns)

    # Clean other columns to ensure they are numeric where needed
    numeric_columns = ['price', 'Mileage', 'Engine', 'modelYear', 'Kms Driven', 'Displacement', 'Length', 'Width', 'Height', 'Wheel Base', 'km', 'Year of Manufacture']
    for col in numeric_columns:
        if col in filtered_data.columns:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce').fillna(0)

    # Display filtered results before prediction
    if not filtered_data.empty:
        st.write(f"Filtered Data ({len(filtered_data)} records):")
        st.write(filtered_data)

        # Predict button
        if st.button('Predict'):
            try:
                required_features = [col for col in required_columns if col in filtered_data.columns]
                additional_features = [col for col in additional_columns if col in filtered_data.columns]
                all_features = required_features + additional_features

                # Ensure the dataframe matches the preprocessor's expected input
                missing_features = set(required_features + additional_features) - set(filtered_data.columns)
                if missing_features:
                    st.error(f"Missing features: {', '.join(missing_features)}")
                    st.stop()

                features = filtered_data[all_features]

                # Preprocess the features
                features_preprocessed = preprocessor.transform(features)

                # Make predictions
                if is_regressor:
                    predictions = model.predict(features_preprocessed)
                    predictions_in_lakhs = (predictions / 100).round()
                    filtered_data['price'] = predictions_in_lakhs

                    result_data = filtered_data[['price', 'modelYear', 'Year of Manufacture', 'Seats']]
                    st.write("Filtered Data with Predictions:")
                    st.write(result_data)
                else:
                    st.error("Model type is not supported.")
            except ValueError as ve:
                st.error(f"Value error during preprocessing: {ve}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("No matching records found with the current filters.")

# About page
elif page == "About":
    st.title("About")
    st.write("""
        This app was developed to predict car prices using machine learning. 
        The app uses a regression model to estimate the prices based on various features.
    """)

    # Display an image
    image_path = 'C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/About_image.png'
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Car Dheko - About Us', use_column_width=True)
    else:
        st.warning("About image not found. Please ensure the image file is in the specified location.")
