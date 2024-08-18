import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the preprocessor and the best model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_model.pkl')

# Streamlit interface
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# Display the title and a brief description
st.title('Car Dheko - Car Price Prediction')
st.write("""
    Welcome to the Car Dheko Car Price Prediction app! 
    Upload a CSV file containing car details, and use the filters to refine your search. 
    Click 'Predict' to estimate the car prices.
""")

# Display car image
image = Image.open("C:\\Users\\Siva\\Capstone Projects\\Car Dheko\\Datasets\\car_image.jpeg") # Make sure you have this image file in your working directory
st.image(image, caption='Predict Car Prices with Car Dheko', use_column_width=True)

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # Define required columns and additional columns for the model
    required_columns = ['Fuel type', 'Original Equipment Manufacturer', 'Ownership']
    additional_columns = ['Model', 'Mileage', 'Transmission', 'Color', 'Seats', 'Model Year', 'Body type', 'Tyre Type']
    
    # Check if required columns are present
    missing_required_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_required_columns:
        st.error(f"Missing required columns in the CSV file: {', '.join(missing_required_columns)}")
    else:
        st.success("CSV file uploaded successfully!")

        # Initialize session state for filters
        if 'filters' not in st.session_state:
            st.session_state.filters = {col: None for col in required_columns}
        
        # Display filter options and update session state
        filter_cols = [col for col in required_columns if col in input_data.columns]
        for column in filter_cols:
            unique_values = input_data[column].unique()
            selected_value = st.selectbox(
                f"Filter by {column}",
                options=[None] + list(unique_values),
                index=(list(unique_values).index(st.session_state.filters[column]) + 1
                       if st.session_state.filters[column] in unique_values else 0),
                key=column
            )
            st.session_state.filters[column] = selected_value
        
        # Apply filters to the data
        filtered_data = input_data.copy()
        for column, value in st.session_state.filters.items():
            if value is not None and column in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[column] == value]
        
        # Ensure all required columns are present
        for col in required_columns + additional_columns:
            if col not in filtered_data.columns:
                filtered_data[col] = None  # Add missing columns with None or default values

        # Display filtered results before prediction
        if not filtered_data.empty:
            st.write(f"Filtered Data ({len(filtered_data)} records):")
            st.write(filtered_data)
            
            # Predict button
            if st.button('Predict'):
                try:
                    # Extract features for prediction
                    features = filtered_data[required_columns + additional_columns]
                    
                    # Preprocess the features
                    features_preprocessed = preprocessor.transform(features)
                    
                    # Make predictions
                    predictions = model.predict(features_preprocessed)
                    
                    # Convert predictions to lakhs
                    predictions_in_lakhs = (predictions * 10000000).round()
                    
                    # Add predictions to the filtered data
                    filtered_data['Price in Lakhs'] = predictions_in_lakhs
                    
                    # Filter columns to show only 'Model' and 'Price in Lakhs'
                    result_data = filtered_data[['Model', 'Price in Lakhs']]
                    
                    st.write("Filtered Data with Predicted Prices:")
                    st.write(result_data)
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("No matching records found with the current filters.")
else:
    st.info("Please upload a CSV file to start the prediction process.")
