import streamlit as st
import joblib
import pandas as pd

# Load the preprocessor and the best model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_model.pkl')

# Streamlit interface
st.title('Car Dheko Car Price Prediction')

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
        for column in st.session_state.filters.keys():
            if column in input_data.columns:
                unique_values = input_data[column].unique()
                selected_value = st.selectbox(
                    column,
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
