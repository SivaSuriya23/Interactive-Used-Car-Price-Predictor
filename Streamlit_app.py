import pandas as pd
import joblib
import streamlit as st

# Caching the model and data to prevent reloading multiple times
@st.cache_resource
def load_model_and_data():
    # Load the model and preprocessing info
    ml_model = joblib.load('best_random_forest_model.pkl')
    encoded_columns = joblib.load('encoded_columns.pkl')  # Columns used during model training
    df_cars_initial = pd.read_excel("all_cities_cars_with_url.xlsx")
    df_cars_with_url_initial = df_cars_initial[['Original Equipment Manufacturer', 'url_model']].drop_duplicates()
    df_cars = df_cars_initial.drop('url_model', axis=1)

    # Define categorical columns and extract unique values
    categorical_columns = ['Fuel Type', 'transmission', 'Original Equipment Manufacturer', 'model', 'Insurance Validity', 'Color']
    unique_values = {col: df_cars[col].unique().tolist() for col in categorical_columns}
    brand_model_mapping = df_cars.groupby('Original Equipment Manufacturer')['model'].unique().to_dict()

    return ml_model, encoded_columns, df_cars, df_cars_with_url_initial, unique_values, brand_model_mapping

# Preprocess the input data for prediction
def preprocess_input(data, encoded_columns, categorical_columns):
    # Apply one-hot encoding and reindex to match the model's expected input
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Ensure the columns match the model's expected features (subset the columns)
    data_encoded = data_encoded.reindex(columns=encoded_columns, fill_value=0)
    
    # Ensure the data has the right shape (number of columns should match the model's expected)
    if data_encoded.shape[1] != len(encoded_columns):
        raise ValueError(f"Feature shape mismatch: expected {len(encoded_columns)} features, but got {data_encoded.shape[1]}")

    return data_encoded

# Prediction function with caching (underscore to prevent caching of the model)
@st.cache_data
def predict_price(_ml_model, processed_input):
    return int(_ml_model.predict(processed_input)[0])

# Format the price to INR with commas
def format_inr(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]] )
    return "".join([r] + d)

def main():
    # Load the model and data once to optimize performance
    ml_model, encoded_columns, df_cars, df_cars_with_url_initial, unique_values, brand_model_mapping = load_model_and_data()

    # Sidebar navigation
    st.sidebar.header('Navigation')
    page = st.sidebar.radio("Go to", ["Home", "Predict Car Price"])

    # Display content based on the selected page
    if page == "Home":
        st.title('Car Dheko: Used Car Price Predictor')
        st.markdown('<h2 style="font-weight: bold; font-size: 24px;">This Predictor can reliably estimate car prices based on many features such as brand, model, mileage, year, condition, etc. This will help sellers set competitive prices for their listings and assist buyers in determining fair prices for cars they are interested in buying.</h2>', unsafe_allow_html=True)

    elif page == "Predict Car Price":
        st.title("Estimated Car Price")
        st.sidebar.header('Enter car details:')

        # Display sidebar image
        st.sidebar.image('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/car_image.jpeg')

        # Input for Brand and dynamically populate Models
        selected_brand = st.sidebar.selectbox('Brand', unique_values['Original Equipment Manufacturer'])
        models = brand_model_mapping.get(selected_brand, [])
        selected_model = st.sidebar.selectbox('Model', models)

        # Input for other categorical and numerical features
        color = st.sidebar.selectbox('Color', unique_values['Color'])
        transmission = st.sidebar.selectbox('Transmission', unique_values['transmission'])
        ft = st.sidebar.selectbox('Fuel Type', unique_values['Fuel Type'])
        insurance_validity = st.sidebar.selectbox('Insurance Validity', unique_values['Insurance Validity'])
        modelYear = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024)

        # Prepare the input data
        input_data = pd.DataFrame({
            'Original Equipment Manufacturer': [selected_brand],
            'model': [selected_model],
            'Color': [color],
            'Fuel Type': [ft],
            'transmission': [transmission],
            'Insurance Validity': [insurance_validity],
            'modelYear': [modelYear],
        })

        if st.sidebar.button('Predict'):
            try:
                # Preprocess the input data
                processed_input = preprocess_input(input_data, encoded_columns, ['Fuel Type', 'transmission', 'Original Equipment Manufacturer', 'model', 'Insurance Validity', 'Color'])
                
                # Debugging: Check the shape and contents of the processed input
                #st.write("Processed input data:", processed_input)
                #st.write("Processed input shape:", processed_input.shape)

                # Show a spinner while waiting for prediction
                with st.spinner('Predicting...'):
                    prediction = predict_price(ml_model, processed_input)
                    formatted_price = format_inr(prediction)
                    
                    # Display result
                    st.markdown(f'''
                        <div style="padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #FFF8DC;">
                            <h2 style="color: #006400;"> {selected_model}</h2>
                            <h1 style="color: #8B0000;">₹ {formatted_price}</h1>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Display image of the selected brand
                    brand_image_row = df_cars_with_url_initial[df_cars_with_url_initial['Original Equipment Manufacturer'] == selected_brand]
                    if not brand_image_row.empty:
                        brand_image_url = brand_image_row['url_model'].values[0]

                        if isinstance(brand_image_url, str) and brand_image_url.strip():
                            st.image(brand_image_url, caption=f'{selected_brand} Image')
                        else:
                            st.warning('Image URL not available for the selected brand.')
                    else:
                        st.warning(f'No image found for the selected brand: {selected_brand}')

                    # Display min and max price range for the selected model
                    matching_cars = df_cars[df_cars['model'] == selected_model]
                    if not matching_cars.empty:
                        min_price = matching_cars['price'].min()
                        max_price = matching_cars['price'].max()

                        st.markdown(f'''
                            <div style="padding: 20px; border: 2px solid #006400; border-radius: 10px; background-color: #FFF8DC;">
                                <h2 style="color: #1E90FF;">Price Range for {selected_model} in Inventory</h2>
                                <div style="display: flex; justify-content: space-between;">
                                    <div style="padding: 10px; border: 1px solid #006400; border-radius: 5px; background-color: #F8F8FF;">
                                        <strong>Min Price:</strong><br>
                                        ₹ {format_inr(min_price)}
                                    </div>
                                    <div style="padding: 10px; border: 1px solid #006400; border-radius: 5px; background-color: #F8F8FF;">
                                        <strong>Max Price:</strong><br>
                                        ₹ {format_inr(max_price)}
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.write(f'No available cars found for the model: {selected_model}')
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
