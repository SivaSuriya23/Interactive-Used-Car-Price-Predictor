import pandas as pd
import joblib
from xgboost import XGBRegressor
import streamlit as st

# Loading the saved model and preprocessing information
xgb_model = joblib.load('xgboost_model.pkl')
encoded_columns = joblib.load('encoded_columns.pkl')

# Loading the dataset to get unique values for dropdown options
df_cars_initial = pd.read_excel("all_cities_cars_with_url.xlsx")

df_cars_with_url_initial = df_cars_initial[['oem', 'url_model']]
df_cars_url = df_cars_with_url_initial.drop_duplicates()

df_cars = df_cars_initial.drop('url_model', axis=1)

# Define categorical columns and extract unique values
categorical_columns = ['ft', 'transmission', 'oem', 'model', 'Insurance Validity', 'Color',
                       'Location', 'RTO_grouped']

unique_values = {col: df_cars[col].unique().tolist() for col in categorical_columns}

# Create a mapping of brands to their models
brand_model_mapping = df_cars.groupby('oem')['model'].unique().to_dict()


def preprocess_input(data, encoded_columns, categorical_columns):
    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Reindex the columns to ensure the same columns as training data
    data_encoded = data_encoded.reindex(columns=encoded_columns, fill_value=0)

    return data_encoded


def predict_price(input_data):
    # Preprocess the input data
    processed_input = preprocess_input(input_data, encoded_columns, categorical_columns)

    # Make prediction
    prediction = xgb_model.predict(processed_input)

    return prediction[0]


def format_inr(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)


def main():
    # Sidebar navigation
    st.sidebar.header('Navigation')
    page = st.sidebar.radio("Go to", ["Home", "Predict Car Price"])

    # Display content based on the selected page
    if page == "Home":
        st.title('Car Dheko: Used Car Price Predictor')
        # Display main image only on the Home page
        st.image('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/used_cars_image.jpg', caption='Used Cars Overview')
    
    elif page == "Predict Car Price":
        st.title("Estimated Car Price")
        st.sidebar.header('Enter car details:')

        # Display sidebar image
        st.sidebar.image('C:/Users/Siva/Capstone Projects/Car Dheko/Datasets/car_image.jpeg')

        # Input for Brand and dynamically populate Models
        selected_brand = st.sidebar.selectbox('Brand', unique_values['oem'])
        if selected_brand in brand_model_mapping:
            models = brand_model_mapping[selected_brand].tolist()

        selected_model = st.sidebar.selectbox('Model', models)

        # Input for other categorical and numerical features
        color = st.sidebar.selectbox('Color', unique_values['Color'])
        transmission = st.sidebar.selectbox('Transmission', unique_values['transmission'])
        ft = st.sidebar.selectbox('Fuel Type', unique_values['ft'])
        insurance_validity = st.sidebar.selectbox('Insurance Validity', unique_values['Insurance Validity'])
        location = st.sidebar.selectbox('Location', unique_values['Location'])
        rto_grouped = st.sidebar.selectbox('RTO Grouped', unique_values['RTO_grouped'])

        modelYear = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024)
        turbo_charger = st.sidebar.checkbox('Turbo Charger')
        km = st.sidebar.number_input('Kms driven', min_value=0)
        engineCC = st.sidebar.number_input('Engine CC', min_value=0)

        # Prepare the input data
        input_data = pd.DataFrame({
            'oem': [selected_brand],
            'model': [selected_model],
            'Color': [color],
            'ft': [ft],
            'transmission': [transmission],
            'Insurance Validity': [insurance_validity],
            'Location': [location],
            'RTO_grouped': [rto_grouped],
            'modelYear': [modelYear],
            'km': [km],
            'Displacement': [engineCC],
            'Turbo Charger': [turbo_charger]
        })

        if st.sidebar.button('Predict'):
            prediction = predict_price(input_data)
            formatted_price = format_inr(prediction)

            st.markdown(f'''
                <div style="padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
                    <h2 style="color: #4CAF50;"> {selected_model}</h2>
                    <h1 style="color: #4CAF50;">₹ {formatted_price}</h1>
                </div>
            ''', unsafe_allow_html=True)

            # Display image of the selected brand
            brand_image_row = df_cars_with_url_initial[df_cars_with_url_initial['oem'] == selected_brand]
            if not brand_image_row.empty:
                brand_image_url = brand_image_row['url_model'].values[0]
                st.image(brand_image_url, caption=f'{selected_brand} Image')

            # Display min and max price range for the selected model
            matching_cars = df_cars[df_cars['model'] == selected_model]
            if not matching_cars.empty:
                min_price = matching_cars['price'].min()
                max_price = matching_cars['price'].max()

                st.markdown(f'''
                    <div style="padding: 20px; border: 2px solid #2196F3; border-radius: 10px; background-color: #f1f1f1;">
                        <h2 style="color: #2196F3;">Price Range for {selected_model} in Inventory</h2>
                        <div style="display: flex; justify-content: space-between;">
                            <div style="padding: 10px; border: 1px solid #2196F3; border-radius: 5px; background-color: #e3f2fd;">
                                <strong>Min Price:</strong><br>
                                ₹ {format_inr(min_price)}
                            </div>
                            <div style="padding: 10px; border: 1px solid #2196F3; border-radius: 5px; background-color: #e3f2fd;">
                                <strong>Max Price:</strong><br>
                                ₹ {format_inr(max_price)}
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.write(f'No available cars found for the model: {selected_model}')


if __name__ == '__main__':
    main()
