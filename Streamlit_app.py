import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Set page config at the top of the script
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# Load the trained model and preprocessing steps
model = joblib.load('C:\\Users\\Siva\\Capstone Projects\\Car Dheko\\Datasets\\car_price_prediction_model.pkl')
label_encoders = joblib.load('C:\\Users\\Siva\\Capstone Projects\\Car Dheko\\Datasets\\label_encoders.pkl')
scalers = joblib.load('C:\\Users\\Siva\\Capstone Projects\\Car Dheko\\Datasets\\scalers.pkl')

# Load dataset for filtering and identifying similar data
data = pd.read_csv('C:\\Users\\Siva\\Capstone Projects\\Car Dheko\\Datasets\\all_datasets_cleaned_Raw.csv')

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0])

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df

# Create PDF function
def create_pdf(prediction, car_age, mileage_normalized):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add title and result
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, "Car Price Prediction Result")
    
    c.setFont("Helvetica", 16)
    c.drawString(100, height - 140, f"Predicted Price: {prediction:,.2f} Lakhs")
    c.drawString(100, height - 180, f"Car Age: {car_age} years")
    c.drawString(100, height - 220, f"Efficiency Score: {mileage_normalized:,.2f} km/year")

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# Main page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Car Price Prediction", "About"])

    if page == "Home":
        st.title("Welcome to the Car Dheko - Car Price Prediction App")
        st.write("This app allows you to predict car prices based on various features.")

        image_paths = [
            'car_image.jpeg'
        ]

        for image_path in image_paths:
            full_path = os.path.join(image_path)
            if os.path.isfile(full_path):
                image = Image.open(full_path)
                st.image(image, caption=image_path.split('/')[-1], use_column_width=True)
            else:
                st.warning(f"Image not found: {full_path}. Please ensure the image file is in the specified location.")

    elif page == "Car Price Prediction":
        st.title('Predicted Car Price')

        # Sidebar inputs under the "About" page
        st.sidebar.header('Input Car Features')

        # Get user inputs with visual representation
        selected_oem = visual_selectbox('1. Original Equipment Manufacturer (OEM)', data['oem'].unique())
        filtered_data = filter_data(oem=selected_oem)
        selected_model = visual_selectbox('2. Car Model', filtered_data['model'].unique())

        filtered_data = filter_data(oem=selected_oem, model=selected_model)
        body_type = visual_selectbox('3. Body Type', filtered_data['bt'].unique())

        filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type)
        fuel_type = visual_selectbox('4. Fuel Type', filtered_data['ft'].unique())

        filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
        transmission = visual_selectbox('5. Transmission Type', filtered_data['transmission'].unique())

        filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
        seat_count = visual_selectbox('6. Seats', filtered_data['Seats'].unique())

        filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type, seats=seat_count)
        selected_variant = visual_selectbox('7. Variant Name', filtered_data['variantName'].unique())

        modelYear = st.sidebar.number_input('8. Year of Manufacture', min_value=1980, max_value=2024, value=2015)
        ownerNo = st.sidebar.number_input('9. Number of Previous Owners', min_value=0, max_value=10, value=1)
        km = st.sidebar.number_input('10. Kilometers Driven', min_value=0, max_value=500000, value=10000)

        # Adjust mileage slider
        min_mileage = np.floor(filtered_data['mileage'].min())
        max_mileage = np.ceil(filtered_data['mileage'].max())
        mileage = st.sidebar.slider('11. Mileage (kmpl)', min_value=float(min_mileage), max_value=float(max_mileage), value=float(min_mileage), step=0.5)

        city = visual_selectbox('12. City', data['City'].unique())

        # Button to trigger prediction
        if st.sidebar.button('Predict'):
            user_input_data = {
                'ft': [fuel_type],
                'bt': [body_type],
                'km': [km],
                'transmission': [transmission],
                'ownerNo': [ownerNo],
                'oem': [selected_oem],
                'model': [selected_model],
                'modelYear': [modelYear],
                'variantName': [selected_variant],
                'City': [city],
                'mileage': [mileage],
                'Seats': [seat_count],
                'car_age': [2024 - modelYear],
                'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
                'mileage_normalized': [mileage / (2024 - modelYear)]
            }

            user_df = pd.DataFrame(user_input_data)
            user_df = user_df[features]
            user_df = preprocess_input(user_df)

            if user_df.notnull().all().all():
                try:
                    # Make prediction
                    prediction = model.predict(user_df)
                    
                    st.markdown(f"""
                        <div class="result-container">
                            <h2 class="prediction-title">Predicted Car Price</h2>
                            <p class="prediction-value">₹{prediction[0]:,.2f}</p>
                            <p class="info">Car Age: {user_df['car_age'][0]} years</p>
                            <p class="info">Efficiency Score: {user_df['mileage_normalized'][0]:,.2f} km/year</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Generate and display PDF
                    pdf = create_pdf(prediction[0], user_df['car_age'][0], user_df['mileage_normalized'][0])
                    st.download_button(
                        label="Download Prediction as PDF",
                        data=pdf,
                        file_name="car_price_prediction_result.pdf",
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"Error in prediction: {e}")
            else:
                missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
                st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")
                
    elif page == "About":
        st.title("About")

        image_path = 'About_image.png'
        full_path = os.path.join(image_path)
        if os.path.isfile(full_path):
            image = Image.open(full_path)
            st.image(image, caption='Car Dheko - About Us', use_column_width=True)
        else:
            st.warning(f"About image not found: {full_path}. Please ensure the image file is in the specified location.")

                
# Set background colors
input_background_color = "#FFE4C4"
result_background_color = "#7FFFD4"
sidebar_background_color = "#FFE4C4"
filter_background_color = "#FFE4C4"

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        background-color: {result_background_color};
    }}
    .stButton>button {{
        background-color: #7FFFD4;  /* Aquamarine */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }}
    .stButton>button:hover {{
        background-color: #66CDAA;  /* Darker Aquamarine */
    }}
    .result-container {{
        text-align: center;
        background-color: {result_background_color};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .prediction-title {{
        font-size: 28px;
        color: #800000;  /* Maroon */
    }}
    .prediction-value {{
        font-size: 36px;
        font-weight: bold;
        color: #191970;  /* MidnightBlue */
    }}
    .info {{
        font-size: 18px;
        color: #000000;  /* Black */
    }}
    .sidebar .sidebar-content {{
        background-color: {sidebar_background_color};
        color: #FFD700;  /* Gold */
    }}
    .sidebar .sidebar-content .stSelectbox, .sidebar .sidebar-content .stNumberInput, .sidebar .sidebar-content .stSlider {{
        background-color: {filter_background_color};
        color: #9370DB;  /* MediumPurple */
    }}
    .stSelectbox>div>div>div, .stNumberInput>div>div>div, .stSlider>div>div>div {{
        border: 1px solid #F0FFF0;
        border-radius: 5px;
    }}
    .stSelectbox>div>div>div>button, .stNumberInput>div>div>div>button {{
        color: #00FF00;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Get user inputs with visual representation
def visual_selectbox(label, options, index=0):
    selected_option = st.sidebar.selectbox(label, options, index=index)
    return selected_option


if __name__ == '__main__':
    main()
