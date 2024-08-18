# Car-Dheko_Project

## 1. Project Overview

**Objective:**
The primary goal of this project was to develop an interactive web application that predicts the prices of used cars based on various features using machine learning. The application, built with Streamlit, is designed to provide accurate and user-friendly price estimates for both customers and sales representatives.

## 2. Project Scope

We utilized historical data on used car prices from CarDekho, which included features such as make, model, year, fuel type, transmission type, and other relevant attributes across different cities. The task was to build a machine learning model to predict car prices and integrate this model into a Streamlit web application.

## 3. Approach

### 3.1 Data Processing

**a) Import and Concatenate:**
- **Datasets:** Imported multiple city-specific datasets and converted them into a structured format.
- **City Column:** Added a 'City' column to each dataset to track the origin of the data.
- **Concatenation:** Merged all datasets into a single, unified dataset.

**b) Handling Missing Values:**
- **Numerical Columns:** Filled missing values using mean, median, or mode imputation.
- **Categorical Columns:** Addressed missing values by using mode imputation or creating a new category.

**c) Standardizing Data Formats:**
- **Format Conversion:** Standardized data formats, such as removing units (e.g., 'kms') from strings and converting them to numerical values.

**d) Encoding Categorical Variables:**
- **Nominal Variables:** Applied one-hot encoding to convert nominal categorical variables into numerical format.
- **Ordinal Variables:** Used label encoding or ordinal encoding for ordinal categorical variables.

**e) Normalizing Numerical Features:**
- **Scaling:** Applied Min-Max Scaling or Standard Scaling to normalize numerical features to a standard range (0 to 1) where applicable.

**f) Removing Outliers:**
- **Outlier Detection:** Identified and managed outliers using the IQR (Interquartile Range) method or Z-score analysis to avoid skewing the model.

### 3.2 Exploratory Data Analysis (EDA)

**a) Descriptive Statistics:**
- **Summary Statistics:** Calculated mean, median, mode, standard deviation, and other statistical measures to understand data distribution.

**b) Data Visualization:**
- **Visual Tools:** Created scatter plots, histograms, box plots, and correlation heatmaps to visualize patterns and relationships in the data.

**c) Feature Selection:**
- **Importance Analysis:** Used correlation analysis, feature importance from models, and domain knowledge to identify key features impacting car prices.

### 3.3 Model Development

**a) Train-Test Split:**
- **Dataset Division:** Split the dataset into training and testing sets with common ratios of 70-30 or 80-20.

**b) Model Selection:**
- **Algorithms:** Selected appropriate machine learning algorithms for price prediction, including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting Machines.

**c) Model Training:**
- **Training Process:** Trained models using the training dataset and applied cross-validation to ensure robust performance.

**d) Hyperparameter Tuning:**
- **Optimization:** Optimized model parameters using techniques such as Grid Search or Random Search to improve performance.

### 3.4 Model Evaluation

**a) Performance Metrics:**
- **Evaluation Metrics:** Assessed model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

**b) Model Comparison:**
- **Comparison:** Compared various models based on evaluation metrics to select the best-performing model.

### 3.5 Optimization

**a) Feature Engineering:**
- **Feature Creation:** Engineered new features or modified existing ones based on insights from EDA and domain knowledge.

**b) Regularization:**
- **Regularization Techniques:** Applied Lasso (L1) and Ridge (L2) regularization to prevent overfitting and enhance model generalization.

### 3.6 Deployment

**a) Streamlit Application:**
- **Deployment:** Developed and deployed the final model using Streamlit, creating an interactive web application for real-time price predictions.

**b) User Interface Design:**
- **UI Design:** Ensured the application was user-friendly with clear instructions and robust error handling.

## 4. Results

1. **Machine Learning Model:** Developed a functional and accurate model for predicting used car prices.
2. **Data Analysis:** Provided comprehensive analysis and visualizations of the dataset.
3. **Documentation:** Produced detailed documentation explaining the methodology, models used, and evaluation results.
4. **Streamlit Application:** Delivered an interactive web application for real-time price predictions based on user input.

## 5. Technical Tags

- Data Preprocessing
- Machine Learning
- Price Prediction
- Regression
- Python
- Pandas
- Scikit-Learn
- Exploratory Data Analysis (EDA)
- Streamlit
- Model Deployment
