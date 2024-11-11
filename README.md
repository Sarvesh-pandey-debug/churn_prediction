# churn_prediction
# Entire code for model deployment using streamlit 
# Import necessary libraries
import pandas as pd
import pickle
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Streamlit app title
st.title("Customer Churn Prediction Model Training")

# Load and preprocess data
st.write("Loading and preprocessing data...")

try:
    # Replace this path with your file path if necessary
    data = pd.read_csv('C:/Users/sar/Downloads/Churn_Modelling.csv')

    # Dropping irrelevant columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encoding categorical variables
    data['Geography'] = data['Geography'].astype('category').cat.codes
    data['Gender'] = data['Gender'].astype('category').cat.codes

    # Splitting data into features and target variable
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    st.write("Data loaded and preprocessed successfully.")

    # Train the model
    st.write("Training the Gradient Boosting Classifier model...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    st.write("Model training complete.")

    # Save the model and scaler
    with open("churn_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    st.success("Model and scaler have been successfully saved as `churn_model.pkl` and `scaler.pkl`.")

except Exception as e:
    st.error(f"An error occurred: {e}")
