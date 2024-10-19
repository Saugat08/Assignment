# prompt: Make streamlit app for rf_model

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
filename = 'rf_model.pkl'
rf_model = pickle.load(open(filename, 'rb'))

# Create the Streamlit app
st.title("Outbreak Likelihood Prediction App")

# Get user input for features
st.header("Enter Feature Values:")

# Replace with the actual feature names and input types from your dataset
region = st.selectbox("Region", ["Region A", "Region B", "Region C"])  # Example
population_density = st.number_input("Population Density")
temperature = st.number_input("Average Temperature")
humidity = st.number_input("Average Humidity")
# ... other feature inputs

# Create a dictionary to store the input values
user_input = {
    "Region": region,
    "Population Density": population_density,
    "Average Temperature": temperature,
    "Average Humidity": humidity,
    # ... other feature inputs
}

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Assuming you need to one-hot encode the region
# Replace 'Region' with the actual column name
categorical_features = ['Region']
encoder = pickle.load(open('encoder.pkl', 'rb'))  # Load the encoder from a file
encoded_features = encoder.transform(input_df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
input_df = input_df.drop(categorical_features, axis=1)
input_df = pd.concat([input_df, encoded_df], axis=1)

# Make prediction
if st.button("Predict Outbreak Likelihood"):
    prediction = rf_model.predict(input_df)[0]
    st.header("Prediction:")
    st.write(f"The predicted outbreak likelihood is: {prediction:.2f}")

    # Optionally, you can classify the prediction into categories
    thresholds = [0.3, 0.7] # Adjust thresholds as needed
    if prediction < thresholds[0]:
        st.write("Category: Low")
    elif prediction < thresholds[1]:
        st.write("Category: Medium")
    else:
        st.write("Category: High")

# Add explanation or other information as needed
st.markdown("""
This app predicts the likelihood of an outbreak based on various environmental and population factors.
""")