# app.py
import h5py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
# Function to load model using h5py
def load_model_h5py(model_path):
    with h5py.File(model_path, 'r') as file:
        model_json = file.attrs['model_config']
        model = model_from_json(model_json.decode('utf-8'))
        model.load_weights(model_path)
    return model

# Load your model using the function
model = load_model_h5py("sales_forecast_model.h5")

def predict_sales(input_sequence):
    input_sequence = np.array(input_sequence).reshape(1, len(input_sequence), len(input_sequence[0]))
    predictions = model.predict(input_sequence)
    return predictions.flatten().tolist()

def main():
    st.title("Sales Forecasting App")

    # Get input sequence from the user
    input_sequence = st.text_area("Enter your input sequence (comma-separated values):")

    # Convert the input string to a list of lists
    input_sequence = [list(map(float, row.split(','))) for row in input_sequence.split('\n')]

    if st.button("Predict"):
        # Make predictions
        predictions = predict_sales(input_sequence)

        # Display the predictions
        st.write("Predicted Sales:")
        st.write(predictions)

if __name__ == "__main__":
    main()
