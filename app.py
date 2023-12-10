# app.py

import streamlit as st
import tensorflow as tf
import numpy as np

# Load the resaved model using tf.saved_model.load
model_path = "resaved_sales_forecast_model"
model = tf.saved_model.load(model_path)

# Get the concrete function
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

def predict_sales(input_sequence):
    # Convert the input string to a list of lists
    input_sequence = [list(map(float, row.split(','))) for row in input_sequence.split('\n')]

    # Preprocess the input
    input_array = np.array(input_sequence).reshape(1, len(input_sequence), len(input_sequence[0]))

    # Make predictions
    predictions = concrete_func(tf.constant(input_array))
    return predictions['dense'].numpy().flatten().tolist()

def main():
    st.title("Sales Forecasting App")

    # Get input sequence from the user
    input_sequence = st.text_area("Enter your input sequence (comma-separated values):")

    if st.button("Predict"):
        # Make predictions
        predictions = predict_sales(input_sequence)

        # Display the predictions
        st.write("Predicted Sales:")
        st.write(predictions)

if __name__ == "__main__":
    main()
