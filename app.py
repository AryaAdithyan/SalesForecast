import streamlit as st
import tensorflow as tf
import numpy as np

# Load the resaved model
model_path = "resaved_sales_forecast_model"

# Load the model using TensorFlow SavedModel format
model = tf.saved_model.load(model_path)

def predict_sales(input_sequence):
    input_sequence = np.array(input_sequence).reshape(1, len(input_sequence), len(input_sequence[0]))
    # Use the 'serving_default' signature to make predictions
    predictions = model.signatures['serving_default'](tf.constant(input_sequence, dtype=tf.float32))
    return predictions['output'].numpy().flatten().tolist()

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
