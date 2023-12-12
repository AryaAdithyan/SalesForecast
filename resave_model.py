# resave_model.py

import os
import tensorflow as tf

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Load the original model
original_model = tf.keras.models.load_model("sales_forecast_model_saved_model")

# Specify the directory path
export_dir = "resaved_sales_forecast_model"

# Ensure the directory exists
os.makedirs(export_dir, exist_ok=True)

# Print the contents of the current directory
print("Contents of Current Directory:", os.listdir())

# Save the model in the SavedModel format
tf.saved_model.save(original_model, export_dir)
