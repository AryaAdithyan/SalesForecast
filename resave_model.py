# resave_model.py

import os
import tensorflow as tf

# Load the original model
original_model = tf.keras.models.load_model("sales_forecast_model_saved_model")

# Specify the directory path
export_dir = "resaved_sales_forecast_model"

# Ensure the directory exists
os.makedirs(export_dir, exist_ok=True)

# Save the model in the SavedModel format
original_model.save(export_dir, save_format="tf")
