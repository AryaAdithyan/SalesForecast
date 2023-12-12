# resave_model.py

import os
import shutil
import tensorflow as tf

# Load the original model
original_model = tf.keras.models.load_model("sales_forecast_model_saved_model")

# Specify the directory path
export_dir = "resaved_sales_forecast_model"

# Ensure the directory exists
os.makedirs(export_dir, exist_ok=True)

# Save the model in the SavedModel format
tf.saved_model.save(original_model, export_dir)

# Optionally, copy the variables folder to the new directory
shutil.copytree("sales_forecast_model_saved_model/variables", f"{export_dir}/variables")
