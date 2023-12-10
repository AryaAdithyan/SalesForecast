import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "sales_forecast_model_saved_model"
model = load_model(model_path)

# Save the model again in TensorFlow SavedModel format
saved_model_path = "sales_forecast_model_saved_model_resaved"
model.save(saved_model_path, save_format="tf")
