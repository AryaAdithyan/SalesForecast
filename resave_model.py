# resave_model.py

import tensorflow as tf

# Load the original model
original_model = tf.keras.models.load_model("sales_forecast_model_saved_model")

# Save the model in a different format
tf.saved_model.save(original_model, "resaved_sales_forecast_model")
