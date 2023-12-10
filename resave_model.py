from tensorflow.keras.models import load_model, save_model

# Load your model
model = load_model("sales_forecast_model")

# Save the model again
save_model(model, "sales_forecast_model")
