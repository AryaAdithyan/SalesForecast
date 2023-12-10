import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load your sales data
# Assuming you have a CSV file named 'salesdaily.csv' with columns specified
# Adjust the file path accordingly

file_path = 'salesdaily.csv'
daily_data = pd.read_csv(file_path)

# Extract relevant features
features = ['datum', 'M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06', 'Year', 'Month', 'Hour', 'Weekday Name']
daily_data = daily_data[features]

# Pivot the table to have drug categories as columns
daily_data_pivot = daily_data.pivot(index='datum', columns='Weekday Name', values=['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06'])

# Handle missing values if any
daily_data_pivot = daily_data_pivot.fillna(0)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data_pivot)

# Create sequences for training
sequence_length = 10  # You can adjust this based on your preference
X_train = []
y_train = []

for i in range(len(scaled_data) - sequence_length):
    X_train.append(scaled_data[i:i+sequence_length])
    y_train.append(scaled_data[i+sequence_length])

X_train, y_train = np.array(X_train), np.array(y_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=len(features) - 1))  # One neuron for each feature except 'datum'
model.compile(optimizer='adam', loss='mean_squared_error')

# Save the best model during training
checkpoint = ModelCheckpoint("sales_forecast_model.h5", save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[checkpoint])
# Save the model in a different format (e.g., TensorFlow SavedModel format)
model.save("sales_forecast_model_saved_model", save_format="tf")
