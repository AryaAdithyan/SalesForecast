import pandas as pd

# Load the daily sales data
daily_data = pd.read_csv("salesdaily.csv")

# Explore the dataset
print(daily_data.head())
# Check for missing values and handle if needed
daily_data.isnull().sum()

# Extract relevant features (date, drug category, sales quantity)
features = ['date', 'category', 'quantity']
daily_data = daily_data[features]

# Pivot the table to have drug categories as columns
daily_data_pivot = daily_data.pivot(index='date', columns='category', values='quantity')

# Handle missing values if any
daily_data_pivot = daily_data_pivot.fillna(0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data_pivot)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
import numpy as np

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : (i + sequence_length)]
        sequences.append(sequence)
    return np.array(sequences)

sequence_length = 10  # You can adjust this based on your preference
X_train = create_sequences(train_data, sequence_length)
X_test = create_sequences(test_data, sequence_length)

# Target variable is the next day's sales for each drug category
y_train = train_data[sequence_length:]
y_test = test_data[sequence_length:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=len(daily_data['category'].unique())))  # One neuron for each drug category
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
model.save("sales_forecast_model.h5")
