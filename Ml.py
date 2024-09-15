import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# Example: Loading a CSV file
data = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/Ml_assignment_121ad0001/121ad0005_datasetML.csv')
# Drop non-numeric columns
data = data.select_dtypes(include=[np.number])
# Fill missing values with mean (or other methods like median, mode, etc.)
data.fillna(data.mean(), inplace=True)

# Normalize data (min-max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data.plot(figsize=(24,12))

# Sequence creation
def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X), np.array(Y)

SEQ_LENGTH = 20
X, Y = create_sequences(scaled_data, SEQ_LENGTH)

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, len(data.columns))))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(len(data.columns)))  # For multivariate dataset

model.compile(optimizer='adam', loss='mse')
# Splitting data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Train the model
model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_test, Y_test))
predictions = model.predict(X_test)
y_pred = model.predict(X_test)

# Inverse the normalization to get real values
real_preds = scaler.inverse_transform(predictions)
real_test = scaler.inverse_transform(Y_test)

# MSE and RMSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(real_test, real_preds)
rmse = np.sqrt(mse)

plt.plot(Y_test, label='True Values', color='blue')
plt.plot(y_pred, label='Predictions', color='red')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Traffic Flow')
plt.title('True Values vs. Predictions')
plt.show()
data.plot(figsize=(24,12))


print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
