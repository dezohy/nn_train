import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([50, 100, 150, 200, 250], dtype=float)
Y = np.array([100, 200, 300, 400, 500], dtype=float)

X_normalized = X / np.max(X)
Y_normalized = Y / np.max(X)

model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X_normalized, Y_normalized, epochs=3000)

input_data = np.array([100], dtype=float) / np.max(X)
prediction = model.predict(input_data)
predicted_price = prediction * np.max(Y)
print('Передбачена віртість будинку:', predicted_price[0])
