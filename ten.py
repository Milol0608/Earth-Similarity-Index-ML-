import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
data = pd.read_excel(r'C:\Users\milob\OneDrive\Escritorio\PLANETS_WITH_ESI.xlsm')

# Drop non-numeric columns
data = data.select_dtypes(include=[np.number])

# Fill missing values with the mean of each column
data = data.fillna(data.mean())

# Split the data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values  # ESI values

# Scale features and target variable
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Check for NaNs or infinities
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the model
inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['mae', 'accuracy']
)

class DebugCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss')}, mae = {logs.get('mae')}")

model.fit(X_train, y_train, batch_size=100, epochs=150, verbose=2, callbacks=[DebugCallback()])
model.evaluate(X_test, y_test, batch_size=100, verbose=2)
