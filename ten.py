import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
import matplotlib

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
x = layers.Dense(256, activation='sigmoid')(inputs)
x = layers.Dense(128, activation='sigmoid')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae', 'accuracy']
)

class DebugCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss')}, mae = {logs.get('mae')}")

model.fit(X_train, y_train, batch_size=100, epochs=1500, verbose=2, callbacks=[DebugCallback()])
model.evaluate(X_test, y_test, batch_size=300, verbose=2)

# Save the trained model
model.save('esi_model.h5')

# Use a representative subset of the background data for SHAP
background = shap.sample(X_test, 100)  # Using 100 samples as background

# Evaluate feature importance using SHAP
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test[:10])  # Using a smaller test set for demonstration

shap.summary_plot(shap_values, X_test[:10], feature_names=data.columns[:-1].tolist())

# Load the saved model for prediction
model = keras.models.load_model('esi_model.h5')

# Predict ESI for new params
venus_params =[
    0.815,  # P_MASS
    None,   # P_MASS_ERROR_MIN
    None,   # P_MASS_ERROR_MAX
    None,   # P_MASS_LIMIT
    0.949,  # P_RADIUS
    None,   # P_RADIUS_ERROR_MIN
    None,   # P_RADIUS_ERROR_MAX
    None,   # P_RADIUS_LIMIT
    225,    # P_PERIOD
    None,   # P_PERIOD_ERROR_MIN
    None,   # P_PERIOD_ERROR_MAX
    None,   # P_PERIOD_LIMIT
    0.723,  # P_SEMI_MAJOR_AXIS
    None,   # P_SEMI_MAJOR_AXIS_ERROR_MIN
    None,   # P_SEMI_MAJOR_AXIS_ERROR_MAX
    None,   # P_SEMI_MAJOR_AXIS_LIMIT
    0.0068, # P_ECCENTRICITY
    None,   # P_ECCENTRICITY_ERROR_MIN
    None,   # P_ECCENTRICITY_ERROR_MAX
    None,   # P_ECCENTRICITY_LIMIT
    3.39,   # P_INCLINATION
    None,   # P_INCLINATION_ERROR_MIN
    None,   # P_INCLINATION_ERROR_MAX
    None,   # P_INCLINATION_LIMIT
    None,   # P_OMEGA
    None,   # P_OMEGA_ERROR_MIN
    None,   # P_OMEGA_ERROR_MAX
    None,   # P_OMEGA_LIMIT
    6.7525, # S_RA
    -16.7161, # S_DEC
    "06h 45m 09s", # S_RA_STR
    "-16° 42' 58\"", # S_DEC_STR
    -26.74, # S_MAG
    None,   # S_MAG_ERROR_MIN
    None,   # S_MAG_ERROR_MAX
    0.723,  # S_DISTANCE
    None,   # S_DISTANCE_ERROR_MIN
    None,   # S_DISTANCE_ERROR_MAX
    5778,   # S_TEMPERATURE
    None,   # S_TEMPERATURE_ERROR_MIN
    None,   # S_TEMPERATURE_ERROR_MAX
    None,   # S_TEMPERATURE_LIMIT
    1,      # S_MASS
    None,   # S_MASS_ERROR_MIN
    None,   # S_MASS_ERROR_MAX
    None,   # S_MASS_LIMIT
    1,      # S_RADIUS
    None,   # S_RADIUS_ERROR_MIN
    None,   # S_RADIUS_ERROR_MAX
    None,   # S_RADIUS_LIMIT
    0.0122, # S_METALLICITY
    None,   # S_METALLICITY_ERROR_MIN
    None,   # S_METALLICITY_ERROR_MAX
    None,   # S_METALLICITY_LIMIT
    4.6,    # S_AGE
    None,   # S_AGE_ERROR_MIN
    None,   # S_AGE_ERROR_MAX
    None,   # S_AGE_LIMIT
    0,      # S_LOG_LUM
    None,   # S_LOG_LUM_ERROR_MIN
    None,   # S_LOG_LUM_ERROR_MAX
    None,   # S_LOG_LUM_LIMIT
    4.44,   # S_LOG_G
    None,   # S_LOG_G_ERROR_MIN
    None,   # S_LOG_G_ERROR_MAX
    None,   # S_LOG_G_LIMIT
    10.36,  # P_ESCAPE
    None,   # P_POTENTIAL
    8.87,   # P_GRAVITY
    5.24,   # P_DENSITY
    1.07,   # P_HILL_SPHERE
    0.723,  # P_DISTANCE
    None,   # P_PERIASTRON
    None,   # P_APASTRON
    None,   # P_DISTANCE_EFF
    1.92,   # P_FLUX
    None,   # P_FLUX_MIN
    None,   # P_FLUX_MAX
    232,    # P_TEMP_EQUIL
    None,   # P_TEMP_EQUIL_MIN
    None,   # P_TEMP_EQUIL_MAX
    737,    # P_TEMP_SURF
    None,   # P_TEMP_SURF_MIN
    None,   # P_TEMP_SURF_MAX
    "06h 45m 09s", # S_RA_TXT
    "-16° 42' 58\"", # S_DEC_TXT
    1,      # S_LUMINOSITY
    0.95,   # S_HZ_OPT_MIN
    1.37,   # S_HZ_OPT_MAX
    0.95,   # S_HZ_CON_MIN
    1.37,   # S_HZ_CON_MAX
    None,   # S_HZ_CON0_MIN
    None,   # S_HZ_CON0_MAX
    None,   # S_HZ_CON1_MIN
    None,   # S_HZ_CON1_MAX
    None,   # S_SNOW_LINE
    None,   # S_ABIO_ZONE
    None,   # S_TIDAL_LOCK
    None,   # P_HABZONE_OPT
    None,   # P_HABZONE_CON
    "No"    # P_HABITABLE
]
new_params = np.array(venus_params).reshape(1, -1)
new_params_scaled = scaler_X.transform(new_params)
predicted_esi_scaled = model.predict(new_params_scaled)
predicted_esi = scaler_y.inverse_transform(predicted_esi_scaled)

print(f"Predicted ESI: {predicted_esi[0][0]}")
