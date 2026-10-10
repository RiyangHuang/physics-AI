import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Dense, TimeDistributed, Masking
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. Load the same dataset
# ============================================================================
data = np.load('/root/autodl-tmp/simulated_data.npy')
T, F = data.shape[1], data.shape[2]

# ============================================================================
# 2. Apply global normalization
# ============================================================================
scaler = MinMaxScaler()
data_flat = data.reshape(-1, F)
data_scaled = scaler.fit_transform(data_flat).reshape(-1, T, F)

# ============================================================================
# 3. Construct samples for next-step prediction (many-to-many)
# ============================================================================
X, y = [], []
for traj in data_scaled:
    X.append(traj[:-1])      # Time steps 0 through T-2
    y.append(traj[1:])       # Time steps 1 through T-1
X = np.array(X)              # (samples, T-1, 2)
y = np.array(y)

# ============================================================================
# 4. Split the data into training and validation sets
# ============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 5. Build a 1D CNN model that accepts variable-length sequences
# ============================================================================
model = Sequential([
    Input(shape=(None, 2)),                    # Variable number of time steps
    Masking(mask_value=0.0),                   # Optional: for future padding to a uniform length
    # Stack three causal dilated convolution layers, preserving the number of time steps
    Conv1D(64, 3, padding='causal', dilation_rate=1, activation='relu'),
    Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu'),
    Conv1D(64, 3, padding='causal', dilation_rate=4, activation='relu'),
    # Use a 1x1 convolution to reduce the channels to 2, outputting (az, po) at each time step
    Conv1D(2, 1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ============================================================================
# 6. Train the model
# ============================================================================
es = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[es]
)

# ============================================================================
# 7. Save the model and scaler parameters
# ============================================================================
model.save('/root/autodl-tmp/cnn_dynamic.h5')
np.save('/root/autodl-tmp/cnn_scaler_data_min.npy', scaler.data_min_)
np.save('/root/autodl-tmp/cnn_scaler_data_max.npy', scaler.data_max_)
print('model saved')