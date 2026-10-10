import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Parameter settings
# -------------------------------
simulated_data_path = '/root/autodl/LSTM/data/SIM.csv'
lstm_gt_path = '/root/autodl/LSTM/data/TRUE.csv'
model_save_path = '/root/autodl/LSTM/model/lstm_model.keras'

time_steps = 20
batch_size = 32
epochs = 100
validation_split = 0.2

# -------------------------------
# 2. Data loading
# -------------------------------
print("Loading data...")

simulated_data = pd.read_csv(simulated_data_path, header=None)
lstm_gt = pd.read_csv(lstm_gt_path, header=None)

print("Simulated data shape:", simulated_data.shape)
print("GT data shape:", lstm_gt.shape)

if simulated_data.shape[0] != lstm_gt.shape[0]:
    raise Exception("Simulated data and GT data row counts are inconsistent.")

# -------------------------------
# 3. Data preprocessing
# -------------------------------
X = simulated_data.values
y = lstm_gt.iloc[:, 1:3].values

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps, :])
        ys.append(y[i + time_steps, :])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)
print(f"Total training samples generated: {X_seq.shape[0]}")

# -------------------------------
# 4. Build the BiLSTM model
# -------------------------------
model = Sequential([
    Bidirectional(LSTM(100, activation='tanh', return_sequences=True), input_shape=(time_steps, X_seq.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(100, activation='tanh')),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(2)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# -------------------------------
# 5. Train the model
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_seq,
    y_seq,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    verbose=1,
    callbacks=[early_stop]
)

# -------------------------------
# 6. Save the model
# -------------------------------
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")
print("Training completed.")