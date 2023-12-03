import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = 'dataset/'  # Path to the dataset folder
actions = ['action1', 'action2', 'action3']  # List of actions

# Prepare the data
seq_length = 10
X = []
y = []

for action in actions:
    raw_data = np.load(f"{dataset_path}raw_{action}.npy")
    seq_data = np.load(f"{dataset_path}seq_{action}.npy")
    X.append(raw_data)
    y.append(seq_data)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(seq_length * 3)  # Output layer with flattened landmarks (seq_length * 3 features)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# Save the trained model
model.save('trained_model.h5')
