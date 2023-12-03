import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset (replace with your own loading logic)
raw_dataset = np.load("sign_language_raw_dataset.npy")
sequence_dataset = np.load("sign_language_sequence_dataset.npy")

# Split the raw dataset into training and testing sets
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    raw_dataset[:, :-1], raw_dataset[:, -1], test_size=0.2, random_state=42)

# Split the sequence dataset into training and testing sets
X_sequence_train, X_sequence_test, y_sequence_train, y_sequence_test = train_test_split(
    sequence_dataset[:, :-1], sequence_dataset[:, -1], test_size=0.2, random_state=42)

# Define the LSTM model for raw data
raw_model = Sequential()
raw_model.add(LSTM(64, input_shape=(X_raw_train.shape[1], X_raw_train.shape[2])))
raw_model.add(Dense(1, activation='sigmoid'))
raw_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the raw model
raw_model.fit(X_raw_train, y_raw_train, epochs=10, batch_size=32, validation_data=(X_raw_test, y_raw_test))

# Evaluate the raw model
y_raw_pred = raw_model.predict_classes(X_raw_test)
accuracy_raw = accuracy_score(y_raw_test, y_raw_pred)
precision_raw = precision_score(y_raw_test, y_raw_pred)

# Define the LSTM model for sequence data
sequence_model = Sequential()
sequence_model.add(LSTM(64, input_shape=(X_sequence_train.shape[1], X_sequence_train.shape[2])))
sequence_model.add(Dense(1, activation='sigmoid'))
sequence_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the sequence model
sequence_model.fit(X_sequence_train, y_sequence_train, epochs=10, batch_size=32, validation_data=(X_sequence_test, y_sequence_test))

# Evaluate the sequence model
y_sequence_pred = sequence_model.predict_classes(X_sequence_test)
accuracy_sequence = accuracy_score(y_sequence_test, y_sequence_pred)
precision_sequence = precision_score(y_sequence_test, y_sequence_pred)

# Print evaluation metrics
print("Raw Data Model:")
print(f"Accuracy: {accuracy_raw}")
print(f"Precision: {precision_raw}")

print("\nSequence Data Model:")
print(f"Accuracy: {accuracy_sequence}")
print(f"Precision: {precision_sequence}")
