import cv2
import numpy as np
import os
import time

import mediapipe as mp

# Path to save the dataset
dataset_dir = "dataset"

# Initialize camera
camera = cv2.VideoCapture(0)

# Prompt the user to enter a sentence
sentence = input("Enter a sentence: ")

# Create a folder to store the dataset for the sentence
sentence_dir = os.path.join(dataset_dir, sentence.replace(" ", "_"))
os.makedirs(sentence_dir, exist_ok=True)

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_draw= mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Countdown before starting recording
for count in range(5, 0, -1):
    print(count)
    time.sleep(1)

# Start recording the dataset
recording = True
while recording:
    ret, frame = camera.read()

    # Flip the frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Recording - Press 'q' to stop", frame)

    # Save the frame as an image
    image_path = os.path.join(sentence_dir, f"{len(os.listdir(sentence_dir)) + 1}.jpg")
    cv2.imwrite(image_path, frame)

    # Stop recording if hands are not detected in the frame
    if not results.multi_hand_landmarks:
        recording = False

    # Stop recording if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        recording = False

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
