import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Set up camera
camera = cv2.VideoCapture(0)

# Create a directory to store the dataset
dataset_dir = "sign_language_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize recording status and sentence
recording = False
sentence = ""
frames = []

# Set up Mediapipe hand detection
mp_hands = mp.solutions.hands
mp_draw= mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

while True:
    # Prompt user to enter a sentence or start/stop recording
    if not recording:
        sentence = input("Enter a sentence to record (or 'q' to quit): ")
        if sentence.lower() == 'q':
            break
        print(f"Press 'r' to start recording for '{sentence}'...")
    else:
        print(f"Recording for '{sentence}'... Press 'r' to stop.")

    # Start recording when 'r' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('r') and not recording:
        recording = True
        print(f"Started recording for '{sentence}'... Press 'r' to stop.")

    # Stop recording when 'r' key is pressed again
    if cv2.waitKey(1) & 0xFF == ord('r') and recording:
        recording = False
        print(f"Stopped recording for '{sentence}'.")
        time.sleep(1)  # Add a delay to avoid key overlap when starting a new recording

        # Save frames as NumPy array
        if len(frames) > 0:
            frames_array = np.array(frames)
            filename = f"{sentence}_{time.strftime('%Y%m%d%H%M%S')}.npy"
            file_path = os.path.join(dataset_dir, filename)
            np.save(file_path, frames_array)
            frames = []

    # Capture frame from the camera
    ret, frame = camera.read()

    # Flip the frame horizontally for natural hand orientation
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hand detection
    results = hands.process(frame_rgb)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow("Sign Language Dataset Collection", frame)

    # Append frame to the frames list if recording
    if recording:
        frames.append(frame)

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
