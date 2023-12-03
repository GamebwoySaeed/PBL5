import cv2
import mediapipe as mp
import numpy as np
import time
import os

seq_length = 10
secs_for_action = 5  # Time in seconds to record each action

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # Initialize webcam

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)  # Create folder to save dataset

while True:
    user_input = input("Enter a word or sentence (or press 'X' to exit): ")

    if user_input.upper() == 'X':
        break

    while cap.isOpened():
        data = []  # List to store recorded data

        # Wait for a few seconds before starting the action recording
        print(f"Collecting {user_input} action...")
        time.sleep(3)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]  # Store landmark x,y,z coordinates

                    # Get direction vectors of bones from parent to child joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joints
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joints
                    v = v2 - v1

                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Calculate angles using arccos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                               v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                               v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radians to degrees

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, 0)  # Append action number (0 for user input)

                    d = np.concatenate([joint.flatten(), angle_label])  # Flatten joint coordinates and concatenate with the label

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

        data = np.array(data)  # Convert data to a numpy array
        print(user_input, data.shape)
        
        # Create a directory for the current sentence if it doesn't exist
        sentence_dir = os.path.join('dataset', user_input)
        os.makedirs(sentence_dir, exist_ok=True)
        
        # Get the count of existing recordings for the current sentence
        folder_index = len(os.listdir(sentence_dir)) + 1
        
        np.save(os.path.join(sentence_dir, f'raw_{user_input}_{folder_index}_{created_time}'), data)  # Save data as a .npy file

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(user_input, full_seq_data.shape)
        np.save(os.path.join(sentence_dir, f'seq_{user_input}_{folder_index}_{created_time}'), full_seq_data)  # Save sequence data as a .npy file

        print("Recording saved.")

        print("Press any key to continue recording or 'X' to exit")
        key = cv2.waitKey(0)
        
        if key == ord('x'):
            break

    if key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
