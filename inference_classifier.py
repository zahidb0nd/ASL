import numpy as np
import cv2
import mediapipe as mp
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Setup MediaPipe Hand Landmarker with new API
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Make sure hand_landmarker.task is in the same folder
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks manually
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                x_, y_, data_hands = [], [], []
                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks:
                    data_hands.append(landmark.x - min(x_))
                    data_hands.append(landmark.y - min(y_))

                prediction = model.predict([np.asarray(data_hands)])
                predicted_label = labels_dict[int(prediction[0])]

                # Display prediction
                x1 = int(min(x_) * frame.shape[1])
                y1 = int(min(y_) * frame.shape[0])
                cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.putText(frame, "Press 'Esc' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()