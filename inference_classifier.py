import cv2
import numpy as np
import mediapipe as mp
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras
import asl_utils
import os

# Check if model exists
if not os.path.exists('./asl_model.h5'):
    print("Error: asl_model.h5 not found.")
    exit(1)

# Load model
try:
    model = keras.models.load_model('./asl_model.h5')
    with open('./class_indices.json', 'r') as f:
        class_indices = json.load(f)
    labels_dict = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

cap = cv2.VideoCapture(0)

# Setup MediaPipe Hand Landmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Check if task file exists
if not os.path.exists('hand_landmarker.task'):
    print("Error: hand_landmarker.task not found.")
    exit(1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

# Connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

with HandLandmarker.create_from_options(options) as landmarker:
    print("Starting camera... Press 'Esc' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw landmarks
                landmark_points = []
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    landmark_points.append((x, y))

                for conn in HAND_CONNECTIONS:
                     cv2.line(frame, landmark_points[conn[0]], landmark_points[conn[1]], (0, 255, 0), 2)

                for pt in landmark_points:
                    cv2.circle(frame, pt, 4, (0, 255, 0), -1)

                # Bounding box logic
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]

                h, w = frame.shape[:2]
                x_min = max(0, int(min(x_coords) * w) - 20)
                x_max = min(w, int(max(x_coords) * w) + 20)
                y_min = max(0, int(min(y_coords) * h) - 20)
                y_max = min(h, int(max(y_coords) * h) + 20)

                # Draw bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Extract hand region from RGB frame for prediction
                hand_img = frame_rgb[y_min:y_max, x_min:x_max]
                
                if hand_img.size > 0:
                    try:
                        hand_img_resized = cv2.resize(hand_img, (64, 64))
                        hand_img_array = np.expand_dims(hand_img_resized / 255.0, axis=0)

                        prediction = model.predict(hand_img_array, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        predicted_letter = labels_dict[predicted_class]

                        # Display prediction
                        cv2.putText(frame, predicted_letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                        # Get description
                        desc = asl_utils.get_asl_description(predicted_letter)
                        # We limit printing to avoid console spam, or print only on change (not implemented here for simplicity)
                        # print(f"Predicted: {predicted_letter} | Ref: {desc}")
                    except Exception as e:
                        print(f"Prediction error: {e}")

        cv2.putText(frame, "Press 'Esc' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
