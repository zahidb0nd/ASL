import os
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = './data'

# Setup MediaPipe Hand Landmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

labels = []
data = []

with HandLandmarker.create_from_options(options) as landmarker:
    # Iterating over the classes
    for dir_ in os.listdir(DATA_DIR):
        for image_file in os.listdir(os.path.join(DATA_DIR, dir_)):
            landmark_data = []
            x_ = []
            y_ = []
            
            # Loading the data
            image_path = os.path.join(DATA_DIR, dir_, image_file)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_results = landmarker.detect(mp_image)
            
            # Getting hand landmarks
            if detection_results.hand_landmarks:
                for hand_landmarks in detection_results.hand_landmarks:
                    for landmark in hand_landmarks:
                        x_.append(landmark.x)
                        y_.append(landmark.y)
                    
                    for landmark in hand_landmarks:
                        landmark_data.append(landmark.x - min(x_))
                        landmark_data.append(landmark.y - min(y_))
                
                data.append(landmark_data)
                labels.append(dir_)

print(f'Processed {len(data)} images across {len(set(labels))} classes')

# Saving data and labels
with open('data.pickle', 'wb') as data_file:
    pickle.dump({'data': data, 'labels': labels}, data_file)

print('Dataset created successfully!')