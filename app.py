import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Config
st.set_page_config(layout="wide", page_title="ASL Translator")

# Load Model & Labels
@st.cache_resource
def load_model_and_labels():
    try:
        # Load the trained model
        model = tf.keras.models.load_model('asl_model.h5')
        # Load the class labels
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        # Reverse the dictionary to map {0: 'A', 1: 'B'...}
        labels = {v: k for k, v in class_indices.items()}
        return model, labels
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, labels = load_model_and_labels()

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("Real-Time ASL Translator")

# -----------------------------------------------------------------------------
# Image Processing Logic
# -----------------------------------------------------------------------------
def process_frame(frame):
    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(img_rgb)
        
        prediction_text = "Waiting for hand..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get Bounding Box
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding
                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Crop the hand
                hand_img = frame[y_min:y_max, x_min:x_max]
                
                if hand_img.size != 0:
                    try:
                        # Resize to 64x64 to match training
                        img_resize = cv2.resize(hand_img, (64, 64))
                        img_normalized = img_resize / 255.0
                        img_reshaped = np.reshape(img_normalized, (1, 64, 64, 3))
                        
                        # Predict
                        if model:
                            prediction = model.predict(img_reshaped, verbose=0)
                            predicted_index = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            if confidence > 0.5:
                                predicted_char = labels[predicted_index]
                                prediction_text = f"Prediction: {predicted_char} ({int(confidence*100)}%)"
                                
                                # Draw result on screen
                                cv2.rectangle(frame, (x_min, y_min-40), (x_max, y_min), (0, 255, 0), -1)
                                cv2.putText(frame, predicted_char, (x_min + 10, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    except Exception as e:
                        pass

    return frame

# -----------------------------------------------------------------------------
# WebRTC Video Processor
# -----------------------------------------------------------------------------
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the image
        img = process_frame(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout
st.write("### Camera Feed")
st.write("Allow camera access and show your hand.")

webrtc_streamer(
    key="asl-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)