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
        model = tf.keras.models.load_model('asl_model.h5')
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

# CSS for styling
st.markdown("""
    <style>
    .main-text { font-size: 40px; font-weight: bold; text-align: center; color: #4CAF50; }
    .sub-text { font-size: 20px; text-align: center; color: #555; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-text">Real-Time ASL Translator</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Image Processing Logic (The Brain)
# -----------------------------------------------------------------------------
def process_frame(frame, hands_detector):
    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    prediction_text = "Waiting for hand..."
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 1. Get Bounding Box of the hand
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
            
            # 2. Crop the hand
            hand_img = frame[y_min:y_max, x_min:x_max]
            
            if hand_img.size != 0:
                try:
                    # 3. Preprocess for TensorFlow (Must match training: 64x64, normalized)
                    img_resize = cv2.resize(hand_img, (64, 64))
                    img_normalized = img_resize / 255.0
                    img_reshaped = np.reshape(img_normalized, (1, 64, 64, 3))
                    
                    # 4. Predict
                    if model:
                        prediction = model.predict(img_reshaped, verbose=0)
                        predicted_index = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        if confidence > 0.7: # Threshold to avoid noise
                            predicted_char = labels[predicted_index]
                            prediction_text = f"Prediction: {predicted_char} ({int(confidence*100)}%)"
                            
                            # Draw text on screen
                            cv2.rectangle(frame, (x_min, y_min-40), (x_max, y_min), (0, 255, 0), -1)
                            cv2.putText(frame, predicted_char, (x_min + 10, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                except Exception as e:
                    pass

    return frame, prediction_text

# -----------------------------------------------------------------------------
# WebRTC Streamer (Video Logic)
# -----------------------------------------------------------------------------
class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the image
        img, _ = process_frame(img, self.hands)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Camera Feed")
    webrtc_streamer(
        key="asl-translator",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.write("### Instructions")
    st.info("1. Allow camera access.\n2. Ensure your hand is visible.\n3. The app will detect gestures automatically.")
    if not model:
        st.warning("⚠️ Model not found! Please upload 'asl_model.h5' and 'class_indices.json' to your GitHub repo.")