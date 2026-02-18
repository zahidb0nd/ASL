import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import Counter
import asl_utils
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from tensorflow import keras
import queue

st.set_page_config(page_title="Sign Language Translator", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    h1 { color: white !important; text-align: center; font-size: 3rem !important; 
         font-weight: 700 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    h2, h3 { color: white !important; font-weight: 600 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px; padding: 10px 20px;
        font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px; border: 2px solid #667eea; font-size: 1.1rem;
    }
    img { border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); }
    code {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important; padding: 5px 15px; border-radius: 8px;
        font-size: 1.5rem; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

@st.cache_resource
def load_tf_model():
    model = keras.models.load_model('./asl_model.h5')
    with open('./class_indices.json', 'r') as f:
        class_indices = json.load(f)
    labels_dict = {v: k for k, v in class_indices.items()}
    return model, labels_dict

model, labels_dict = load_tf_model()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

for key, default in [
    ('sentence', []), ('last_letter', ""), ('last_letter_time', 0),
    ('prediction_buffer', []), ('letter_cooldown', 0), ('edit_mode', False),
    ('current_letter', ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_tf_model()[0]
        self.labels_dict = load_tf_model()[1]
        
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.prediction_buffer = []
        self.last_letter = ""
        self.last_letter_time = 0
        self.letter_cooldown = 0
        self.frame_count = 0
        self.letter_queue = queue.Queue()
        self.detected_letter = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Only process every 2 frames
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")
        
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        
        current_time = time.time()
        current_letter = ""
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmark_points = []
                for lm in hand_landmarks:
                    x = int(lm.x * img.shape[1])
                    y = int(lm.y * img.shape[0])
                    landmark_points.append((x, y))
                
                for conn in HAND_CONNECTIONS:
                    cv2.line(frame_rgb, landmark_points[conn[0]], 
                            landmark_points[conn[1]], (102, 126, 234), 3)
                for pt in landmark_points:
                    cv2.circle(frame_rgb, pt, 6, (118, 75, 162), -1)
                
                # Extract hand region
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                
                h, w = img.shape[:2]
                x_min = max(0, int(min(x_coords) * w) - 20)
                x_max = min(w, int(max(x_coords) * w) + 20)
                y_min = max(0, int(min(y_coords) * h) - 20)
                y_max = min(h, int(max(y_coords) * h) + 20)
                
                hand_img = frame_rgb[y_min:y_max, x_min:x_max]
                
                if hand_img.size > 0:
                    hand_img_resized = cv2.resize(hand_img, (64, 64))
                    hand_img_array = np.expand_dims(hand_img_resized / 255.0, axis=0)
                    
                    prediction = self.model.predict(hand_img_array, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    predicted_letter = self.labels_dict[predicted_class]
                    
                    self.prediction_buffer.append(predicted_letter)
                    if len(self.prediction_buffer) > 10:
                        self.prediction_buffer.pop(0)
                    
                    current_letter = Counter(self.prediction_buffer).most_common(1)[0][0] \
                        if len(self.prediction_buffer) >= 5 else predicted_letter
                    
                    if current_letter == self.last_letter:
                        if current_time - self.last_letter_time > 1.0:
                            if current_time > self.letter_cooldown:
                                self.letter_queue.put(current_letter)
                                self.letter_cooldown = current_time + 2.0
                    else:
                        self.last_letter = current_letter
                        self.last_letter_time = current_time
                    
                    cv2.putText(frame_rgb, current_letter, (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_TRIPLEX, 2.5, (255, 255, 255), 4)
                    cv2.putText(frame_rgb, current_letter, (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_TRIPLEX, 2.5, (102, 126, 234), 2)
        else:
            self.last_letter = ""
        
        self.detected_letter = current_letter
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")


st.markdown("<h1>🤟 Real-Time Sign Language Translator</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.subheader("📹 Live Camera Feed")
    st.success("📱 Works on mobile and desktop!")
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

@st.fragment(run_every=0.5)
def poll_and_update_ui():
    if ctx.video_processor:
        # Check for new committed letters
        try:
            while True:
                letter = ctx.video_processor.letter_queue.get_nowait()
                st.session_state.sentence.append(letter)
        except queue.Empty:
            pass

        # Update current letter for display
        st.session_state.current_letter = ctx.video_processor.detected_letter

    st.subheader("📝 Detected Sentence")
    
    current_letter = st.session_state.get("current_letter", "")
    if current_letter:
        st.markdown(f"**Current Letter:** `{current_letter}`")
    else:
        st.markdown("**Current Letter:** _None_")
    
    if st.button("✏️ Edit Text" if not st.session_state.edit_mode else "✅ Done Editing", key="edit_btn"):
        st.session_state.edit_mode = not st.session_state.edit_mode
        st.rerun()
    
    sentence_str = "".join(st.session_state.sentence)
    if st.session_state.edit_mode:
        edited_text = st.text_area("✍️ Edit your text:", sentence_str, height=100, key="editor")
        if edited_text != sentence_str:
            st.session_state.sentence = list(edited_text)
    else:
        st.markdown(f"### {sentence_str if sentence_str else '_Start signing..._'}")
    
    st.markdown("---")
    st.subheader("💡 Smart Suggestions")
    
    words_in_sentence = sentence_str.strip().split()
    ends_with_space = sentence_str.endswith(" ")
    
    if ends_with_space and words_in_sentence:
        next_suggestions = asl_utils.get_next_word_suggestions(words_in_sentence[-1])
        if next_suggestions:
            st.markdown("**Next word:**")
            for idx, sug in enumerate(next_suggestions):
                if st.button(f"➡️ {sug.upper()}", key=f"next_{sug}_{idx}"):
                    st.session_state.sentence.extend(sug.upper() + " ")
                    st.rerun()
        else:
            st.info("✨ Keep signing...")
    elif words_in_sentence and not ends_with_space:
        word_sugs = asl_utils.get_word_suggestions(words_in_sentence[-1])
        if word_sugs:
            st.markdown("**Complete word:**")
            for idx, sug in enumerate(word_sugs[:3]):
                if st.button(f"📝 {sug.upper()}", key=f"word_{sug}_{idx}"):
                    completed = " ".join(words_in_sentence[:-1])
                    st.session_state.sentence = list((completed + " " if completed else "") + sug.upper() + " ")
                    st.rerun()
        else:
            st.info("✨ Keep signing...")
    else:
        st.info("✨ Start signing to see suggestions")
    
    st.markdown("---")
    st.subheader("🎮 Controls")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        if st.button("➕", use_container_width=True, key="space_btn"):
            st.session_state.sentence.append(" ")
            st.rerun()
    with col_b:
        if st.button("⬅️", use_container_width=True, key="back_btn"):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.rerun()
    with col_c:
        if st.button("🗑️", use_container_width=True, key="clear_btn"):
            st.session_state.sentence = []
            st.rerun()
    with col_d:
        if st.button("💾", use_container_width=True, key="save_btn"):
            if sentence_str:
                st.success("✅ Saved!")
            else:
                st.warning("⚠️ Nothing!")

with col2:
    poll_and_update_ui()

with col3:
    st.subheader("📚 ASL Reference")

    @st.fragment(run_every=0.5)
    def reference_view():
        current_letter = st.session_state.get("current_letter", "")
        if current_letter:
            st.image(asl_utils.get_asl_image_url(current_letter),
                    caption=f"ASL Sign for '{current_letter}'", use_container_width=True)
            st.markdown(f"**Description:**\n\n{asl_utils.get_asl_description(current_letter)}")
        else:
            st.info("Start signing to see reference details.")

    reference_view()
