import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import Counter, defaultdict
import nltk
from nltk.corpus import words, brown
from nltk import bigrams
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('words')
    nltk.download('brown')

# Page config
st.set_page_config(
    page_title="Sign Language Translator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0 !important;
    }
    h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 1.1rem;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        margin: 20px 0;
    }
    code {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 5px 15px;
        border-radius: 8px;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# RTC config (no Twilio)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load model
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

model = load_model()

# Build language model
@st.cache_resource
def build_language_model():
    english_words = set(w.lower() for w in words.words() if len(w) > 2)
    brown_words = [w.lower() for w in brown.words()]
    bigram_model = defaultdict(Counter)
    for w1, w2 in bigrams(brown_words):
        if w1.isalpha() and w2.isalpha():
            bigram_model[w1][w2] += 1
    return english_words, bigram_model

english_words, bigram_model = build_language_model()

def get_word_suggestions(partial_word, max_suggestions=5):
    if not partial_word or len(partial_word) < 2:
        return []
    partial_lower = partial_word.lower()
    suggestions = [w for w in english_words if w.startswith(partial_lower)]
    suggestions.sort(key=lambda x: (len(x), x))
    return suggestions[:max_suggestions]

def get_next_word_suggestions(last_word, max_suggestions=3):
    if not last_word:
        return []
    last_word_lower = last_word.lower()
    if last_word_lower in bigram_model:
        next_words = bigram_model[last_word_lower].most_common(max_suggestions)
        return [word for word, _ in next_words]
    return []

# ASL descriptions
ASL_DESCRIPTIONS = {
    'A': 'Make a fist with your dominant hand, thumb resting on the side.',
    'B': 'Hold your fingers straight up and together, thumb tucked across palm.',
    'C': 'Curve your fingers and thumb into a C shape.',
    'D': 'Point index finger up, remaining fingers touch thumb forming a circle.',
    'E': 'Curl all fingers down, thumb tucked under fingers.',
    'F': 'Connect index finger and thumb in a circle, other fingers extended.',
    'G': 'Point index finger sideways, thumb parallel pointing out.',
    'H': 'Point index and middle fingers sideways together.',
    'I': 'Raise pinky finger, other fingers curled into fist.',
    'J': 'Raise pinky and draw a J shape in the air.',
    'K': 'Index and middle fingers up in a V, thumb between them.',
    'L': 'Make an L shape with index finger up and thumb out.',
    'M': 'Tuck three fingers over thumb.',
    'N': 'Tuck two fingers over thumb.',
    'O': 'Curve all fingers to meet thumb, forming an O shape.',
    'P': 'Like K but pointed downward.',
    'Q': 'Like G but pointed downward.',
    'R': 'Cross middle finger over index finger.',
    'S': 'Make a fist with thumb over fingers.',
    'T': 'Make a fist with thumb between index and middle fingers.',
    'U': 'Hold index and middle fingers straight up together.',
    'V': 'Hold index and middle fingers up in a V (peace sign).',
    'W': 'Hold index, middle, and ring fingers up spread apart.',
    'X': 'Hook index finger into a curved/bent position.',
    'Y': 'Extend thumb and pinky, curl other fingers (hang loose).',
    'Z': 'Draw a Z in the air with your index finger.',
}

def get_asl_description(letter):
    return ASL_DESCRIPTIONS.get(letter.upper(), "No description available.")

def get_asl_image_url(letter):
    return f"https://www.handspeak.com/word/search/img/asl-alphabet/{letter.lower()}.jpg"

# MediaPipe labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
               16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# Session state defaults
for key, default in [
    ('sentence', []),
    ('last_letter', ""),
    ('last_letter_time', 0),
    ('prediction_buffer', []),
    ('letter_cooldown', 0),
    ('edit_mode', False),
    ('current_letter', ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()

        # ✅ Create HandLandmarker ONCE here, not every frame
        BaseOptions = python.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.prediction_buffer = []
        self.last_letter = ""
        self.last_letter_time = 0
        self.letter_cooldown = 0
        self.current_letter = ""
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # ✅ Only process every 3rd frame to reduce CPU load
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24"
            )

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # ✅ Reuse the same landmarker instance
        results = self.landmarker.detect(mp_image)

        current_time = time.time()
        current_letter = ""

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmark_points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    landmark_points.append((x, y))

                for connection in HAND_CONNECTIONS:
                    cv2.line(frame_rgb,
                             landmark_points[connection[0]],
                             landmark_points[connection[1]],
                             (102, 126, 234), 3)

                for point in landmark_points:
                    cv2.circle(frame_rgb, point, 6, (118, 75, 162), -1)

                x_, y_, data_hands = [], [], []
                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                for landmark in hand_landmarks:
                    data_hands.append(landmark.x - min(x_))
                    data_hands.append(landmark.y - min(y_))

                prediction = self.model.predict([np.asarray(data_hands)])
                predicted_letter = labels_dict[int(prediction[0])]

                self.prediction_buffer.append(predicted_letter)
                if len(self.prediction_buffer) > 10:
                    self.prediction_buffer.pop(0)

                current_letter = Counter(self.prediction_buffer).most_common(1)[0][0] \
                    if len(self.prediction_buffer) >= 5 else predicted_letter

                if current_letter == self.last_letter:
                    if current_time - self.last_letter_time > 1.0:
                        if current_time > self.letter_cooldown:
                            st.session_state.sentence.append(current_letter)
                            st.session_state.current_letter = current_letter
                            self.letter_cooldown = current_time + 2.0
                else:
                    self.last_letter = current_letter
                    self.last_letter_time = current_time

                x1 = int(min(x_) * img.shape[1])
                y1 = int(min(y_) * img.shape[0])
                cv2.putText(frame_rgb, current_letter, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 2.5, (255, 255, 255), 4)
                cv2.putText(frame_rgb, current_letter, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 2.5, (102, 126, 234), 2)
        else:
            self.last_letter = ""

        self.current_letter = current_letter
        st.session_state.current_letter = current_letter

        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")


# ---- UI ----
st.markdown("<h1>🤟 Real-Time Sign Language Translator</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.subheader("📹 Live Camera Feed")
    st.info("📱 Works on both desktop and mobile!")
    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📝 Detected Sentence")

    current_letter = st.session_state.get("current_letter", "")
    if current_letter:
        st.markdown(f"**Current Letter:** `{current_letter}`")
    else:
        st.markdown("**Current Letter:** _None_")

    if st.button("✏️ Edit Text" if not st.session_state.edit_mode else "✅ Done Editing"):
        st.session_state.edit_mode = not st.session_state.edit_mode
        st.rerun()

    sentence_str = "".join(st.session_state.sentence)
    if st.session_state.edit_mode:
        edited_text = st.text_area("✍️ Edit your text:", sentence_str, height=100, key="editor")
        if edited_text != sentence_str:
            st.session_state.sentence = list(edited_text)
            sentence_str = edited_text
    else:
        st.markdown(f"### {sentence_str if sentence_str else '_Start signing..._'}")

    st.markdown("---")
    st.subheader("💡 Smart Suggestions")

    words_in_sentence = sentence_str.strip().split()
    ends_with_space = sentence_str.endswith(" ")

    if ends_with_space and words_in_sentence:
        previous_word = words_in_sentence[-1]
        next_suggestions = get_next_word_suggestions(previous_word)
        if next_suggestions:
            st.markdown("**Next word:**")
            for idx, sug in enumerate(next_suggestions):
                if st.button(f"➡️ {sug.upper()}", key=f"next_{sug}_{idx}"):
                    st.session_state.sentence.extend(sug.upper() + " ")
                    st.rerun()
        else:
            st.info("✨ Keep signing...")
    elif words_in_sentence and not ends_with_space:
        current_word = words_in_sentence[-1]
        word_sugs = get_word_suggestions(current_word)
        if word_sugs:
            st.markdown("**Complete word:**")
            for idx, sug in enumerate(word_sugs[:3]):
                if st.button(f"📝 {sug.upper()}", key=f"word_{sug}_{idx}"):
                    completed = " ".join(words_in_sentence[:-1])
                    st.session_state.sentence = list(
                        (completed + " " if completed else "") + sug.upper() + " ")
                    st.rerun()
        else:
            st.info("✨ Keep signing...")
    else:
        st.info("✨ Start signing to see suggestions")

    st.markdown("---")
    st.subheader("🎮 Controls")
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    with col_a:
        if st.button("➕ Space", use_container_width=True):
            st.session_state.sentence.append(" ")
            st.rerun()
    with col_b:
        if st.button("⬅️ Delete", use_container_width=True):
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.rerun()
    with col_c:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence = []
            st.session_state.last_letter = ""
            st.rerun()
    with col_d:
        if st.button("💾 Save", use_container_width=True):
            sentence_str = "".join(st.session_state.sentence)
            if sentence_str:
                with open("detected_text.txt", "w") as f:
                    f.write(sentence_str)
                st.success("✅ Saved!")
            else:
                st.warning("⚠️ Nothing to save!")

with col3:
    st.subheader("📚 ASL Reference")
    if current_letter:
        desc = get_asl_description(current_letter)
        img_url = get_asl_image_url(current_letter)
        st.image(img_url, caption=f"ASL Sign for '{current_letter}'", use_container_width=True)
        st.markdown(f"**Description:**\n\n{desc}")
    else:
        st.info("Start signing to see reference details.")
