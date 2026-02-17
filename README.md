# Real-Time Sign Language Translator

This project implements a Real-Time Sign Language Translator using computer vision and machine learning. It captures hand gestures via a webcam, detects hand landmarks using MediaPipe, and classifies them into letters (A-Z) using a Random Forest Classifier. The translation is displayed in a user-friendly Streamlit web application.

## Features

- **Real-Time Hand Detection:** Uses MediaPipe to detect hand landmarks with high accuracy.
- **Sign Classification:** Classifies hand gestures into 26 letters (A-Z) using a pre-trained Random Forest model.
- **Interactive Web Interface:** A polished Streamlit app (`app.py`) for easy interaction.
- **Sentence Construction:** Forms sentences from detected letters with smart suggestions.
- **Smart Suggestions:** Utilizes NLTK for word completion and next-word prediction based on context.
- **Text Editing:** Allows users to edit the detected text and save it to a file.
- **Data Collection & Training:** Scripts included for collecting custom datasets and retraining the model.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sign-language-translator.git
    cd sign-language-translator
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If `requirements.txt` is not provided, install the following packages manually:*
    ```bash
    pip install streamlit opencv-python mediapipe scikit-learn numpy nltk Pillow
    ```

## Usage

### 1. Running the Web Application

The main interface is the Streamlit app. To run it:

```bash
streamlit run app.py
```

This will open the application in your default web browser. Click the "Start Camera" checkbox to begin translating.

### 2. Running Inference Script (Alternative)

For a simple OpenCV window without the web interface:

```bash
python inference_classifier.py
```

Press `Esc` to exit the window.

### 3. Training Custom Model (Optional)

If you want to train the model on your own data:

1.  **Collect Data:**
    Run `collect_imgs.py` to capture images for each letter (A-Z).
    ```bash
    python collect_imgs.py
    ```
    Press `Q` when ready to start capturing for each class.

2.  **Create Dataset:**
    Process the collected images into a dataset of landmarks.
    ```bash
    python create_dataset.py
    ```
    This will generate `data.pickle`.

3.  **Train Classifier:**
    Train the Random Forest model.
    ```bash
    python train_classifier.py
    ```
    This will save the trained model as `model.p`.

## Project Structure

- `app.py`: Main Streamlit application.
- `collect_imgs.py`: Script to collect image data for training.
- `create_dataset.py`: Script to process images and create the dataset (`data.pickle`).
- `train_classifier.py`: Script to train the Random Forest model.
- `inference_classifier.py`: Script for real-time inference using OpenCV.
- `model.p`: Pre-trained Random Forest model.
- `data.pickle`: Processed dataset (landmarks and labels).
- `hand_landmarker.task`: MediaPipe model file for hand landmark detection.
- `data/`: Directory to store collected images (created by `collect_imgs.py`).
- `ML_Final_Report.pdf`: Project report.

## Requirements

- Python 3.7+
- streamlit
- opencv-python
- mediapipe
- scikit-learn
- numpy
- nltk
- Pillow

## Acknowledgments

- **MediaPipe:** For robust hand landmark detection.
- **Streamlit:** For building the interactive web application.
- **Scikit-learn:** For the Random Forest Classifier implementation.
- **NLTK:** For natural language processing features.
