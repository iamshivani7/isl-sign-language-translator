import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from isl_core import feature_columns, normalize_landmarks

DATA_PATH = Path(os.environ.get("ISL_DATA_PATH", "data/samples.csv"))
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
COLUMNS = ["label"] + feature_columns()

st.set_page_config(page_title="ISL Sign Translator", page_icon="🤟")
st.title("🤟 Sign Language to Text and Speech")
st.caption("Experimental static-gesture recognition prototype")
st.warning("This demo recognizes only gestures trained by the user. It is not a complete or validated ISL translator.")

if "samples" not in st.session_state:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        try:
            st.session_state.samples = pd.read_csv(DATA_PATH)
        except Exception:
            st.session_state.samples = pd.DataFrame(columns=COLUMNS)
    else:
        st.session_state.samples = pd.DataFrame(columns=COLUMNS)

def extract_hand(frame_rgb):
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.55) as detector:
        result = detector.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None
    points = [[lm.x, lm.y] for lm in result.multi_hand_landmarks[0].landmark]
    return normalize_landmarks(points)

st.header("1. Collect Gesture Samples")
label = st.text_input("Gesture label (example: HELLO)", value="HELLO").strip().upper()
photo = st.camera_input("Capture a training gesture")

if photo is not None and label:
    raw = np.frombuffer(photo.getvalue(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = extract_hand(rgb)
        if features is None:
            st.error("No hand detected. Try again.")
        elif st.button("Save this sample"):
            row = {"label": label}
            row.update(dict(zip(feature_columns(), features)))
            st.session_state.samples = pd.concat([st.session_state.samples, pd.DataFrame([row])], ignore_index=True)
            st.session_state.samples.to_csv(DATA_PATH, index=False)
            st.success("Training sample saved.")

samples = st.session_state.samples
if not samples.empty:
    st.write("Collected samples by label:")
    st.write(samples["label"].value_counts())
    if st.button("Clear all samples"):
        st.session_state.samples = pd.DataFrame(columns=COLUMNS)
        st.session_state.samples.to_csv(DATA_PATH, index=False)
        st.rerun()

st.header("2. Train the Model")
if not samples.empty and samples["label"].nunique() >= 2:
    if st.button("Train gesture classifier"):
        X = samples[feature_columns()].astype(float).to_numpy()
        y = samples["label"].astype(str).to_numpy()
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(3, len(y))))
        model.fit(X, y)
        st.session_state.model = model
        st.success("Model trained successfully.")
else:
    st.info("Collect examples for at least two different gesture labels before training.")

st.header("3. Predict a Gesture")
if "model" in st.session_state:
    test_photo = st.camera_input("Capture a gesture to recognize", key="prediction_camera")
    if test_photo is not None:
        raw = np.frombuffer(test_photo.getvalue(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features = extract_hand(rgb)
            if features is None:
                st.error("No hand detected.")
            else:
                prediction = st.session_state.model.predict([features])[0]
                st.session_state.prediction = str(prediction)
                st.subheader(f"Recognized label: {prediction}")
    if st.button("Speak last prediction"):
        prediction = st.session_state.get("prediction", "")
        if prediction:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(prediction.replace("_", " "))
                engine.runAndWait()
                st.success("Speech playback completed.")
            except Exception as error:
                st.error(f"Speech is unavailable in this environment. Details: {error}")
        else:
            st.info("Capture a gesture first.")
else:
    st.info("Train the model before making predictions.")

st.caption("Educational prototype. Do not use for critical communication.")
