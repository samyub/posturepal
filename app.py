import streamlit as st
import cv2
import numpy as np
import joblib

from mediapipe_pose import MediaPipePoseExtractor
from normalize_landmarks import normalize_landmarks
from features import extract_features
from posture_rules import posture_score

# -----------------------------
# LOAD MODELS
# -----------------------------
classifier = joblib.load("pose_classifier.joblib")
label_encoder = joblib.load("label_encoder.joblib")

pose_extractor = MediaPipePoseExtractor()

st.set_page_config(page_title="PosturePal", layout="centered")
st.title("🧘 PosturePal — Yoga Posture Checker")

uploaded = st.file_uploader("Upload a yoga pose image", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    landmarks = pose_extractor.extract_landmarks(image)

    if landmarks is None:
        st.error("No pose detected. Try a clearer image.")
    else:
        normalized = normalize_landmarks(landmarks)
        features = extract_features(normalized).reshape(1, -1)

        pred = classifier.predict(features)[0]
        pose_name = label_encoder.inverse_transform([pred])[0]

        score, feedback = posture_score(pose_name, normalized)

        st.subheader(f"🧠 Detected Pose: {pose_name}")
        st.subheader(f"📊 Posture Score: {score}%")

        if feedback:
            st.warning("⚠️ Corrections:")
            for tip in feedback:
                st.write("•", tip)
        else:
            st.success("✅ Excellent posture!")
