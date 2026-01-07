POSTUREPAL – README.txt
======================

PosturePal is a Yoga posture detection and correction system built using
MediaPipe, OpenCV, and Machine Learning. The application detects a yoga
asana from an uploaded image, evaluates posture correctness, and provides
feedback through a Streamlit web interface.

--------------------------------------------------
FILES IN THIS REPOSITORY
--------------------------------------------------

app.py
    Streamlit web application. Users upload an image and receive:
    - Detected yoga pose
    - Posture correctness score
    - Feedback for correction

mediapipe_pose.py
    Wrapper around MediaPipe Pose for extracting 33 body landmarks.

normalize_landmarks.py
    Normalizes pose landmarks using translation and scale invariance.

features.py
    Converts normalized landmarks into numerical feature vectors
    (angles + coordinates) for ML inference.

name.py
    Dataset processing pipeline that:
    - Iterates through dataset folders
    - Extracts MediaPipe landmarks
    - Normalizes landmarks
    - Builds a usable dataset

pose_classifier.joblib
    Trained machine learning model for pose classification.

label_encoder.joblib
    Label encoder used to convert predicted class indices back to pose names.

--------------------------------------------------
PYTHON & PACKAGE VERSIONS USED
--------------------------------------------------

Python Version:
    Python 3.11.x

Installed Packages (exact versions):

    numpy==1.26.4
    opencv-python==4.9.0.80
    mediapipe==0.10.14        (installed with --no-deps)
    scikit-learn==1.4.2
    streamlit==1.32.2
    matplotlib==3.8.4
    joblib==1.3.2

IMPORTANT:
    MediaPipe MUST be installed with --no-deps to avoid conflicts
    with NumPy 2.x and JAX.

--------------------------------------------------
STEP-BY-STEP SETUP (ANACONDA PROMPT)
--------------------------------------------------

1. Install Anaconda (if not already installed)
   Download from:
   https://www.anaconda.com/products/distribution

2. Open Anaconda Prompt

3. Create a new environment with Python 3.11

   conda create -n posturepal python=3.11 -y
   conda activate posturepal

4. Upgrade pip

   python -m pip install --upgrade pip

5. Install required packages (EXACT versions)

   pip install numpy==1.26.4
   pip install opencv-python==4.9.0.80
   pip install scikit-learn==1.4.2
   pip install streamlit==1.32.2
   pip install matplotlib==3.8.4
   pip install joblib==1.3.2

6. Install MediaPipe safely

   pip install mediapipe==0.10.14 --no-deps

7. Verify installation

   python

   >>> import cv2, numpy, mediapipe
   >>> print(cv2.__version__)
   >>> print(numpy.__version__)
   >>> print(mediapipe.__version__)

--------------------------------------------------
DATASET STRUCTURE (FOR name.py)
--------------------------------------------------

The dataset should be organized as:

C:\PosturePal\archive\
    ├── Adho Mukha Svanasana\
    ├── Virabhadrasana Two\
    ├── Vrksasana\
    └── ...

Each folder name represents the yoga pose label
and contains images of that pose.

--------------------------------------------------
RUNNING THE STREAMLIT APPLICATION
--------------------------------------------------

1. Activate the environment

   conda activate posturepal

2. Navigate to the project directory

   cd C:\PosturePal

3. Start the Streamlit app

   streamlit run app.py

4. A browser window will open automatically.
   Upload a yoga image to see:
   - Predicted pose
   - Posture score
   - Correction feedback

--------------------------------------------------
IMPORTANT NOTES
--------------------------------------------------

- Ensure the following files exist in the project root:
      app.py
      mediapipe_pose.py
      normalize_landmarks.py
      features.py
      pose_classifier.joblib
      label_encoder.joblib

- Do NOT name any file "cv2.py" or create a folder named "cv2".
  This will break OpenCV imports.

- If Streamlit throws module import errors, ensure:
  - All files are in the same directory
  - Streamlit is restarted after adding new files

--------------------------------------------------
PROJECT STATUS
--------------------------------------------------

This project successfully:
- Detects yoga poses using ML
- Evaluates posture correctness
- Provides user-friendly feedback via a web app

--------------------------------------------------
END OF README
--------------------------------------------------
