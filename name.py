import cv2
print(cv2.__file__)
print(cv2.__version__)

import os
import numpy as np
import mediapipe as mp

# ===============================
# CONFIG
# ===============================
DATASET_ROOT = r"C:\PosturePal\archive"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# ===============================
# MEDIAPIPE INITIALIZATION
# ===============================
mp_pose = mp.solutions.pose

pose_model = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# ===============================
# LANDMARK NORMALIZATION CONSTANTS
# ===============================
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# ===============================
# NORMALIZATION FUNCTION
# ===============================
def normalize_landmarks(landmarks):
    """
    Normalize MediaPipe landmarks:
    - Center on hip midpoint
    - Scale by torso length (hips -> shoulders)
    """

    if landmarks.shape != (33, 4):
        raise ValueError("Invalid landmark shape")

    normalized = landmarks.copy()

    # --- Translation (center on hips) ---
    hip_center = (
        normalized[LEFT_HIP][:2] + normalized[RIGHT_HIP][:2]
    ) / 2.0

    normalized[:, 0:2] -= hip_center

    # --- Scale (torso length) ---
    shoulder_center = (
        normalized[LEFT_SHOULDER][:2] + normalized[RIGHT_SHOULDER][:2]
    ) / 2.0

    torso_length = np.linalg.norm(shoulder_center)

    if torso_length < 1e-6:
        raise ValueError("Torso too small")

    normalized[:, 0:3] /= torso_length

    return normalized


# ===============================
# LANDMARK EXTRACTION
# ===============================
def extract_landmarks(image_bgr):
    """
    Runs MediaPipe Pose on a BGR image.
    Returns (33,4) landmark array or None.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if not results.pose_landmarks:
        return None

    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
        dtype=np.float32
    )


# ===============================
# MAIN DATASET PIPELINE
# ===============================
def build_dataset():
    dataset = []
    total_images = 0
    successful = 0

    print("\n📂 Scanning dataset...")

    for asana_name in os.listdir(DATASET_ROOT):
        asana_path = os.path.join(DATASET_ROOT, asana_name)

        if not os.path.isdir(asana_path):
            continue

        print(f"\n🧘 Processing pose: {asana_name}")

        for filename in os.listdir(asana_path):
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            total_images += 1
            img_path = os.path.join(asana_path, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"⚠️ Could not read: {img_path}")
                continue

            landmarks = extract_landmarks(image)

            if landmarks is None:
                print(f"❌ No pose detected: {img_path}")
                continue

            try:
                normalized = normalize_landmarks(landmarks)
            except ValueError:
                print(f"❌ Normalization failed: {img_path}")
                continue

            dataset.append({
                "label": asana_name,
                "image_path": img_path,
                "landmarks": normalized
            })

            successful += 1

    print("\n✅ Dataset build complete")
    print(f"📸 Total images scanned: {total_images}")
    print(f"💪 Successful poses: {successful}")
    print(f"🚫 Failed images: {total_images - successful}")

    return dataset


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    data = build_dataset()
    print(f"\n🎯 Final dataset size: {len(data)} samples")

