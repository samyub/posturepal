import numpy as np

# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def angle(a, b, c):
    """
    Compute angle (degrees) at point b given points a-b-c
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extract_features(landmarks):
    """
    Input: normalized landmarks (33,4)
    Output: 1D feature vector
    """

    xy = landmarks[:, :2]

    features = []

    # --- Arm angles ---
    features.append(angle(xy[LEFT_SHOULDER], xy[LEFT_ELBOW], xy[LEFT_WRIST]))
    features.append(angle(xy[RIGHT_SHOULDER], xy[RIGHT_ELBOW], xy[RIGHT_WRIST]))

    # --- Leg angles ---
    features.append(angle(xy[LEFT_HIP], xy[LEFT_KNEE], xy[LEFT_ANKLE]))
    features.append(angle(xy[RIGHT_HIP], xy[RIGHT_KNEE], xy[RIGHT_ANKLE]))

    # --- Hip angles ---
    features.append(angle(xy[LEFT_SHOULDER], xy[LEFT_HIP], xy[LEFT_KNEE]))
    features.append(angle(xy[RIGHT_SHOULDER], xy[RIGHT_HIP], xy[RIGHT_KNEE]))

    # --- Torso lean ---
    shoulder_center = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    hip_center = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2
    torso_vector = shoulder_center - hip_center
    features.extend(torso_vector.tolist())

    # --- Optional: flattened coordinates ---
    features.extend(xy.flatten().tolist())

    return np.array(features, dtype=np.float32)
