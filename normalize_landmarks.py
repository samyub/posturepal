import numpy as np

# MediaPipe landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def normalize_landmarks(landmarks):
    """
    Normalize MediaPipe pose landmarks.

    Steps:
    1. Center pose at hip midpoint (translation)
    2. Scale using torso length (shoulders to hips)

    Input:
        landmarks: np.ndarray of shape (33, 4)
                   [x, y, z, visibility]

    Output:
        np.ndarray of shape (33, 4)
    """

    if landmarks is None or landmarks.shape != (33, 4):
        raise ValueError("Expected landmarks of shape (33, 4)")

    normalized = landmarks.copy()

    # -----------------------------
    # Translation (center on hips)
    # -----------------------------
    hip_center = (
        normalized[LEFT_HIP][:2] + normalized[RIGHT_HIP][:2]
    ) / 2.0

    normalized[:, 0:2] -= hip_center

    # -----------------------------
    # Scale (torso length)
    # -----------------------------
    shoulder_center = (
        normalized[LEFT_SHOULDER][:2] + normalized[RIGHT_SHOULDER][:2]
    ) / 2.0

    torso_length = np.linalg.norm(shoulder_center)

    if torso_length < 1e-6:
        raise ValueError("Torso length too small for normalization")

    normalized[:, 0:3] /= torso_length

    return normalized
