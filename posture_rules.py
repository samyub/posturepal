import numpy as np

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
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def posture_score(pose_name, landmarks):
    """
    Returns:
      score (0–100)
      feedback (list of strings)
    """

    xy = landmarks[:, :2]
    score = 100
    feedback = []

    # ---------------------------
    # WARRIOR TWO
    # ---------------------------
    if pose_name == "Virabhadrasana Two":
        front_knee_angle = angle(
            xy[LEFT_HIP], xy[LEFT_KNEE], xy[LEFT_ANKLE]
        )

        if front_knee_angle > 110:
            score -= 15
            feedback.append("Bend your front knee more (aim for ~90°).")

        shoulder_diff = abs(xy[LEFT_SHOULDER][1] - xy[RIGHT_SHOULDER][1])
        if shoulder_diff > 0.1:
            score -= 10
            feedback.append("Keep shoulders level.")

    # ---------------------------
    # TREE
    # ---------------------------
    elif pose_name == "Vrksasana":
        hip_diff = abs(xy[LEFT_HIP][0] - xy[RIGHT_HIP][0])
        if hip_diff > 0.15:
            score -= 15
            feedback.append("Keep hips aligned.")

    # ---------------------------
    # PLANK
    # ---------------------------
    elif pose_name == "Phalakasana":
        shoulder_hip = angle(
            xy[LEFT_SHOULDER], xy[LEFT_HIP], xy[LEFT_ANKLE]
        )
        if shoulder_hip < 160:
            score -= 20
            feedback.append("Keep your body in a straight line.")

    # ---------------------------
    # CHAIR
    # ---------------------------
    elif pose_name == "Utkatasana":
        knee_angle = angle(
            xy[LEFT_HIP], xy[LEFT_KNEE], xy[LEFT_ANKLE]
        )
        if knee_angle > 120:
            score -= 15
            feedback.append("Sit deeper into the pose.")

    # ---------------------------
    # DOWNWARD DOG
    # ---------------------------
    elif pose_name == "Adho Mukha Svanasana":
        back_angle = angle(
            xy[LEFT_WRIST], xy[LEFT_SHOULDER], xy[LEFT_HIP]
        )
        if back_angle < 150:
            score -= 15
            feedback.append("Lengthen your spine.")

    return max(score, 0), feedback
