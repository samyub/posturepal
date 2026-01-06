import cv2
import numpy as np
import mediapipe as mp


class MediaPipePoseExtractor:
    """
    MediaPipe Pose wrapper for landmark extraction and visualization.
    """

    def __init__(
        self,
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence
        )

    def extract_landmarks(self, image_bgr):
        """
        Extract 33 pose landmarks from a BGR image.

        Returns:
            np.ndarray of shape (33, 4)
            [x, y, z, visibility]
        or:
            None if no pose detected
        """

        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )

        return landmarks

    def draw_landmarks(self, image_bgr):
        """
        Draw pose landmarks on the image (for debugging/visualization).
        """

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return image_bgr

        annotated = image_bgr.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

        return annotated
