import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DETECTION_SCALE = 0.6
MAX_MISSED_FRAMES = 3

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 8, 9, 12
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

# A finger counts as extended when its tip is clearly farther from the wrist
# than its middle joint (PIP), and folded when it is not. The gap in between
# is a dead zone that keeps half-bent fingers from flickering between states.
EXTENDED_RATIO = 1.15
FOLDED_RATIO = 1.0

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class TrackerError(RuntimeError):
    pass


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise TrackerError(
                f"MediaPipe model not found at {model_path}\n"
                f"Download it from:\n  {MODEL_URL}\n"
                f"and save it as {model_path}"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except (RuntimeError, ValueError) as error:
            raise TrackerError(f"Could not initialise MediaPipe HandLandmarker: {error}") from error

        self.hands = []
        self._missed_frames = 0
        self._start_time = time.monotonic()
        self._last_timestamp = -1

    def update(self, frame):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        timestamp = max(int((time.monotonic() - self._start_time) * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        result = self._landmarker.detect_for_video(image, timestamp)

        if result.hand_landmarks:
            self.hands = [
                [(int(p.x * width), int(p.y * height)) for p in hand]
                for hand in result.hand_landmarks
            ]
            self._missed_frames = 0
        else:
            self._missed_frames += 1
            if self._missed_frames > MAX_MISSED_FRAMES:
                self.hands = []

    def close(self):
        self._landmarker.close()

    def all_hands(self):
        return self.hands

    def draw(self, frame):
        for hand in self.hands:
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, hand[start], hand[end], (255, 255, 255), 2)
            for point in hand:
                cv2.circle(frame, point, 4, (0, 255, 0), -1)

    def _main_hand(self):
        return self.hands[0] if self.hands else None

    @staticmethod
    def _palm_size(hand):
        return distance(hand[WRIST], hand[MIDDLE_MCP])

    @staticmethod
    def _tip_ratio(hand, tip_id):
        wrist = hand[WRIST]
        return distance(hand[tip_id], wrist) / max(distance(hand[tip_id - 2], wrist), 1e-6)

    def _extended(self, hand, finger):
        return self._tip_ratio(hand, FINGER_TIPS[finger]) > EXTENDED_RATIO

    def _folded(self, hand, finger):
        return self._tip_ratio(hand, FINGER_TIPS[finger]) < FOLDED_RATIO

    def index_tip(self):
        hand = self._main_hand()
        return hand[INDEX_TIP] if hand else None

    def pinch_ratio(self):
        # Thumb-to-index distance as a fraction of palm size, smallest over all hands,
        # so the value does not depend on how far the hand is from the camera.
        ratios = [
            distance(hand[THUMB_TIP], hand[INDEX_TIP]) / max(self._palm_size(hand), 1e-6)
            for hand in self.hands
        ]
        return min(ratios) if ratios else None

    def is_index_up(self):
        hand = self._main_hand()
        return bool(hand) and self._extended(hand, "index")

    def is_middle_up(self):
        hand = self._main_hand()
        return bool(hand) and self._extended(hand, "middle")

    def is_open_palm(self):
        hand = self._main_hand()
        return bool(hand) and all(self._extended(hand, finger) for finger in FINGER_TIPS)

    def is_fist(self):
        hand = self._main_hand()
        return bool(hand) and all(self._folded(hand, finger) for finger in FINGER_TIPS)

    def index_middle_spread(self):
        hand = self._main_hand()
        if not hand:
            return 0.0
        return distance(hand[INDEX_TIP], hand[MIDDLE_TIP]) / max(self._palm_size(hand), 1e-6)

    def two_hand_points(self):
        if len(self.hands) < 2:
            return []
        return [
            point
            for hand in self.hands[:2]
            for point in (hand[THUMB_TIP], hand[INDEX_TIP])
        ]
