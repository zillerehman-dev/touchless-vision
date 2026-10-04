import time

import cv2

from tracking import smooth

CALIBRATION_SECONDS = 5
BOX_KEEP = 0.5
BOX_GRACE_FRAMES = 3
FADE_STEP = 0.05


class GhostMode:
    def __init__(self):
        self.reset()

    def reset(self):
        self.background = None
        self.calibration_start = None
        self.box = None
        self.frames_without_box = 0
        self.fade = 0.0
        self.fade_target = 0.0

    def update(self, frame, hand):
        if self.background is None:
            return self._calibrate(frame)

        self._track_box(frame.shape, hand.corners)
        if self.box:
            x1, y1, x2, y2 = (int(v) for v in self.box)
            if x2 > x1 and y2 > y1:
                frame[y1:y2, x1:x2] = self.background[y1:y2, x1:x2]

        if hand.just_pinched:
            self.fade_target = 1.0 - self.fade_target
        if self.fade < self.fade_target:
            self.fade = min(self.fade_target, self.fade + FADE_STEP)
        elif self.fade > self.fade_target:
            self.fade = max(self.fade_target, self.fade - FADE_STEP)
        if self.fade > 0:
            frame = cv2.addWeighted(self.background, self.fade, frame, 1 - self.fade, 0)

        if self.box:
            x1, y1, x2, y2 = (int(v) for v in self.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
        return frame

    def _calibrate(self, frame):
        if self.calibration_start is None:
            self.calibration_start = time.monotonic()
        remaining = CALIBRATION_SECONDS - (time.monotonic() - self.calibration_start)
        if remaining > 0:
            cv2.putText(frame, "Step out of frame!", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Capturing background in {int(remaining) + 1}...", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            self.background = frame.copy()
        return frame

    def _track_box(self, shape, corners):
        height, width = shape[:2]
        if len(corners) == 4:
            xs = [x for x, _ in corners]
            ys = [y for _, y in corners]
            target = (max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys)))
            self.box = smooth(self.box, target, BOX_KEEP)
            self.frames_without_box = 0
        else:
            self.frames_without_box += 1
            if self.frames_without_box > BOX_GRACE_FRAMES:
                self.box = None
