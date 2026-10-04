import cv2
import numpy as np

from features import CANVAS, GHOST

SIDEBAR_WIDTH = 260
SIDEBAR_PADDING = 20
BUTTON_TOP = 130
BUTTON_HEIGHT = 80
BUTTON_GAP = 20
BUTTONS = (("Ghost Mode", GHOST), ("Air Canvas", CANVAS))

PANEL_COLOR = 30
BUTTON_COLOR = (60, 60, 60)
HOVER_COLOR = (0, 165, 255)
CURSOR_IDLE = (255, 0, 0)
CURSOR_PINCH = (0, 0, 255)


class MenuMode:
    def __init__(self):
        self.choice = None
        self._panel = None

    def reset(self):
        self.choice = None

    def button_rects(self, frame_width):
        left = frame_width - SIDEBAR_WIDTH + SIDEBAR_PADDING
        right = frame_width - SIDEBAR_PADDING
        rects = []
        for i, (label, state) in enumerate(BUTTONS):
            top = BUTTON_TOP + i * (BUTTON_HEIGHT + BUTTON_GAP)
            rects.append((label, state, (left, top, right, top + BUTTON_HEIGHT)))
        return rects

    def update(self, frame, hand):
        height, width = frame.shape[:2]
        self._draw_sidebar(frame, width, height)

        cursor = tuple(int(v) for v in hand.cursor) if hand.cursor else None
        rects = self.button_rects(width)
        hovered = self._button_at(rects, cursor)
        for label, state, rect in rects:
            self._draw_button(frame, label, rect, state == hovered)

        # A pinch moves the fingertip, so select what was under it just before the pinch began.
        press_point = hand.cursor_before or hand.cursor
        press_point = tuple(int(v) for v in press_point) if press_point else None
        self.choice = self._button_at(rects, press_point) if hand.just_pinched else None

        if cursor:
            cv2.circle(frame, cursor, 9, CURSOR_PINCH if hand.pinching else CURSOR_IDLE, -1)
            cv2.circle(frame, cursor, 11, (255, 255, 255), 1)
        return frame

    @staticmethod
    def _button_at(rects, point):
        if point is None:
            return None
        for _, state, (x1, y1, x2, y2) in rects:
            if x1 < point[0] < x2 and y1 < point[1] < y2:
                return state
        return None

    def _draw_sidebar(self, frame, width, height):
        left = width - SIDEBAR_WIDTH
        region = frame[:, left:]
        if self._panel is None or self._panel.shape != region.shape:
            self._panel = np.full(region.shape, PANEL_COLOR, dtype=np.uint8)
        frame[:, left:] = cv2.addWeighted(self._panel, 0.75, region, 0.25, 0)
        cv2.putText(frame, "MENU", (left + SIDEBAR_PADDING, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "pinch to select", (left + SIDEBAR_PADDING, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    @staticmethod
    def _draw_button(frame, label, rect, hovered):
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), HOVER_COLOR if hovered else BUTTON_COLOR, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, label, (x1 + (x2 - x1 - text_w) // 2, y1 + (y2 - y1 + text_h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
