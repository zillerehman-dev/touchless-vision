import cv2
import numpy as np

from tracking import smooth

COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255))
BRUSH_THICKNESS = 6

TOOLBAR_HEIGHT = 70
SWATCH_RADIUS = 22
SWATCH_SPACING = 70
SWATCH_START_X = 60
TOOLBAR_RIGHT = SWATCH_START_X + len(COLORS) * SWATCH_SPACING

ERASER_MIN_SIZE = 30
ERASER_MAX_SIZE = 160
SPREAD_MIN = 0.2
SPREAD_MAX = 0.8
ERASER_KEEP = 0.7


class CanvasMode:
    def __init__(self):
        self.canvas = None
        self.ink = None
        self.reset()

    def reset(self):
        self.canvas = None
        self.ink = None
        self.color_index = 0
        self.last_draw_point = None
        self.last_erase_point = None
        self.eraser_size = ERASER_MIN_SIZE

    def update(self, frame, hand):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)
            self.ink = np.zeros(frame.shape[:2], dtype=np.uint8)

        cursor = tuple(int(v) for v in hand.cursor) if hand.cursor else None
        in_toolbar = cursor is not None and cursor[0] < TOOLBAR_RIGHT and cursor[1] < TOOLBAR_HEIGHT
        erasing = cursor is not None and not in_toolbar and hand.open_palm
        drawing = cursor is not None and not in_toolbar and not erasing \
            and hand.index_up and not hand.middle_up

        if hand.just_pinched:
            press_point = hand.cursor_before or hand.cursor
            if press_point:
                self._pick_color(press_point)

        if erasing:
            self._erase(cursor, hand.spread)
        else:
            self.last_erase_point = None

        if drawing:
            self._draw(cursor)
        else:
            self.last_draw_point = None

        frame = cv2.copyTo(self.canvas, self.ink, frame)
        self._draw_toolbar(frame)
        self._draw_cursor(frame, cursor, erasing, paused=hand.middle_up)
        return frame

    def _stroke(self, start, end, color, ink_value, thickness):
        cv2.line(self.canvas, start, end, color, thickness)
        cv2.line(self.ink, start, end, ink_value, thickness)
        cv2.circle(self.canvas, end, thickness // 2, color, -1)
        cv2.circle(self.ink, end, thickness // 2, ink_value, -1)

    def _draw(self, point):
        start = self.last_draw_point or point
        self._stroke(start, point, COLORS[self.color_index], 255, BRUSH_THICKNESS)
        self.last_draw_point = point

    def _erase(self, point, spread):
        t = min(max((spread - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN), 0.0), 1.0)
        target = ERASER_MIN_SIZE + t * (ERASER_MAX_SIZE - ERASER_MIN_SIZE)
        self.eraser_size = smooth(self.eraser_size, target, ERASER_KEEP)
        start = self.last_erase_point or point
        self._stroke(start, point, (0, 0, 0), 0, int(self.eraser_size))
        self.last_erase_point = point

    def _pick_color(self, cursor):
        for i in range(len(COLORS)):
            center = self._swatch_center(i)
            if (cursor[0] - center[0]) ** 2 + (cursor[1] - center[1]) ** 2 <= SWATCH_RADIUS ** 2:
                self.color_index = i

    @staticmethod
    def _swatch_center(index):
        return SWATCH_START_X + index * SWATCH_SPACING, TOOLBAR_HEIGHT // 2

    def _draw_toolbar(self, frame):
        region = frame[:TOOLBAR_HEIGHT, :TOOLBAR_RIGHT + 10]
        panel = region.copy()
        middle_y = TOOLBAR_HEIGHT // 2
        radius = middle_y - 8
        cv2.line(panel, (10 + radius, middle_y), (TOOLBAR_RIGHT - radius, middle_y), (30, 30, 30), 2 * radius)
        frame[:TOOLBAR_HEIGHT, :TOOLBAR_RIGHT + 10] = cv2.addWeighted(panel, 0.8, region, 0.2, 0)

        for i, color in enumerate(COLORS):
            center = self._swatch_center(i)
            cv2.circle(frame, center, SWATCH_RADIUS, color, -1)
            if i == self.color_index:
                cv2.circle(frame, center, SWATCH_RADIUS + 4, (255, 255, 255), 2)

    def _draw_cursor(self, frame, cursor, erasing, paused):
        if cursor is None:
            return
        if erasing:
            cv2.circle(frame, cursor, int(self.eraser_size) // 2, (200, 200, 200), 2)
        elif cursor[1] >= TOOLBAR_HEIGHT or cursor[0] >= TOOLBAR_RIGHT:
            color = (150, 150, 150) if paused else COLORS[self.color_index]
            cv2.circle(frame, cursor, 8, color, -1)
