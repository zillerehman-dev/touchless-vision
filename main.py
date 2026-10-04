import argparse
import sys
import time
import cv2
from features import CANVAS, GHOST, MENU
from features.air_draw import CanvasMode
from features.invisibility import GhostMode
from features.menu import MenuMode
from tracking import smooth
from tracking.hand_detector import HandTracker, TrackerError

WINDOW_NAME = "Touchless Vision"
FRAME_SIZE = (1280, 720)
MAX_READ_FAILURES = 30

CURSOR_KEEP = 0.5

# Hysteresis: a pinch starts below the first value and only ends above the
# second, so a hand hovering near the threshold cannot flicker on and off.
PINCH_START = 0.35
PINCH_RELEASE = 0.50

RETURN_WINDOW_SECONDS = 1.2


class CameraError(RuntimeError):
    pass


class HandState:
    def __init__(self):
        self.cursor = None
        self.cursor_before = None
        self.pinching = False
        self.just_pinched = False
        self.open_palm = False
        self.fist = False
        self.index_up = False
        self.middle_up = False
        self.spread = 0.0
        self.corners = []
        self.back_to_menu = False
        self._palm_seen = False
        self._fist_times = []

    def update(self, tracker):
        tip = tracker.index_tip()
        self.cursor_before = self.cursor
        self.cursor = smooth(self.cursor, tip, CURSOR_KEEP) if tip else None

        ratio = tracker.pinch_ratio()
        was_pinching = self.pinching
        if ratio is None:
            self.pinching = False
        elif self.pinching:
            self.pinching = ratio < PINCH_RELEASE
        else:
            self.pinching = ratio < PINCH_START
        self.just_pinched = self.pinching and not was_pinching

        self.open_palm = tracker.is_open_palm()
        self.fist = tracker.is_fist()
        self.index_up = tracker.is_index_up()
        self.middle_up = tracker.is_middle_up()
        self.spread = tracker.index_middle_spread()
        self.corners = tracker.two_hand_points()
        self.back_to_menu = self._detect_palm_fist_twice()

    def _detect_palm_fist_twice(self):
        if self.open_palm:
            self._palm_seen = True
            return False
        if not (self.fist and self._palm_seen):
            return False
        self._palm_seen = False
        now = time.monotonic()
        self._fist_times = [t for t in self._fist_times if now - t < RETURN_WINDOW_SECONDS] + [now]
        if len(self._fist_times) < 2:
            return False
        self._fist_times.clear()
        return True


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            return capture
        capture.release()
    raise CameraError(
        f"Could not open camera {index}. Run `python camera_check.py` to list working "
        "cameras, close other apps that use the webcam, and check camera permissions."
    )


def read_frame(capture, failures):
    ok, frame = capture.read()
    if ok and frame is not None:
        return cv2.flip(frame, 1), 0
    failures += 1
    if failures >= MAX_READ_FAILURES:
        raise CameraError("The camera stopped delivering frames.")
    time.sleep(0.03)
    return None, failures


def run(capture, tracker, show_fps, fullscreen):
    modes = {MENU: MenuMode(), GHOST: GhostMode(), CANVAS: CanvasMode()}
    state = MENU
    hand = HandState()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def enter(new_state):
        modes[new_state].reset()
        return new_state

    failures = 0
    fps = None
    last_time = time.perf_counter()
    reported_size = False

    while True:
        frame, failures = read_frame(capture, failures)
        if frame is None:
            continue
        if not reported_size:
            height, width = frame.shape[:2]
            if (width, height) != FRAME_SIZE:
                print(f"Camera does not support {FRAME_SIZE[0]}x{FRAME_SIZE[1]}; using {width}x{height}.")
            reported_size = True

        tracker.update(frame)
        hand.update(tracker)

        if hand.back_to_menu and state != MENU:
            state = enter(MENU)

        frame = modes[state].update(frame, hand)
        if state == MENU and modes[MENU].choice:
            state = enter(modes[MENU].choice)

        tracker.draw(frame)

        if show_fps:
            now = time.perf_counter()
            fps = smooth(fps, 1.0 / max(now - last_time, 1e-6), 0.9)
            last_time = now
            cv2.putText(frame, f"{fps:.0f} FPS", (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and state == GHOST:
            modes[GHOST].reset()
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break


def main():
    parser = argparse.ArgumentParser(description="Gesture Vision")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--fps", action="store_true", help="show FPS counter")
    parser.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    args = parser.parse_args()

    try:
        tracker = HandTracker()
    except TrackerError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        capture = open_camera(args.camera)
    except CameraError as error:
        print(f"Error: {error}", file=sys.stderr)
        tracker.close()
        return 1

    try:
        run(capture, tracker, args.fps, args.fullscreen)
    except CameraError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
