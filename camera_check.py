import sys

import cv2

MAX_INDEX = 5


def probe(index):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    try:
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        return None if not ok else frame.shape[1::-1]
    finally:
        capture.release()


def main():
    working = []
    for index in range(MAX_INDEX + 1):
        size = probe(index)
        if size:
            working.append(index)
            print(f"camera {index}: working, {size[0]}x{size[1]}")
        else:
            print(f"camera {index}: not available")
    if not working:
        print("No camera found. Close other apps using the webcam and check OS camera permissions.")
        return 1
    print(f"Run with: python main.py --camera {working[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
