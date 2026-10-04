# touchless-vision

A touchless computer-vision application that runs on a webcam and is controlled entirely by hand gestures. It has a gesture-driven menu and two modes: Ghost Mode, which makes you or part of the scene invisible, and Air Canvas, which lets you draw in the air.

Built with Python, OpenCV and the MediaPipe Tasks API.


## Features

- Gesture menu on the right side of the screen, selected by pinching.
- **Ghost Mode**
  - 5 second background calibration.
  - Rectangle invisibility: the box formed by both hands' thumb and index fingertips shows the stored background.
  - Full invisibility: a pinch fades you out into the background, and another pinch fades you back.
  - `r` recalibrates the background.
- **Air Canvas**
  - Draw with the index finger on a separate canvas layer.
  - Four-colour toolbar (red, green, blue, white), chosen by pinching.
  - Raise the middle finger as well to move without drawing.
  - Open palm turns the cursor into an eraser, sized by how far apart the index and middle fingers are.
- Return to the menu from either mode with open palm, then fist, twice.
- Mirrored camera view, hand skeleton overlay, clear error messages for a missing camera or model.

## Gesture controls

| Gesture | Action |
|---|---|
| Index finger | Move the cursor, draw |
| Pinch (thumb + index) | Select a menu button, pick a colour, toggle full invisibility |
| Index + middle finger up | Pause drawing |
| Open palm | Eraser (Air Canvas) |
| Open palm, then fist, twice quickly | Back to the menu |
| Both hands, thumb + index of each | Rectangle invisibility (Ghost Mode) |

| Key | Action |
|---|---|
| `q` or `Esc` | Quit |
| `r` | Recalibrate the background (Ghost Mode) |

## Architecture

```
Camera -> OpenCV frame -> mirror -> HandTracker (MediaPipe)
       -> HandState (gestures) -> current mode -> rendered frame -> display
```


## Technology stack

- Python 3.9 or newer (a version supported by your MediaPipe release)
- OpenCV (`opencv-python`)
- MediaPipe Tasks API (`HandLandmarker`)
- NumPy

## Project structure

```
touchless-vision/
├── main.py              entry point, main loop, gesture state, mode switching
├── camera_check.py      lists working camera indices
├── requirements.txt
├── models/              put hand_landmarker.task here
├── tracking/
│   ├── __init__.py      smoothing helper
│   └── hand_detector.py MediaPipe wrapper and hand queries
└── features/
    ├── __init__.py      state names
    ├── menu.py
    ├── invisibility.py
    └── air_draw.py
```

## Installation

```bash
git clone https://github.com/zillerehman-dev/touchless-vision.git
cd touchless-vision

python -m venv .venv
.venv\Scripts\activate        # macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### MediaPipe model setup

The hand model is not stored in the repository. Download it once and place it at `models/hand_landmarker.task`:

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

If the file is missing, the program stops with a message showing this link and the expected path.

### Camera setup

List the cameras OpenCV can open:

```bash
python camera_check.py
```

## Running

```bash
python main.py                  # camera 0
python main.py --camera 1       # another camera
python main.py --fps            # show an FPS counter
python main.py --fullscreen
```

Use good, even lighting in front of you, and keep your hand about 30 to 60 cm from the camera.

## Troubleshooting

- **Could not open camera**: run `python camera_check.py`, pick a working index, close other programs that use the webcam, and check your operating system's camera permissions.
- **Model not found**: download `hand_landmarker.task` into `models/` as described above.
- **Nothing responds**: if no skeleton appears on your hand, detection is failing. Improve the lighting and make sure your whole hand is in view.
- **Pinch triggers too easily or not at all**: adjust `PINCH_START` and `PINCH_RELEASE` in `main.py`.
- **Ghost Mode shows the wrong background**: press `r` and step out of the frame during the countdown. Keep the camera still afterwards.
- **Low frame rate**: lower `FRAME_SIZE` in `main.py` or `DETECTION_SCALE` in `tracking/hand_detector.py`.

