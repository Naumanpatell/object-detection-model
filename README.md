# Cat Sticker Face Detector

A webcam app that detects a face, classifies its facial expression (happy, shocked, angry, sad, sleepy, neutral) and shows a matching cat meme sticker next to the face.

Built for the Computational Intelligence coursework task *"describe an approach to detecting objects in images"*.

## Setup

Requires Python 3.9–3.12 (MediaPipe 0.10.14 has no wheels for newer versions).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** On this machine, Windows blocks running executables from inside the OneDrive folder, so the venv lives at `%USERPROFILE%\.venvs\cat-sticker` instead (created with a `uv`-installed Python 3.12). Use that venv's `python.exe` as shown below.

Optionally, add cat images to `stickers/` (see [stickers/README.md](stickers/README.md)). Without them, coloured placeholder boxes are shown.

## Running

```powershell
# Webcam mode
python cat_sticker_faces.py
python cat_sticker_faces.py --camera 1          # if the default webcam is wrong

# Single-image mode (for report figures; no webcam needed)
python cat_sticker_faces.py --image photo.jpg                 # saves screenshots/photo_annotated.png
python cat_sticker_faces.py --image photo.jpg --out fig1.png
```

When the webcam starts, look at the camera with a **neutral face** for about 40 frames (around 2 seconds) while it calibrates.

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Recalibrate (hold a neutral face again) |
| `a` | Calibrate angry: frown hard for ~1.5 s to personalise the angry threshold |
| `d` | Toggle the debug overlay (feature values vs baseline) |
| `s` | Save a screenshot to `screenshots/` |

## Pipeline overview

The system is a classic **detect → describe → classify** pipeline. Each stage is a separate function in `cat_sticker_faces.py`.

1. **Detection (`detect_face`)**: Google MediaPipe Face Mesh, a lightweight CNN, locates a face and regresses 468 3D facial landmarks. It is limited to one face. A bounding box is derived from the minimum and maximum landmark coordinates, so the "object detection" output is both a box and a dense set of keypoints.
2. **Feature extraction (`extract_features`)**: Six geometric features are computed from landmark pairs. Each is divided by the face width (landmark 234 to 454), which makes it invariant to the distance from the camera:
   - `mouth_open`: inner-lip gap (13 to 14)
   - `mouth_width`: mouth-corner distance (61 to 291)
   - `smile_curve`: lip-centre y minus mouth-corner y (positive when the corners turn up)
   - `eye_open`: mean eyelid gap of both eyes (159 to 145, 386 to 374)
   - `brow`: mean brow-to-upper-eyelid distance (105 to 159, 334 to 386)
   - `brow_gap`: distance between the inner ends of the eyebrows (55 to 285), which shrinks when frowning
3. **Calibration (`Calibrator`)**: The features from the first 40 frames that contain a face are averaged into a personal neutral **baseline**. This compensates for differences in face shape between people, so the rules measure *change* in expression, not absolute geometry.
4. **Classification (`classify_expression`)**: A rule-based classifier compares each feature to the baseline. The rules are checked in priority order and the first match wins:
   | Order | Expression | Rule |
   |---|---|---|
   | 1 | sleepy  | `eye_open` < 0.55 × baseline |
   | 2 | happy   | `mouth_width` > 1.10 × baseline |
   | 3 | shocked | `mouth_open` rises > 0.08 **or** `brow` rises > 0.025 |
   | 4 | angry   | frown score > personal threshold (see below) |
   | 5 | sad     | `smile_curve` drops by more than 0.012 |
   | 6 | neutral | otherwise |

   All thresholds are in the `TH` dictionary at the top of the file.

   **Personalised angry threshold.** The *frown score* is the relative brow drop plus the relative brow squeeze, each as a fraction of the person's own neutral value: `(base_brow − brow)/base_brow + (base_gap − brow_gap)/base_gap`. During neutral calibration the standard deviation of this score (the person's landmark jitter) is measured, and the angry threshold is set to `max(0.08, 3 × jitter)`. Pressing `a` runs an optional frown calibration (about 30 frames): the threshold is then set halfway between the person's neutral score (0) and their mean frown score (but never below 2.5 × jitter), so it adapts to how strongly each person frowns.
5. **Temporal smoothing (`LabelSmoother`)**: A majority vote over the last 8 frame labels removes single-frame flicker caused by landmark jitter.
6. **Overlay (`draw_sticker`)**: The sticker for the predicted label is scaled to the face width, alpha-blended to the right of the face (or to the left near the frame edge), and clipped safely at the image borders.

In `--image` mode there is no calibration, so the hand-set `NEUTRAL_DEFAULTS` are used as the baseline. Results are therefore less reliable than in webcam mode.

## Limitations

- **Lighting**: Poor or uneven lighting reduces landmark accuracy, which adds noise to the features.
- **Head pose**: Features are 2D image distances, so turning or tilting the head changes them (e.g. a turned head narrows the face width and distorts the ratios). Large poses can cause false labels.
- **Single face**: Only one face is tracked (`max_num_faces=1`).
- **Hand-tuned thresholds**: The rules and thresholds were chosen by hand and tested on a handful of faces. They are not learned from data and are not validated on a labelled dataset, so accuracy is unmeasured and varies between people.
- **Coarse expression model**: Six classes based on a few distances cannot capture subtle or mixed expressions (e.g. a closed-mouth smile may read as neutral). The fixed rule order also means only one label can win.
- **Calibration assumption**: The user must actually hold a neutral face during calibration. Otherwise every later prediction is biased.

## Future work

- **Learned classifier**: Train a CNN on the **FER-2013** dataset (48×48 greyscale faces, 7 emotion classes) to replace the hand-written rules. Use the Face Mesh bounding box to crop and align the face before passing it to the network.
- Alternatively, train a small MLP or SVM on the landmark coordinates themselves (pose-normalised), which needs far less compute than an image CNN.
- Evaluate quantitatively (accuracy, confusion matrix) on a held-out labelled test set and compare the rules with the learned models.
- Use 3D landmarks or head-pose estimation to normalise for head rotation.
- Support multiple faces, with one sticker and one smoother per tracked face.
