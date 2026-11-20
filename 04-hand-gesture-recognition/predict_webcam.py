"""Live gesture prediction from a webcam (or a single image) using the trained SVM.

    python predict_webcam.py                 # webcam, press q to quit
    python predict_webcam.py --image hand.png

Note: the model was trained on near-infrared Leap Motion frames (dark background,
bright hand). It works best with a plain dark background and good lighting.
"""
import argparse
import sys
from pathlib import Path

import cv2
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import preprocess  # noqa: E402

bundle = joblib.load(Path(__file__).resolve().parent / "results/gesture_svm.joblib")
model, GESTURES = bundle["model"], bundle["gestures"]


def predict(gray):
    return GESTURES[int(model.predict(preprocess(gray)[None])[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    args = ap.parse_args()
    if args.image:
        print(predict(cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)))
        return
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(frame, predict(gray), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.imshow("gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
