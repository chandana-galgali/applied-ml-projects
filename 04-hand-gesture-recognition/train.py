"""Hand gesture recognition (10 gestures) from near-infrared images.

Pipeline: grayscale -> crop to the hand (Otsu threshold) -> 64x64 -> HOG -> PCA -> RBF SVM.
Evaluated two ways:
  * random split        - same people appear in train and test (optimistic)
  * leave-people-out    - test subjects are never seen in training (honest:
                          this is what a real gesture-control app faces)

Dataset: Kaggle "Hand Gesture Recognition Database" (leapGestRecog), 20,000 images,
         10 subjects x 10 gestures x 200 frames.
Run:     python train.py --data ../data/leapgestrecog
         python predict_webcam.py   # live demo with the saved model
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.style import PALETTE, plt, results_dir, save, save_json  # noqa: E402

SEED = 42
W, H = 64, 64
HOG = cv2.HOGDescriptor((W, H), (16, 16), (8, 8), (8, 8), 9)
FULL_W, FULL_H = 128, 48  # full-frame baseline keeps the 640x240 aspect ratio
HOG_FULL = cv2.HOGDescriptor((FULL_W, FULL_H), (16, 16), (8, 8), (8, 8), 9)
GESTURES = ["palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c", "down"]


def crop_hand(gray, pad=10):
    """Square crop around the bright hand region so position/scale no longer matter."""
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return gray
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)
    c = gray[y0:y1, x0:x1]
    side = max(c.shape)
    sq = np.zeros((side, side), np.uint8)
    oy, ox = (side - c.shape[0]) // 2, (side - c.shape[1]) // 2
    sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return sq


def preprocess(gray):
    hand = cv2.resize(crop_hand(gray), (W, H), interpolation=cv2.INTER_AREA)
    return HOG.compute(cv2.equalizeHist(hand)).ravel().astype(np.float32)


def preprocess_full(gray):
    full = cv2.resize(gray, (FULL_W, FULL_H), interpolation=cv2.INTER_AREA)
    return HOG_FULL.compute(cv2.equalizeHist(full)).ravel().astype(np.float32)


def load_one(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return preprocess(g), preprocess_full(g)


def scan(root: Path):
    paths, labels, subjects = [], [], []
    for subj in sorted(os.listdir(root)):
        sdir = root / subj
        if not sdir.is_dir() or not subj.isdigit():
            continue
        for gdir in sorted(os.listdir(sdir)):
            g = int(gdir.split("_")[0]) - 1  # "01_palm" -> 0
            for f in sorted(os.listdir(sdir / gdir)):
                if f.endswith(".png"):
                    paths.append(str(sdir / gdir / f)); labels.append(g); subjects.append(int(subj))
    return paths, np.array(labels), np.array(subjects)


def make_model(C=10):
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=150, random_state=SEED)),
                     ("svm", SVC(kernel="rbf", C=C, gamma="scale"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/leapgestrecog"))
    args = ap.parse_args()
    out = results_dir(__file__)
    cache = Path(__file__).resolve().parent / ".cache"; cache.mkdir(exist_ok=True)

    paths, y, groups = scan(Path(args.data))
    print(f"{len(paths)} images, {len(set(groups))} subjects, {len(set(y))} gestures")
    cf = cache / "hog_v2.npz"
    if cf.exists():
        d = np.load(cf); X, X_full = d["crop"], d["full"]
    else:
        t = time.time()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            feats = list(ex.map(load_one, paths, chunksize=256))
        X = np.stack([f[0] for f in feats]); X_full = np.stack([f[1] for f in feats])
        np.savez(cf, crop=X, full=X_full); print(f"HOG features {X.shape} in {time.time()-t:.0f}s")
    gkf = GroupKFold(n_splits=5)

    # ---- ablation: same model on the full frame (no hand crop) ----
    yp_full = np.zeros_like(y)
    for tr, te in gkf.split(X_full, y, groups):
        yp_full[te] = make_model().fit(X_full[tr], y[tr]).predict(X_full[te])
    full_acc = accuracy_score(y, yp_full)
    print(f"full-frame leave-people-out accuracy: {full_acc:.4f}")

    # ---- random split (optimistic) ----
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    m = make_model().fit(X[tr], y[tr])
    rand_acc = accuracy_score(y[te], m.predict(X[te]))
    print(f"random split accuracy: {rand_acc:.4f}")

    # ---- leave-people-out: 5 folds, 2 unseen subjects each ----
    y_pred = np.zeros_like(y)
    fold_acc = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = make_model().fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        fold_acc.append(accuracy_score(y[te], y_pred[te]))
        print(f"fold {k}: unseen subjects {sorted(set(groups[te]))} acc={fold_acc[-1]:.4f}")
    lpo_acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)
    rep = classification_report(y, y_pred, target_names=GESTURES, output_dict=True)
    print(classification_report(y, y_pred, target_names=GESTURES))

    # final model on everything, for the webcam demo
    final = make_model().fit(X, y)
    joblib.dump({"model": final, "gestures": GESTURES, "size": (W, H)}, out / "gesture_svm.joblib", compress=3)

    # ---- figures ----
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for (i, j), v in np.ndenumerate(cmn):
        if v >= 0.01:
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "black")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=GESTURES, yticklabels=GESTURES,
           xlabel="Predicted", ylabel="Actual",
           title=f"Leave-people-out confusion matrix (%), acc {lpo_acc:.1%}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, out / "confusion_matrix.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(12, 2.6))
    for g, ax in enumerate(axes.ravel()):
        i = np.where((y == g) & (groups == 3))[0][100]
        ax.imshow(cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE), cmap="gray"); ax.axis("off")
        ax.set_title(f"{GESTURES[g]}  ({rep[GESTURES[g]]['f1-score']*100:.0f}% F1)", fontsize=9)
    fig.suptitle("The 10 gestures, with per-class F1 on unseen people"); save(fig, out / "gestures.png"); plt.close(fig)

    save_json({
        "dataset": {"images": int(len(y)), "subjects": int(len(set(groups))), "classes": GESTURES,
                    "input": f"{W}x{H} grayscale", "feature_dims": int(X.shape[1])},
        "models": {"Full frame, HOG + RBF SVM (baseline)": {
            "leave_people_out_accuracy": round(full_acc, 4)},
            "Hand crop, HOG + PCA + RBF SVM": {
            "random_split_accuracy": round(rand_acc, 4),
            "leave_people_out_accuracy": round(lpo_acc, 4),
            "leave_people_out_macro_f1": round(macro_f1, 4),
            "fold_accuracies": [round(a, 4) for a in fold_acc]}},
        "per_class_f1": {g: round(rep[g]["f1-score"], 4) for g in GESTURES},
        "confusion_matrix_pct": [[round(v * 100, 1) for v in row] for row in cmn],
    }, out / "metrics.json")


if __name__ == "__main__":
    main()
