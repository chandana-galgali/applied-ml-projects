"""Cats vs Dogs image classification with a Support Vector Machine.

Compares the naive approach (raw grayscale pixels -> PCA -> linear SVM) with
hand-crafted image features (HOG shape + HSV colour histograms) -> RBF-kernel SVM.

Dataset: Kaggle "Dogs vs. Cats" (train/ has 25,000 labelled images: cat.*.jpg, dog.*.jpg)
Run:     python train.py --data ../data/dogs-vs-cats
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.style import PALETTE, plt, results_dir, save, save_json  # noqa: E402

SEED = 42
SIZE = 64
HOG = cv2.HOGDescriptor((SIZE, SIZE), (16, 16), (8, 8), (8, 8), 9)


def features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = HOG.compute(gray).ravel()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    hist /= hist.sum() + 1e-6
    raw = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).ravel() / 255.0
    return hog.astype(np.float32), hist.astype(np.float32), raw.astype(np.float32)


def extract(paths, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["hog"], d["hist"], d["raw"]
    t = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        feats = list(ex.map(features, paths, chunksize=256))
    hog, hist, raw = (np.stack([f[i] for f in feats]) for i in range(3))
    np.savez_compressed(cache, hog=hog, hist=hist, raw=raw)
    print(f"extracted features for {len(paths)} images in {time.time()-t:.0f}s")
    return hog, hist, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/dogs-vs-cats"))
    args = ap.parse_args()
    out = results_dir(__file__)
    cache_dir = Path(__file__).resolve().parent / ".cache"; cache_dir.mkdir(exist_ok=True)

    train_dir = Path(args.data) / "train"
    files = sorted(f for f in os.listdir(train_dir) if f.endswith(".jpg"))
    y = np.array([0 if f.startswith("cat") else 1 for f in files])  # 0 = cat, 1 = dog
    paths = [str(train_dir / f) for f in files]
    hog, hist, raw = extract(paths, cache_dir / "train_features.npz")

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    results = {}

    # ---- 1) Baseline: raw pixels -> PCA -> linear SVM (what the original script did) ----
    base = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=SEED)),
                     ("svm", LinearSVC(C=0.01, max_iter=5000, random_state=SEED))])
    base.fit(raw[idx_tr], y[idx_tr])
    p = base.predict(raw[idx_te])
    results["Raw pixels + PCA + Linear SVM"] = {"accuracy": round(accuracy_score(y[idx_te], p), 4)}
    print("baseline", results)

    # ---- 2) HOG + colour histogram -> PCA -> RBF SVM, tuned on a subset ----
    X = np.hstack([hog, hist * 10])
    model = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=300, random_state=SEED)),
                      ("svm", SVC(kernel="rbf", random_state=SEED))])
    rng = np.random.default_rng(SEED)
    tune_idx = rng.choice(idx_tr, 4000, replace=False)
    grid = GridSearchCV(model, {"svm__C": [1, 3, 10], "svm__gamma": ["scale", 0.001]},
                        cv=3, n_jobs=-1, verbose=0)
    t = time.time(); grid.fit(X[tune_idx], y[tune_idx])
    print(f"grid search {time.time()-t:.0f}s best={grid.best_params_} cv={grid.best_score_:.4f}")

    model.set_params(**grid.best_params_)
    t = time.time(); model.fit(X[idx_tr], y[idx_tr]); fit_s = time.time() - t
    scores = model.decision_function(X[idx_te])
    p = (scores > 0).astype(int)
    acc = accuracy_score(y[idx_te], p)
    auc = roc_auc_score(y[idx_te], scores)
    cm = confusion_matrix(y[idx_te], p)
    report = classification_report(y[idx_te], p, target_names=["cat", "dog"], output_dict=True)
    print(classification_report(y[idx_te], p, target_names=["cat", "dog"]))
    results["HOG + colour histogram + RBF SVM"] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}

    # ---- unlabeled Kaggle test set -> predictions file ----
    test_dir = Path(args.data) / "test1"
    if test_dir.exists():
        tfiles = sorted((f for f in os.listdir(test_dir) if f.endswith(".jpg")), key=lambda s: int(s.split(".")[0]))
        th, thi, _ = extract([str(test_dir / f) for f in tfiles], cache_dir / "test_features.npz")
        tp = model.predict(np.hstack([th, thi * 10]))
        with open(out / "test_predictions.csv", "w") as fh:
            fh.write("id,label\n")
            fh.writelines(f"{f.split('.')[0]},{int(v)}\n" for f, v in zip(tfiles, tp))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["cat", "dog"], yticklabels=["cat", "dog"],
           xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (acc {acc:.1%})")
    save(fig, out / "confusion_matrix.png"); plt.close(fig)

    fpr, tpr, _ = roc_curve(y[idx_te], scores)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"RBF SVM (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax.legend()
    save(fig, out / "roc_curve.png"); plt.close(fig)

    # gallery of held-out predictions
    pick = rng.choice(len(idx_te), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for ax, k in zip(axes.ravel(), pick):
        i = idx_te[k]
        img = cv2.cvtColor(cv2.resize(cv2.imread(paths[i]), (128, 128)), cv2.COLOR_BGR2RGB)
        ok = p[k] == y[i]
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"pred: {'dog' if p[k] else 'cat'}", color=PALETTE[5] if ok else PALETTE[7], fontsize=10)
    fig.suptitle("Held-out predictions (green = correct, red = wrong)"); save(fig, out / "sample_predictions.png"); plt.close(fig)

    fpr_s = np.interp(np.linspace(0, 1, 51), fpr, tpr)
    save_json({
        "dataset": {"labelled_images": int(len(y)), "train": int(len(idx_tr)), "test": int(len(idx_te)),
                    "image_size": SIZE, "feature_dims": int(X.shape[1])},
        "best_params": {k: (v if isinstance(v, (int, float)) else str(v)) for k, v in grid.best_params_.items()},
        "fit_seconds": round(fit_s, 1),
        "models": results,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: {m: round(report[c][m], 4) for m in ("precision", "recall", "f1-score")} for c in ("cat", "dog")},
        "roc": {"fpr": [round(v, 3) for v in np.linspace(0, 1, 51)], "tpr": [round(float(v), 4) for v in fpr_s]},
    }, out / "metrics.json")


if __name__ == "__main__":
    main()
