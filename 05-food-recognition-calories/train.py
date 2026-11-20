"""Food recognition + calorie estimation, stage 1: is there food in this image?

Pipeline
  1. Food detector (this script): HOG shape + HSV colour + LBP texture features
     -> RBF SVM, trained on Food-5K (food vs non-food). Runs on any CPU.
  2. Dish classifier: Food-101 transfer-learning CNN (food101_cnn.ipynb, run on a GPU).
  3. Calorie lookup (calories.py): dish name -> kcal per serving / per 100 g.

Dataset: Kaggle "Food-5K" (training 3000 / validation 1000 / evaluation 1000;
         file prefix 1_ = food, 0_ = non-food)
Run:     python train.py --data ../data/food-5k
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
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calories import load_table, lookup  # noqa: E402
from common.style import PALETTE, plt, results_dir, save, save_json  # noqa: E402
from food101_classes import FOOD101  # noqa: E402

SEED = 42
SIZE = 96
HOG = cv2.HOGDescriptor((SIZE, SIZE), (16, 16), (8, 8), (8, 8), 9)


def features(path):
    img = cv2.resize(cv2.imread(path), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = HOG.compute(gray).ravel()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    col = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    col /= col.sum() + 1e-6
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    tex, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return np.concatenate([hog, col * 20, tex * 5]).astype(np.float32)


def load_split(folder: Path):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
    y = np.array([int(f.split("_")[0]) for f in files])  # 1 = food, 0 = non-food
    paths = [str(folder / f) for f in files]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        X = np.stack(list(ex.map(features, paths, chunksize=64)))
    return X, y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/food-5k"))
    args = ap.parse_args()
    out = results_dir(__file__)
    root = Path(args.data)

    t = time.time()
    Xtr, ytr, _ = load_split(root / "training")
    Xva, yva, _ = load_split(root / "validation")
    Xte, yte, pte = load_split(root / "evaluation")
    print(f"features {Xtr.shape[1]} dims, extracted in {time.time()-t:.0f}s")

    # tune C / gamma on the official validation split
    best = None
    for C in [1, 3, 10]:
        for gamma in ["scale", 0.0005]:
            m = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=250, random_state=SEED)),
                          ("svm", SVC(C=C, gamma=gamma))]).fit(Xtr, ytr)
            acc = accuracy_score(yva, m.predict(Xva))
            print(f"C={C} gamma={gamma}: val acc {acc:.4f}")
            if best is None or acc > best[0]:
                best = (acc, C, gamma)
    val_acc, C, gamma = best

    # refit on train+validation, report once on the untouched evaluation split
    model = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=250, random_state=SEED)),
                      ("svm", SVC(C=C, gamma=gamma))]).fit(np.vstack([Xtr, Xva]), np.r_[ytr, yva])
    s = model.decision_function(Xte); p = (s > 0).astype(int)
    acc, auc = accuracy_score(yte, p), roc_auc_score(yte, s)
    cm = confusion_matrix(yte, p)
    rep = classification_report(yte, p, target_names=["non-food", "food"], output_dict=True)
    print(classification_report(yte, p, target_names=["non-food", "food"]))
    joblib.dump(model, out / "food_detector_svm.joblib", compress=3)

    # calorie table + coverage of Food-101 dishes
    table = load_table()
    matched = {c: lookup(c) for c in FOOD101}
    covered = {c: m for c, m in matched.items() if m}
    print(f"calorie table: {len(table)} foods; Food-101 classes with a calorie match: {len(covered)}/101")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Greens"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["non-food", "food"], yticklabels=["non-food", "food"],
           xlabel="Predicted", ylabel="Actual", title=f"Food detector (acc {acc:.1%})")
    save(fig, out / "confusion_matrix.png"); plt.close(fig)

    fpr, tpr, _ = roc_curve(yte, s)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(yte), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for ax, k in zip(axes.ravel(), pick):
        ax.imshow(cv2.cvtColor(cv2.resize(cv2.imread(pte[k]), (128, 128)), cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title("food" if p[k] else "non-food", fontsize=10,
                     color=PALETTE[5] if p[k] == yte[k] else PALETTE[7])
    fig.suptitle("Evaluation-set predictions (green = correct, red = wrong)")
    save(fig, out / "sample_predictions.png"); plt.close(fig)

    top = table.dropna(subset=["kcal_per_100g"]).sort_values("kcal_per_100g", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(top["kcal_per_100g"].clip(upper=900), bins=40, color=PALETTE[1])
    ax.set(title=f"Calorie density of {len(top)} foods in the lookup table",
           xlabel="kcal per 100 g", ylabel="Foods")
    save(fig, out / "calorie_distribution.png"); plt.close(fig)

    save_json({
        "dataset": {"train": int(len(ytr)), "validation": int(len(yva)), "evaluation": int(len(yte)),
                    "image_size": SIZE, "feature_dims": int(Xtr.shape[1])},
        "best_params": {"C": C, "gamma": gamma, "validation_accuracy": round(val_acc, 4)},
        "models": {"HOG + colour + LBP + RBF SVM (food detector)": {
            "accuracy": round(acc, 4), "roc_auc": round(auc, 4),
            "food_recall": round(rep["food"]["recall"], 4),
            "food_precision": round(rep["food"]["precision"], 4)}},
        "confusion_matrix": cm.tolist(),
        "roc": {"fpr": [round(v, 3) for v in np.linspace(0, 1, 51)],
                "tpr": [round(float(v), 4) for v in np.interp(np.linspace(0, 1, 51), fpr, tpr)]},
        "calories": {"table_foods": int(len(table)),
                     "food101_coverage": len(covered),
                     "food101_examples": {c: covered[c] for c in list(covered)[:12]}},
    }, out / "metrics.json")


if __name__ == "__main__":
    main()
