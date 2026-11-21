# Applied Machine Learning Projects

Five end-to-end ML projects covering regression, clustering, SVM image classification, gesture recognition and food recognition with calorie estimation. Each one is a runnable script that writes its metrics and figures to `results/`. Every reported number comes from data the model did not train on.

**🌐 Interactive results site:** https://chandana-galgali.github.io/applied-ml-projects/

| # | Project | Technique | Result |
|---|---|---|---|
| 01 | [House Price Prediction](01-house-price-regression) | Linear / Ridge regression, scikit-learn pipelines | **R² 0.930**, MAE $13,981 (5-fold CV) |
| 02 | [Customer Segmentation](02-customer-segmentation) | K-Means, silhouette & elbow analysis | **5 segments**, silhouette 0.555 |
| 03 | [Cats vs Dogs Classifier](03-cats-vs-dogs-svm) | HOG + colour features, RBF SVM | **80.7%** accuracy (vs 57.0% raw pixels), AUC 0.895 |
| 04 | [Hand Gesture Recognition](04-hand-gesture-recognition) | Hand crop + HOG + SVM; Keras CNN notebook | **83.2%** on *unseen people* across 10 gestures |
| 05 | [Food Recognition & Calories](05-food-recognition-calories) | HOG/colour/LBP SVM; EfficientNet notebook; calorie lookup | **89.0%** food detection, AUC 0.954, 541-food calorie table |

## Quick start

```bash
git clone https://github.com/chandana-galgali/applied-ml-projects.git
cd applied-ml-projects
pip install -r requirements.txt
# download datasets into data/  (see data/README.md)
bash run_all.sh          # trains all five, rebuilds docs/ (≈15 min on a laptop CPU)
```

Or run one project at a time: `cd 02-customer-segmentation && python train.py`.

## Repository layout

```
├── 01-house-price-regression/      train.py, results/
├── 02-customer-segmentation/       train.py, results/
├── 03-cats-vs-dogs-svm/            train.py, results/
├── 04-hand-gesture-recognition/    train.py, predict_webcam.py, gesture_cnn_gpu.ipynb, results/
├── 05-food-recognition-calories/   train.py, calories.py, food101_cnn_gpu.ipynb, assets/, results/
├── common/style.py                 shared plotting style
├── scripts/build_site.py           copies results into docs/ for the website
├── docs/                           GitHub Pages site (index.html + data.js + figures)
└── data/README.md                  dataset download instructions
```

## Evaluation principles

- **No leakage:** preprocessing lives inside scikit-learn `Pipeline`s, so it is fitted only on training folds.
- **Honest splits:** gestures are evaluated on people the model never saw. A random split scores 100% there and is reported only for contrast.
- **Baselines included:** each project shows the simple approach next to the improved one, so the gain is visible.
- **Reproducible:** fixed seeds; `bash run_all.sh` regenerates every number and figure.
