# Datasets

Datasets are not committed (they total ~4 GB). Download each from Kaggle and unzip it here so the folder looks like this:

```
data/
├── house-prices/        train.csv, test.csv
├── mall-customers/      Mall_Customers.csv
├── dogs-vs-cats/        train/ (cat.0.jpg …), test1/
├── leapgestrecog/       00/ … 09/  (subject folders)
└── food-5k/             training/, validation/, evaluation/
```

| Folder | Kaggle dataset |
|---|---|
| `house-prices` | [House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) |
| `mall-customers` | [Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| `dogs-vs-cats` | [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) |
| `leapgestrecog` | [Hand Gesture Recognition Database (leapGestRecog)](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) — copy the inner `leapGestRecog/` contents here |
| `food-5k` | [Food-5K](https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset) |

With the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house-prices
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data/mall-customers --unzip
kaggle competitions download -c dogs-vs-cats -p data/dogs-vs-cats
kaggle datasets download -d gti-upm/leapgestrecog -p data/leapgestrecog --unzip
kaggle datasets download -d trolukovich/food5k-image-dataset -p data/food-5k --unzip
```

Every script also accepts `--data <path>` if you keep the datasets somewhere else.
