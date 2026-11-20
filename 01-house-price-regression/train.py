"""House price prediction with Linear Regression.

Task: predict a house's sale price from its living area, bedrooms and bathrooms,
then check how much a full-feature model improves on that baseline.

Dataset: Kaggle "House Prices - Advanced Regression Techniques" (train.csv / test.csv)
Run:     python train.py --data ../data/house-prices
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.style import PALETTE, plt, results_dir, save_json  # noqa: E402

SEED = 42


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0))
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"].fillna(0)
    return df


def metrics(y_true_log, y_pred_log):
    """Metrics in real dollars plus RMSE on log price (the Kaggle leaderboard metric)."""
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "R2": round(r2_score(y_true, y_pred), 4),
        "RMSE_usd": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 0),
        "MAE_usd": round(float(mean_absolute_error(y_true, y_pred)), 0),
        "MAPE_pct": round(float(np.mean(np.abs(y_true - y_pred) / y_true) * 100), 2),
        "RMSE_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/house-prices"))
    args = ap.parse_args()
    out = results_dir(__file__)

    train = pd.read_csv(Path(args.data) / "train.csv")
    test = pd.read_csv(Path(args.data) / "test.csv")

    # Two well-known outliers: huge houses sold far below value (partial sales).
    outliers = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    print(f"Dropping {outliers.sum()} documented outliers")
    train = train[~outliers].reset_index(drop=True)

    train, test = add_core_features(train), add_core_features(test)
    y = np.log1p(train["SalePrice"])  # log target: prices are right-skewed
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) The task as specified: square footage + bedrooms + bathrooms
    core = ["GrLivArea", "BedroomAbvGr", "TotalBath"]
    lr = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("scale", StandardScaler()),
                   ("lr", LinearRegression())])

    # 2) Regularised linear regression on every feature (one-hot categoricals)
    features = [c for c in train.columns if c not in ("Id", "SalePrice")]
    num = train[features].select_dtypes(include="number").columns.tolist()
    cat = [c for c in features if c not in num]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="None")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat),
    ])
    ridge = Pipeline([("prep", prep), ("ridge", RidgeCV(alphas=np.logspace(-1, 3, 30)))])

    # 3) Non-linear reference model
    gbr = Pipeline([("prep", prep), ("gbr", GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED))])

    models = {
        "Linear Regression (3 features)": (lr, core),
        "Ridge Regression (all features)": (ridge, features),
        "Gradient Boosting (all features)": (gbr, features),
    }
    results, preds = {}, {}
    for name, (model, cols) in models.items():
        p = cross_val_predict(model, train[cols], y, cv=cv)
        preds[name] = p
        results[name] = metrics(y, p)
        print(f"{name:35s} {results[name]}")

    # Coefficients of the 3-feature model, in "% price change per +1 std" terms
    lr.fit(train[core], y)
    coefs = dict(zip(core, lr.named_steps["lr"].coef_))
    coef_pct = {k: round((np.exp(v) - 1) * 100, 2) for k, v in coefs.items()}

    # Kaggle submission from the best linear model (keeps the task a regression task)
    ridge.fit(train[features], y)
    sub = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(ridge.predict(test[features]))})
    sub.to_csv(out / "submission.csv", index=False)

    # ---------- figures ----------
    best_lin = "Ridge Regression (all features)"
    yt = np.expm1(y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, color in zip(axes, ["Linear Regression (3 features)", best_lin], PALETTE):
        yp = np.expm1(preds[name])
        ax.scatter(yt / 1e3, yp / 1e3, s=8, alpha=0.45, color=color)
        lim = [0, max(yt.max(), yp.max()) / 1e3]
        ax.plot(lim, lim, "--", color="#334155", lw=1)
        ax.set(title=f"{name}\nR² = {results[name]['R2']:.3f}",
               xlabel="Actual price ($k)", ylabel="Predicted price ($k)")
    fig.savefig(out / "predicted_vs_actual.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(list(coef_pct), list(coef_pct.values()), color=PALETTE[0])
    ax.set(title="Linear model: price change per +1 std of feature", xlabel="% change in price")
    fig.savefig(out / "coefficients.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    res = yt - np.expm1(preds[best_lin])
    ax.hist(res / 1e3, bins=60, color=PALETTE[2])
    ax.set(title="Ridge residuals (cross-validated)", xlabel="Actual − predicted ($k)", ylabel="Houses")
    fig.savefig(out / "residuals.png"); plt.close(fig)

    # small sample for the website scatter plot
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), 400, replace=False)
    save_json({
        "dataset": {"rows": int(len(train)), "features": len(features), "test_rows": int(len(test))},
        "cv": "5-fold, shuffled, seed 42",
        "models": results,
        "linear_coefficients_pct_per_std": coef_pct,
        "scatter_sample": {
            "actual": [round(float(v)) for v in yt.iloc[idx]],
            "linear3": [round(float(v)) for v in np.expm1(preds["Linear Regression (3 features)"][idx])],
            "ridge": [round(float(v)) for v in np.expm1(preds[best_lin][idx])],
        },
    }, out / "metrics.json")


if __name__ == "__main__":
    main()
