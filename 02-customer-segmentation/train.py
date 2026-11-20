"""Customer segmentation with K-Means.

Task: group retail customers by purchasing behaviour (annual income vs spending score),
choosing the number of clusters with the elbow method AND silhouette score
instead of guessing it.

Dataset: Kaggle "Customer Segmentation Tutorial in Python" (Mall_Customers.csv)
Run:     python train.py --data ../data/mall-customers
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.style import PALETTE, plt, results_dir, save_json  # noqa: E402

SEED = 42
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def name_segment(income, spend, inc_mid, sp_mid):
    """Readable business label from where a centroid sits relative to the averages."""
    tol_i, tol_s = 0.15 * inc_mid, 0.15 * sp_mid
    hi_i, lo_i = income > inc_mid + tol_i, income < inc_mid - tol_i
    hi_s, lo_s = spend > sp_mid + tol_s, spend < sp_mid - tol_s
    if hi_i and hi_s: return "Premium spenders"
    if hi_i and lo_s: return "Careful high earners"
    if lo_i and hi_s: return "Impulsive buyers"
    if lo_i and lo_s: return "Budget-conscious"
    return "Average customers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/mall-customers"))
    args = ap.parse_args()
    out = results_dir(__file__)

    df = pd.read_csv(Path(args.data) / "Mall_Customers.csv")
    X = StandardScaler().fit_transform(df[FEATURES])

    # ---- choose k with two independent criteria ----
    ks = list(range(2, 11))
    inertia, sil, db, ch = [], [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(X, km.labels_)))
        db.append(float(davies_bouldin_score(X, km.labels_)))
        ch.append(float(calinski_harabasz_score(X, km.labels_)))
    best_k = ks[int(np.argmax(sil))]
    print("silhouette by k:", dict(zip(ks, np.round(sil, 3))), "-> best k =", best_k)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(X)
    df["Cluster"] = km.labels_

    inc_mid, sp_mid = df[FEATURES[0]].mean(), df[FEATURES[1]].mean()
    prof = (df.groupby("Cluster")
              .agg(customers=("CustomerID", "count"),
                   avg_age=("Age", "mean"),
                   avg_income_k=(FEATURES[0], "mean"),
                   avg_spending=(FEATURES[1], "mean"),
                   pct_female=("Gender", lambda g: (g == "Female").mean() * 100))
              .round(1))
    prof["segment"] = [name_segment(r.avg_income_k, r.avg_spending, inc_mid, sp_mid)
                       for r in prof.itertuples()]
    print(prof.to_string())
    df["Segment"] = df["Cluster"].map(prof["segment"])
    df.to_csv(out / "clustered_customers.csv", index=False)
    prof.to_csv(out / "segment_profiles.csv")

    # ---- figures ----
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertia, "o-", color=PALETTE[0], label="Inertia (WCSS)")
    ax1.set(xlabel="Number of clusters k", ylabel="Inertia (WCSS)")
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(ks, sil, "s--", color=PALETTE[1], label="Silhouette")
    ax2.set_ylabel("Silhouette score")
    ax1.axvline(best_k, color="#334155", ls=":", lw=1)
    ax1.set_title(f"Elbow + silhouette both point to k = {best_k}")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88)); fig.savefig(out / "elbow_silhouette.png"); plt.close(fig)

    scaler = StandardScaler().fit(df[FEATURES])
    centers = scaler.inverse_transform(km.cluster_centers_)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in range(best_k):
        m = df["Cluster"] == c
        ax.scatter(df.loc[m, FEATURES[0]], df.loc[m, FEATURES[1]], s=28, alpha=0.8,
                   color=PALETTE[c], label=f"{prof.loc[c, 'segment']} ({m.sum()})")
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, color="#0f172a", label="Centroids")
    ax.set(xlabel=FEATURES[0], ylabel=FEATURES[1], title="Customer segments (K-Means)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(out / "segments.png"); plt.close(fig)

    save_json({
        "dataset": {"rows": int(len(df)), "features_used": FEATURES},
        "k_search": {"k": ks, "inertia": [round(v, 2) for v in inertia],
                     "silhouette": [round(v, 4) for v in sil],
                     "davies_bouldin": [round(v, 4) for v in db]},
        "best_k": best_k,
        "final": {"silhouette": round(sil[ks.index(best_k)], 4),
                  "davies_bouldin": round(db[ks.index(best_k)], 4),
                  "calinski_harabasz": round(ch[ks.index(best_k)], 1)},
        "segments": prof.reset_index().to_dict(orient="records"),
        "points": {"income": df[FEATURES[0]].tolist(), "spending": df[FEATURES[1]].tolist(),
                   "cluster": df["Cluster"].tolist()},
        "centroids": [[round(a, 1), round(b, 1)] for a, b in centers],
    }, out / "metrics.json")


if __name__ == "__main__":
    main()
