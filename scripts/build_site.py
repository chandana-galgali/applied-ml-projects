"""Collects every project's results/ into docs/ (the GitHub Pages site).

    python scripts/build_site.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "05-food-recognition-calories"))
from calories import load_table  # noqa: E402

PROJECTS = ["01-house-price-regression", "02-customer-segmentation", "03-cats-vs-dogs-svm",
            "04-hand-gesture-recognition", "05-food-recognition-calories"]

data = {}
for p in PROJECTS:
    res = ROOT / p / "results"
    data[p] = json.loads((res / "metrics.json").read_text())
    dst = DOCS / "img" / p
    dst.mkdir(parents=True, exist_ok=True)
    for png in res.glob("*.png"):
        shutil.copy2(png, dst / png.name)

t = load_table().dropna(subset=["kcal_per_serving"])
data["calorie_table"] = [[r.Food, r.Serving, r.kcal_per_serving, r.kcal_per_100g if r.kcal_per_100g == r.kcal_per_100g else None]
                         for r in t.itertuples()]

(DOCS / "data.js").write_text("window.ML_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {DOCS / 'data.js'} and copied figures for {len(PROJECTS)} projects")
