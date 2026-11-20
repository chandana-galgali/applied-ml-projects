"""Calorie lookup: food name -> calories per serving and per 100 g.

The raw table stores calories as text ("272 cal") and serving sizes as free text
("1 slice (102 g)"). This module parses both, removes duplicates and does fuzzy
name matching so model labels such as "french_fries" or "Pizza " still resolve.
"""
import difflib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

TABLE = Path(__file__).resolve().parent / "assets/food_calories.csv"


@lru_cache(maxsize=1)
def load_table(path: str = str(TABLE)) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Food"] = df["Food"].str.strip()
    df["key"] = df["Food"].str.lower()
    df["kcal_per_serving"] = df["Calories"].str.extract(r"(\d+(?:\.\d+)?)").astype(float)
    df["serving_g"] = df["Serving"].str.extract(r"\((\d+(?:\.\d+)?)\s*g\)").astype(float)
    df["kcal_per_100g"] = (df["kcal_per_serving"] / df["serving_g"] * 100).round(1)
    return df.drop_duplicates("key").reset_index(drop=True)


def normalise(name: str) -> str:
    return re.sub(r"[_\-]+", " ", name).strip().lower()


# Hand-checked corrections where automatic matching picks a different dish.
OVERRIDES = {
    "macarons": None,             # would match "Macaroni"
    "steak": None,                # would match "Chicken Fried Steak"
    "dumplings": None,            # would match "Dumpling Dough"
    "ice cream": "french vanilla ice cream",
    "caesar salad": "chicken caesar salad",
}


def _tokens(s: str):
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split()}


def lookup(name: str, cutoff: float = 0.85):
    """Return a dict for the matching food, or None.

    Matching is deliberately strict - a wrong calorie value is worse than none:
      1. exact name
      2. a table entry containing every word of the query ("lobster bisque" -> "Lobster Bisque Soup")
      3. near-identical spelling (ratio >= 0.85, e.g. "tacos" -> "Taco")
    """
    df = load_table()
    q = normalise(name)
    keys = df["key"].tolist()
    if q in OVERRIDES:
        if OVERRIDES[q] is None:
            return None
        q = OVERRIDES[q]
    if q in keys:
        match = q
    else:
        qt = _tokens(q)
        contains = [k for k in keys if qt <= _tokens(k)]
        if contains:
            match = min(contains, key=len)
        else:
            cands = difflib.get_close_matches(q, keys, n=1, cutoff=cutoff)
            if not cands:
                return None
            match = cands[0]
    score = difflib.SequenceMatcher(None, q, match).ratio()
    row = df[df["key"] == match].iloc[0]
    return {"query": name, "matched": row["Food"], "match_score": round(score, 3),
            "serving": row["Serving"], "kcal_per_serving": float(row["kcal_per_serving"]),
            "kcal_per_100g": None if pd.isna(row["kcal_per_100g"]) else float(row["kcal_per_100g"])}


if __name__ == "__main__":
    for q in ["Pizza", "french_fries", "avocado", "chicken curry", "ice cream", "sushi"]:
        print(q, "->", lookup(q))
