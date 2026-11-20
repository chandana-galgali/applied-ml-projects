"""Shared plotting style and small helpers used by every project."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "font.size": 10,
})


def results_dir(script_file: str) -> Path:
    out = Path(script_file).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    return out


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))
    print(f"saved {path}")


def save(fig, path: Path):
    """Save a figure in the house style.

    Figure-level titles are removed on purpose: the website and READMEs give every
    figure the same caption header, so titles baked into the image would clash.
    Panel titles in multi-panel figures (e.g. one per image) are kept, at one size.
    """
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    positions = {tuple(round(v, 3) for v in ax.get_position().bounds) for ax in fig.axes}
    for ax in fig.axes:
        if len(positions) == 1:
            ax.set_title("")
        else:
            ax.title.set_fontsize(10)
    fig.patch.set_facecolor("white")
    fig.savefig(path, facecolor="white", pad_inches=0.15)
