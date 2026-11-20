"""Common look of all figures, and one function to save them.

The figures use Computer Modern, the font of the LaTeX report.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write figures to files, no window needed
import matplotlib.pyplot as plt

# Colours used in all figures
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
VIOLET = "#6250d6"
GREY = "#8c8c8c"
INK = "#333333"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,     # cmr10 has no Unicode minus sign
    "font.size": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def save_figure(fig, folder, name):
    """Save the figure as <folder>/<name>.png (used by the report and the README)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
