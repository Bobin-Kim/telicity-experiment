#!/usr/bin/env python3
"""Create an ACL-style grouped bar chart for Group C accuracy."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "group_c_before_after_ba"

models = [
    "Qwen2.5-7B",
    "Mistral-7B-v0.3",
    "DeepSeek-7B",
    "Llama-3.1-8B",
    "Gemma-2-9B",
]

before = [0.41, 0.01, 0.00, 0.07, 0.16]
after = [0.55, 0.56, 0.79, 0.53, 0.68]
delta_pp = [14, 55, 79, 46, 52]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def main() -> None:
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
    before_bars = ax.bar(
        x - width / 2,
        before,
        width,
        label="Before",
        color="#9ECAE1",
        edgecolor="#4A7891",
        linewidth=0.55,
        zorder=3,
    )
    after_bars = ax.bar(
        x + width / 2,
        after,
        width,
        label="After",
        color="#2171B5",
        edgecolor="#174F80",
        linewidth=0.55,
        zorder=3,
    )

    for i, (_b, a, delta) in enumerate(zip(before, after, delta_pp)):
        ax.text(
            i + width / 2,
            a + 0.025,
            f"+{delta} pp",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel("Group C Accuracy")
    ax.set_xticks(x, models, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 0.94)
    ax.set_yticks(np.arange(0, 1.0, 0.2))
    ax.grid(axis="y", color="0.88", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.5, width=0.65)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
