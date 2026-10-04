#!/usr/bin/env python3
"""Create an ACL-style four-stage balanced-accuracy figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "rq1_four_stage_ba"

stages = ["Pre-erasure", "Post-erasure", "After attention", "After MLP"]
data = {
    "Qwen2.5-7B": [0.933, 0.778, 0.903, 0.893],
    "Mistral-7B-v0.3": [0.893, 0.500, 0.855, 0.842],
    "DeepSeek-7B": [0.915, 0.570, 0.900, 0.898],
    "Llama-3.1-8B": [0.918, 0.495, 0.893, 0.887],
    "Gemma-2-9B": [0.928, 0.615, 0.905, 0.898],
}

# Okabe-Ito colorblind-safe palette; markers and line styles remain distinct
# when the figure is printed in grayscale.
styles = [
    ("#0072B2", "o", "-"),
    ("#D55E00", "s", "--"),
    ("#009E73", "^", "-."),
    ("#CC79A7", "D", ":"),
    ("#E69F00", "P", (0, (5, 1, 1, 1))),
]

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
    "lines.linewidth": 1.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def main() -> None:
    fig, (ax_erase, ax_reemerge) = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.65),
        sharey=True,
        gridspec_kw={"width_ratios": [2, 3]},
        constrained_layout=True,
    )

    for (label, values), (color, marker, linestyle) in zip(data.items(), styles):
        common_style = {
            "label": label,
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
            "markersize": 4.2,
            "markeredgecolor": "white",
            "markeredgewidth": 0.45,
            "zorder": 3,
        }
        ax_erase.plot(
            [0, 1],
            values[:2],
            **common_style,
        )
        ax_reemerge.plot(
            [0, 1, 2],
            values[1:],
            **common_style,
        )

    for ax in (ax_erase, ax_reemerge):
        ax.axhline(
            0.5,
            color="0.45",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.set_ylim(0.45, 0.97)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.grid(axis="y", color="0.88", linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.5, width=0.65)

    ax_erase.set_title("(a) Erasure effect", fontsize=9, pad=5)
    ax_erase.set_ylabel("Telicity balanced accuracy")
    ax_erase.set_xticks([0, 1], ["Pre-\nerasure", "Post-\nerasure"])

    ax_reemerge.set_title("(b) Re-emergence after erasure", fontsize=9, pad=5)
    ax_reemerge.set_xticks(
        [0, 1, 2],
        ["Post-\nerasure", "After\nattention", "After\nMLP"],
    )
    ax_reemerge.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handlelength=2.3,
        handletextpad=0.45,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
