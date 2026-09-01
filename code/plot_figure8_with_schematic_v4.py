#!/usr/bin/env python3
"""Figure 8 v4: compact orientation schematic plus two quantitative panels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from _paths import RESULTS


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#286D9B"
TEAL = "#2B978A"
ORANGE = "#D87832"
RED = "#BC5149"
DARK = "#30363A"
GREY = "#879198"
CMAP = LinearSegmentedColormap.from_list(
    "fabric", ["#F5F8F9", "#B9D6E2", "#56A0A0", "#D59A4B", "#A84445"]
)


def rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")


def panel(ax, label):
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def contact_cartoon(ax, fabric=False):
    rng = np.random.default_rng(17 if not fabric else 21)
    xy = rng.uniform(0.08, 0.92, (22, 2))
    if fabric:
        for i in range(0, 20, 2):
            x = rng.uniform(.12, .88)
            y = rng.uniform(.16, .72)
            xy[i] = [x, y]
            xy[i + 1] = [x + rng.normal(0, .018), y + rng.uniform(.13, .22)]
    else:
        rng.shuffle(xy)
    for i in range(0, 20, 2):
        ax.plot(xy[i:i+2, 0], xy[i:i+2, 1],
                color=ORANGE if fabric else GREY,
                lw=1.6 if fabric else 1.0, alpha=.9)
    ax.scatter(xy[:20, 0], xy[:20, 1], s=20, c=TEAL,
               edgecolor="white", linewidth=.35, zorder=3)
    ax.annotate("", xy=(.93, .90), xytext=(.63, .90),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    ax.text(.78, .84, "field", ha="center", color=BLUE, fontsize=5.3)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")


def main():
    root = RESULTS / "topology_research" / "rve_directional_scaling"
    phase = rows(root / "phase_summary.csv")
    fem = rows(root / "fem_metrics.csv")
    mesh = json.loads((root / "fem_mesh_verification.json").read_text())

    fabrics = sorted({float(r["fabric_input"]) for r in phase})
    ps = sorted({float(r["p_mean"]) for r in phase})
    m = np.zeros((len(ps), len(fabrics)))
    for i, p in enumerate(ps):
        for j, fabric in enumerate(fabrics):
            q = next(r for r in phase if float(r["p_mean"]) == p and float(r["fabric_input"]) == fabric)
            m[i, j] = float(q["z_over_xy_median"])

    freq = np.array([0.1, 1.0, 10.0])
    iso = np.array([
        float(next(r for r in fem if r["network"] == "isotropic" and float(r["frequency_over_f0"]) == f)["fem_z_over_xy"])
        for f in freq
    ])
    axial = np.array([
        float(next(r for r in fem if r["network"] == "axial_fabric" and float(r["frequency_over_f0"]) == f)["fem_z_over_xy"])
        for f in freq
    ])

    fig = plt.figure(figsize=(183 / 25.4, 122 / 25.4))
    gs = fig.add_gridspec(2, 12, height_ratios=[0.60, 1.30],
                          left=.075, right=.98, bottom=.18, top=.94,
                          hspace=.72, wspace=.42)

    # a: compact conceptual guide, not an independent data panel.
    ax = fig.add_subplot(gs[0, :])
    panel(ax, "a")
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(.00, .98, "Two idealized end-members of contact-orientation order",
            fontsize=8.6, fontweight="bold", va="top")
    r1 = ax.inset_axes([.03, .16, .23, .66])
    r2 = ax.inset_axes([.36, .16, .23, .66])
    contact_cartoon(r1, False)
    contact_cartoon(r2, True)
    ax.text(.145, .05, "statistically isotropic\ncontact orientations",
            ha="center", va="top", fontsize=6.0, fontweight="bold", linespacing=.9)
    ax.text(.475, .05, "persistent conductive-mineral\nfabric",
            ha="center", va="top", fontsize=6.0, fontweight="bold", linespacing=.9)
    ax.text(.70, .67, "random orientations", fontsize=5.5,
            color=GREY, fontweight="bold")
    ax.text(.70, .54, "self-average with increasing volume", fontsize=6.2,
            color=BLUE)
    ax.text(.70, .30, "persistent fabric", fontsize=5.5,
            color=GREY, fontweight="bold")
    ax.text(.70, .17, "directional response remains visible", fontsize=6.2,
            color=RED, fontweight="bold")

    # b: central quantitative result — persistent fabric at sample scale.
    ax1 = fig.add_subplot(gs[1, :7])
    panel(ax1, "b")
    im = ax1.imshow(m, origin="lower", aspect="auto", cmap=CMAP, norm=Normalize(1, 8.5))
    ax1.set_xticks(range(len(fabrics)), [f"{a:.1f}" for a in fabrics])
    ax1.set_yticks(range(len(ps)), [f"{p:.2f}" for p in ps])
    ax1.set_xlabel("imposed fabric")
    ax1.set_ylabel("mean bond occupancy, p")
    ax1.set_title(r"Persistent fabric restores $R_z/R_{xy}$ at $N=1728$", fontsize=8.8, pad=7)
    for i in range(len(ps)):
        for j in range(len(fabrics)):
            val = m[i, j]
            ax1.text(j, i, f"{val:.1f}×", ha="center", va="center", fontsize=6.8,
                     color="white" if val > 3 else DARK, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.035)
    cb.set_label("axial / transverse response", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.0)
    ax1.text(0.5, -0.25,
             r"At $\widetilde{\omega}=1$: fabric 0.2, 0.4 and 0.6 produce $R_z/R_{xy}$ of 1.8–2.1, 3.4–4.0 and 5.7–8.2.",
             transform=ax1.transAxes, ha="center", va="top", fontsize=5.7, color=BLUE)

    # c: causal validation — same geometry and graph, redistribution only.
    ax2 = fig.add_subplot(gs[1, 8:])
    panel(ax2, "c")
    ax2.plot(freq, iso, "o--", color=GREY, lw=1.2, ms=4.0, label="isotropic $G$")
    ax2.plot(freq, axial, "o-", color=RED, lw=1.6, ms=4.2, label="axial-fabric $G$")
    ax2.set_xscale("log")
    ax2.set_xlim(0.07, 14)
    ax2.set_ylim(0.88, 3.75)
    ax2.set_xticks(freq, ["0.1", "1", "10"])
    ax2.set_xlabel(r"frequency, $f/f_0$")
    ax2.set_ylabel(r"FEM $R_z/R_{xy}$")
    ax2.set_title("Full-FEM validation\nsame geometry, graph and total conductance",
                  fontsize=7.8, pad=7)
    ax2.legend(fontsize=5.8, loc="upper left", handlelength=1.5)
    for x, y in zip(freq, iso):
        ax2.text(x, y + 0.16, f"{y:.2f}×", ha="center", va="bottom", fontsize=5.9, color=DARK)
    for x, y in zip(freq, axial):
        if x == 0.1:
            ax2.text(x, y + 0.14, f"{y:.2f}×", ha="center", va="bottom", fontsize=5.9, color=RED, fontweight="bold")
        else:
            ax2.text(x, y - 0.20, f"{y:.2f}×", ha="center", va="top", fontsize=5.9, color=RED, fontweight="bold")
    ax2.text(0.5, -0.25,
             rf"Fine-mesh ratio change at $f/f_0=1$: {100*mesh['relative_ratio_change']:.2f}%.",
             transform=ax2.transAxes, ha="center", va="top", fontsize=5.7, color=RED, fontweight="bold")

    stem = root / "Figure_micro_to_macro_directional_survival_schematic_v4"
    submission_stem = RESULTS / "topology_research" / "jgr_topology_submission" / "figures" / "Figure_8_micro_to_macro_schematic_v4"
    save(fig, stem)
    save(fig, submission_stem)
    plt.close(fig)
    print(submission_stem)


if __name__ == "__main__":
    main()
