#!/usr/bin/env python3
"""Create the main-text three-particle topology benchmark figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _paths import RESULTS


ROOT = RESULTS / "topology_research"
OUT = ROOT / "jgr_topology_submission" / "figures"

NAVY = "#1F5A85"
ORANGE = "#D77835"
TEAL = "#2B9A8B"
RED = "#BD5149"
GREY = "#879099"
GRID = "#D8DEE2"
COLORS = ["#A8C8D9", "#79AEC9", "#4D91B6", NAVY,
          TEAL, "#82A968", "#C8A14A", ORANGE]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6.5,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def panel(ax, label):
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def label_case(case_id: str) -> str:
    names = {
        "disconnected": "disconnected",
        "single_edge_0p02": "single edge 0.02 S",
        "weak_chain": "weak chain",
        "transition_chain": "transition chain",
        "strong_chain": "strong chain",
        "bottleneck_chain": "bottleneck chain",
        "ideal_pair_plus_isolated": "ideal pair + isolated",
        "ideal_cluster": "ideal cluster",
    }
    return names[case_id]


def main():
    data = json.loads((ROOT / "three_particle_topology" / "results.json").read_text())
    rows = data["rows"]
    summaries = data["summaries"]
    cases = [s["case_id"] for s in summaries if s["orientation"] == "parallel"]

    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    gs = fig.add_gridspec(1, 3, left=.055, right=.985, bottom=.21, top=.78,
                          width_ratios=[1.15, 1.15, .95], wspace=.50)

    for col, case in zip(COLORS, cases):
        rr = sorted([r for r in rows if r["orientation"] == "parallel"
                     and r["case_id"] == case],
                    key=lambda r: float(r["frequency_over_f0"]))
        f = np.array([float(r["frequency_over_f0"]) for r in rr])
        y = 1e3 * np.array([float(r["sigma_im_S_m"]) for r in rr])
        ax = fig.add_subplot(gs[0, 0]) if case == cases[0] else ax
        ax.semilogx(f, y, "o-", color=col, lw=1.0, ms=2.3,
                    label=label_case(case))
    panel(ax, "a")
    ax.axvspan(.1, 3, color="#F7E4D6", alpha=.35, lw=0)
    ax.set_xlabel(r"normalized frequency, $f/f_0$")
    ax.set_ylabel(r"$\sigma''$ (mS m$^{-1}$)")
    ax.set_title("Topology changes the field-aligned spectrum", fontsize=7.2, pad=5)
    ax.legend(ncol=2, fontsize=4.7, loc="upper right", handlelength=1.8,
              columnspacing=.8)
    ax.grid(True, which="both", alpha=.22)

    ax_b = fig.add_subplot(gs[0, 1])
    for col, case in zip(COLORS, cases):
        rr = sorted([r for r in rows if r["orientation"] == "transverse"
                     and r["case_id"] == case],
                    key=lambda r: float(r["frequency_over_f0"]))
        f = np.array([float(r["frequency_over_f0"]) for r in rr])
        y = 1e3 * np.array([float(r["sigma_im_S_m"]) for r in rr])
        ax_b.semilogx(f, y, "o-", color=col, lw=1.0, ms=2.3)
    panel(ax_b, "b")
    ax_b.axvspan(.1, 3, color="#F7E4D6", alpha=.35, lw=0)
    ax_b.set_xlabel(r"normalized frequency, $f/f_0$")
    ax_b.set_ylabel(r"$\sigma''$ (mS m$^{-1}$)")
    ax_b.set_title("The same topology is invisible transversely", fontsize=7.2, pad=5)
    ax_b.text(.04, .92, "all eight contact graphs\ncoincide to numerical precision",
              transform=ax_b.transAxes, va="top", color=NAVY,
              fontweight="bold", fontsize=5.8,
              bbox=dict(fc="white", ec="#DCEAF2", pad=2))
    ax_b.grid(True, which="both", alpha=.22)

    ax_c = fig.add_subplot(gs[0, 2])
    base = next(s for s in summaries if s["orientation"] == "parallel"
                and s["case_id"] == "disconnected")
    ss = [s for s in summaries if s["orientation"] == "parallel"]
    x = np.arange(len(cases))
    amp = np.array([float(next(s for s in ss if s["case_id"] == c)["peak_interp"])
                    for c in cases]) / float(base["peak_interp"])
    fp = np.array([float(next(s for s in ss if s["case_id"] == c)["fp_interp_hz"])
                   for c in cases]) / float(base["fp_interp_hz"])
    ax_c.plot(x, amp, "o-", color=ORANGE, lw=1.1, ms=3,
              label="peak amplitude / disconnected")
    ax_c.plot(x, fp, "s-", color=NAVY, lw=1.1, ms=3,
              label="peak frequency / disconnected")
    ax_c.axhline(1, color=GRID, lw=.7)
    panel(ax_c, "c")
    short_names = ["disconnected", "single edge", "weak chain", "transition",
                   "strong chain", "bottleneck", "ideal pair", "ideal cluster"]
    ax_c.set_xticks(x, short_names,
                    rotation=35, ha="right", fontsize=4.8)
    ax_c.set_ylabel("normalized peak descriptor")
    ax_c.set_title("Topology changes amplitude and time scale", fontsize=7.2, pad=5)
    ax_c.legend(fontsize=4.8, loc="upper left")
    ax_c.grid(True, axis="y", alpha=.22)

    fig.suptitle("Contact topology is field-selective in a three-particle chain",
                 fontsize=9.2, fontweight="bold", y=.95)
    save(fig, OUT / "Figure_3_three_particle_topology_benchmark")
    plt.close(fig)


if __name__ == "__main__":
    main()
