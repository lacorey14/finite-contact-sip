#!/usr/bin/env python3
"""Readable revision of the Figure 7 self-averaging atlas.

The original Figure 7 is preserved. This version retains all six evidence
panels while clarifying the occupancy encoding and replacing the low-density
residual scatter with a density view.
"""
from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from plot_jgr_added_key_figures import (
    BLUE, COLORS, DARK, GREY, GRID, NAVY, ORANGE, RED, ROOT, TEAL,
    CMAP_BLUE, OUT, panel, save,
)


def main() -> None:
    raw = json.loads((ROOT / "rve_directional_scaling/raw_results.json").read_text())
    analysis = json.loads((ROOT / "rve_directional_scaling/analysis.json").read_text())
    size = [r for r in raw["rows"] if r["family"] == "size" and
            np.isclose(float(r["omega_over_GC"]), 1.0)]
    ps = sorted({float(r["p_mean"]) for r in size})
    ns = sorted({int(r["n_nodes"]) for r in size})

    fig = plt.figure(figsize=(183/25.4, 180/25.4))
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.02, 1.0],
        left=.06, right=.985, bottom=.075, top=.93,
        hspace=.30, wspace=1.20,
    )
    fig.suptitle(
        "Random contact directions self-average with network size",
        fontsize=9.0, fontweight="bold", y=.985,
    )

    # a: distributional contraction at one representative occupancy.
    ax = fig.add_subplot(gs[0, :4]); panel(ax, "a")
    p0 = .45; show = [64, 512, 4096]
    vals = [[float(r["anisotropy_cv"]) for r in size
             if np.isclose(float(r["p_mean"]), p0) and int(r["n_nodes"]) == n]
            for n in show]
    vp = ax.violinplot(vals, positions=range(3), showmeans=False,
                       showmedians=True, widths=.76)
    for body, col in zip(vp["bodies"], ["#A9CADC", BLUE, NAVY]):
        body.set_facecolor(col); body.set_edgecolor("none"); body.set_alpha(.8)
    vp["cmedians"].set_color(RED); vp["cmedians"].set_linewidth(1.8)
    for i, v in enumerate(vals):
        ax.scatter(i + np.linspace(-.16, .16, len(v)), v, s=6,
                   color=DARK, alpha=.28, zorder=3)
    ax.set_xticks(range(3), [str(n) for n in show])
    ax.set(xlabel="mineral nodes, N", ylabel="directional variability, CV",
           title="Distribution contracts with network size")
    ax.text(.97, .95, "p = 0.45; 32 realizations\nred = median",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4,
            color=GREY)

    # b: hero scaling relation across occupancy.
    ax = fig.add_subplot(gs[0, 4:9]); panel(ax, "b")
    for col, p in zip(COLORS, ps):
        rr = sorted([r for r in analysis["size_summary"]
                     if np.isclose(float(r["p_mean"]), p)],
                    key=lambda r: int(r["n_nodes"]))
        ax.loglog([int(r["n_nodes"]) for r in rr],
                  [float(r["anisotropy_median"]) for r in rr],
                  "o-", color=col, lw=1.2, ms=3, label=f"p={p:.2f}")
    ax.plot([64, 4096], [.30, .30*(4096/64)**-.5], "--", color=DARK,
            lw=.8, label=r"$N^{-1/2}$")
    ax.set(xlabel="mineral nodes, N", ylabel="median directional CV",
           title="The size law persists across occupancy")
    ax.legend(ncol=3, fontsize=5.1, loc="upper right", columnspacing=.7,
              handlelength=1.35)
    ax.grid(True, which="both", alpha=.20)
    ex = [float(r["anisotropy_size_exponent"])
          for r in analysis["size_decay"]]
    ax.text(.97, .08, f"fitted exponent: {min(ex):.2f} to {max(ex):.2f}",
            transform=ax.transAxes, ha="right", color=NAVY,
            fontweight="bold", fontsize=5.7)

    # c: largest-volume isotropy distribution.
    ax = fig.add_subplot(gs[0, 9:]); panel(ax, "c", -.12, 1.05)
    rr = [r for r in size if int(r["n_nodes"]) == 4096]
    dat = [[float(r["min_over_max"]) for r in rr
            if np.isclose(float(r["p_mean"]), p)] for p in ps]
    ax.boxplot(dat, positions=range(len(ps)), widths=.58,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#DCEAF2", edgecolor=NAVY),
               medianprops=dict(color=RED, linewidth=1.5),
               whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    ax.set_xticks(range(len(ps)), [f"{p:.2f}" for p in ps], rotation=35)
    ax.set(xlabel="bond occupancy, p", ylabel="weakest / strongest direction",
           ylim=(.82, 1.01), title="Large networks approach isotropy")
    ax.axhline(1, color=GREY, ls="--", lw=.8)

    # d: connectivity control as a dense occupancy--size heatmap.
    ax = fig.add_subplot(gs[1, :4]); panel(ax, "d")
    ps_d = sorted({float(r["p_mean"]) for r in analysis["size_summary"]})
    ns_d = sorted({int(r["n_nodes"]) for r in analysis["size_summary"]})
    span = np.array([
        [next(float(r["spanning_probability"]) for r in analysis["size_summary"]
              if np.isclose(float(r["p_mean"]), p)
              and int(r["n_nodes"]) == n) for n in ns_d]
        for p in ps_d
    ])
    im = ax.imshow(span, origin="lower", aspect="auto", cmap=CMAP_BLUE,
                   norm=Normalize(0, 1), interpolation="nearest")
    ax.set_xticks(range(len(ns_d)), [str(n) for n in ns_d])
    ax.set_yticks(range(len(ps_d)), [f"{p:.2f}" for p in ps_d])
    ax.set(xlabel="mineral nodes, N", ylabel="mean bond occupancy, p",
           title="Connectivity coverage across network size")
    for i in range(len(ps_d)):
        for j in range(len(ns_d)):
            value = span[i, j]
            ax.text(j, i, f"{100*value:.0f}%", ha="center", va="center",
                    fontsize=4.9, fontweight="bold",
                    color="white" if value >= .55 else DARK)
    for row, label in ((1, "94% → 62%"), (0, "47% → 0%")):
        ax.annotate("", xy=(5.18, row + .28), xytext=(.18, row + .28),
                    arrowprops=dict(arrowstyle="-|>", color=RED,
                                    lw=1.0, shrinkA=0, shrinkB=0))
        ax.text(2.68, row + .36, label, ha="center", va="center",
                fontsize=4.2, fontweight="bold", color=RED,
                bbox=dict(fc="white", ec="none", alpha=.78, pad=.2))
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03,
                      ticks=[0, .5, 1])
    cb.ax.set_title("P(span)", fontsize=4.7, pad=2)
    cb.ax.tick_params(labelsize=4.6, length=2)

    # e: participating-contact control, with a compact shared occupancy key.
    ax = fig.add_subplot(gs[1, 4:8]); panel(ax, "e")
    for col, p in zip(COLORS, ps):
        med = []
        for n in ns:
            q = [float(r["neff_fraction_mean"]) for r in size
                 if np.isclose(float(r["p_mean"]), p) and int(r["n_nodes"]) == n]
            med.append(np.median(q))
        ax.semilogx(ns, med, "o-", color=col, lw=1.05, ms=2.8,
                    label=f"p={p:.2f}")
    ax.set(xlabel="mineral nodes, N",
           ylabel="effective participating-contact fraction",
           title="Participation fraction vs size")
    ax.legend(ncol=3, fontsize=4.7, loc="upper right", columnspacing=.5,
              handlelength=1.2)
    ax.grid(True, which="both", alpha=.20)

    # f: density view of the weak within-design relationship.
    ax = fig.add_subplot(gs[1, 8:]); panel(ax, "f")
    xr = []; yr = []
    for n0 in ns:
        for p0 in ps:
            rr = [r for r in size if int(r["n_nodes"]) == n0 and
                  np.isclose(float(r["p_mean"]), p0)]
            xx = np.array([float(r["top1pct_energy_mean"]) for r in rr])
            yy = np.array([float(r["anisotropy_cv"]) for r in rr])
            xr.extend(xx - xx.mean()); yr.extend(yy - yy.mean())
    x = np.asarray(xr); y = np.asarray(yr)
    hb = ax.hexbin(x, y, gridsize=22, bins="log", mincnt=1,
                   cmap=CMAP_BLUE, linewidths=0)
    ax.axhline(0, color=GREY, lw=.7); ax.axvline(0, color=GREY, lw=.7)
    ax.set(xlabel="within-design Δ top-1% response fraction",
           ylabel="within-design Δ CV",
           title="Response concentration is a weak predictor")
    cb = fig.colorbar(hb, ax=ax, fraction=.045, pad=.03)
    cb.set_label("log10 count", fontsize=5.2); cb.ax.tick_params(labelsize=4.8)
    corr = float(np.corrcoef(x, y)[0, 1])
    ax.text(.97, .95, f"after controlling N and p\nr = {corr:.2f}",
            transform=ax.transAxes, ha="right", va="top", color=NAVY,
            fontweight="bold", fontsize=5.5)

    stem = OUT / "Figure_7_self_averaging_closure_v3"
    save(fig, stem)
    plt.close(fig)
    print(json.dumps({"figure": str(stem), "correlation": corr}, indent=2))


if __name__ == "__main__":
    main()
