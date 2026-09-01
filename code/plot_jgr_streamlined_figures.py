#!/usr/bin/env python3
"""Create streamlined, non-destructive JGR Figures 1 and 2.

The original Figure_1_finite_contact_dynamics and
Figure_2_contact_spectral_anatomy files are left untouched.  This script
creates a compact main-text alternative with no repeated directional or
mesh-convergence panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from _paths import RESULTS


ROOT = RESULTS / "topology_research"
OUT = ROOT / "jgr_topology_submission" / "figures"
IMAGEGEN_SCHEMATIC = OUT / "finite_contact_model_schematic_imagegen_v5_pore_matches_interface_font.png"

NAVY = "#1F5A85"
BLUE = "#3D83B5"
PALE_BLUE = "#DCEAF2"
ORANGE = "#D77835"
PALE_ORANGE = "#F7E4D6"
RED = "#BD5149"
TEAL = "#2B9A8B"
DARK = "#2E3338"
GREY = "#879099"
GRID = "#D8DEE2"
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "contact_blue", ["#F5F8FA", "#AFCFE0", NAVY]
)
CMAP_ORANGE = LinearSegmentedColormap.from_list(
    "contact_orange", ["#FCF8F4", "#EAB88E", "#A94A2F"]
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.0,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
})


def panel(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def load_phase():
    return json.loads((ROOT / "contact_phase_diagram/phase_diagram.json").read_text())


def schematic(ax):
    panel(ax, "a", -0.03, 1.00)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.91, "Coupled mineral–electrolyte model",
            fontweight="bold", fontsize=8.2)
    ax.add_patch(Rectangle((0.04, 0.20), 0.92, 0.58,
                           fc="#F1F7FA", ec=GREY, lw=0.7))
    centres = [(0.30, 0.49), (0.70, 0.49)]
    for x, y in centres:
        ax.add_patch(Circle((x, y), 0.13, fc="#9BC7D9", ec=NAVY, lw=1.0))
        ax.add_patch(Circle((x, y), 0.155, fill=False, ec=ORANGE,
                            lw=2.0, ls=(0, (2, 1))))
    ax.plot([0.43, 0.57], [0.49, 0.49], color=RED, lw=4,
            solid_capstyle="round")
    ax.text(0.50, 0.59, r"finite electronic contact $G_c$", color=RED,
            ha="center", fontweight="bold", fontsize=7.0)
    ax.text(0.16, 0.30, "pore electrolyte", color=NAVY, ha="center")
    ax.text(0.30, 0.71, r"interface capacitance $C_0$", color=ORANGE,
            ha="center", fontsize=6.7)
    ax.text(0.30, 0.47, r"$V_{m,1}$", ha="center", va="center",
            fontweight="bold")
    ax.text(0.70, 0.47, r"$V_{m,2}$", ha="center", va="center",
            fontweight="bold")
    ax.text(0.50, 0.38, r"$\Delta V_m=V_{m,1}-V_{m,2}$", ha="center",
            va="center", color=DARK, fontsize=6.3)
    ax.add_patch(FancyArrowPatch((0.08, 0.11), (0.92, 0.11), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.2, color=DARK))
    ax.text(0.50, 0.025, "applied electric field", ha="center", color=DARK)


def imagegen_schematic(ax):
    """Embed the approved ImageGen schematic without altering its labels."""
    panel(ax, "a", -0.03, 1.00)
    image = plt.imread(IMAGEGEN_SCHEMATIC)
    ax.imshow(image, aspect="equal")
    ax.set_axis_off()


def phase_map(ax, data):
    panel(ax, "b")
    rows = [r for r in data["rows"]
            if np.isfinite(float(r["contact_conductance_S"]))]
    gammas = sorted({float(r["G_over_omega0CA"]) for r in rows
                     if float(r["G_over_omega0CA"]) > 0})
    freqs = sorted({float(r["frequency_over_f0"]) for r in rows})
    mat = np.empty((len(freqs), len(gammas)))
    for j, g in enumerate(gammas):
        vals = sorted(
            [r for r in rows if np.isclose(float(r["G_over_omega0CA"]), g)],
            key=lambda r: float(r["frequency_over_f0"]),
        )
        mat[:, j] = [float(r["contact_dissipation_W"]) for r in vals]
    mat /= max(mat.max(), 1e-30)
    x = np.log10(gammas)
    y = np.log10(freqs)
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=CMAP_BLUE,
                   vmin=0, vmax=1, extent=[x[0], x[-1], y[0], y[-1]])
    # The caption defines the normalized quantity; leaving the compact colorbar
    # unlabeled avoids collision with the neighboring descriptor-axis label.
    plt.colorbar(im, ax=ax, fraction=.045, pad=.025)
    # The broad shaded window is retained, with the strongest spectral
    # transition marked by a narrower pair of guide lines.
    ax.axvspan(np.log10(0.1), np.log10(3), color=ORANGE, alpha=0.12, lw=0)
    ax.axvline(np.log10(0.4), color=RED, lw=0.7, ls=(0, (2, 2)), alpha=.8)
    ax.axvline(np.log10(2.0), color=RED, lw=0.7, ls=(0, (2, 2)), alpha=.8)
    ax.text(np.log10(0.9), y[-1] - 0.10, "strongest transition",
            ha="center", color=RED, fontweight="bold", fontsize=6.1)
    ax.set_xlabel(r"contact ratio, $\gamma=G_c/(\omega_0 C_0 A_p)$")
    ax.set_ylabel(r"normalized frequency, $f/f_0$")
    ax.set_xticks([-2, -1, 0, 1, 2],
                  [r"$10^{-2}$", r"$10^{-1}$", "1", "10", "100"])
    ax.set_yticks([-1, 0, 1], ["0.1", "1", "10"])
    ax.set_title("Contact dissipation localizes the transition",
                 fontsize=7.3, pad=5)
    ax.text(0.98, 0.03, "dark = high dissipation", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.6, color=GREY)


def descriptors(ax, data):
    panel(ax, "c")
    summaries = data["summaries"]
    disconnected = min(summaries,
                       key=lambda r: abs(float(r["G_over_omega0CA"])))
    finite = [r for r in summaries
              if float(r["G_over_omega0CA"]) > 0]
    g = np.array([float(r["G_over_omega0CA"]) for r in finite])
    fp = np.array([float(r["fp_interp_hz"]) for r in finite])
    fp /= float(data["normalization"]["f0_hz"])
    amp = np.array([float(r["peak_interp"]) for r in finite])
    amp /= float(disconnected["peak_interp"])
    width = np.array([float(r["fwhm_decades"]) for r in finite])
    ax.semilogx(g, fp, "o-", ms=3.4, lw=1.25, color=NAVY,
                label=r"peak $f/f_0$")
    ax.semilogx(g, amp, "s-", ms=3.2, lw=1.15, color=ORANGE,
                label="peak amplitude / disconnected")
    ax.semilogx(g, width, "^-", ms=3.2, lw=1.15, color=TEAL,
                label="FWHM (decades)")
    ax.axvspan(0.1, 3, color=PALE_ORANGE, alpha=0.60, lw=0)
    ax.axvline(0.4, color=RED, lw=0.7, ls=(0, (2, 2)), alpha=.8)
    ax.axvline(2.0, color=RED, lw=0.7, ls=(0, (2, 2)), alpha=.8)
    ax.axhline(1, color=GRID, lw=0.7)
    ax.set_xlabel(r"contact ratio, $\gamma$")
    ax.set_ylabel("normalized descriptor")
    ax.set_title("Finite contacts broaden and shift the SIP spectrum",
                 fontsize=7.3, pad=5)
    ax.legend(fontsize=5.7, loc="best")
    ax.grid(True, which="both", alpha=0.22)


def figure1():
    phase = load_phase()
    fig = plt.figure(figsize=(183 / 25.4, 68 / 25.4))
    gs = fig.add_gridspec(1, 3, left=.045, right=.985, bottom=.18, top=.82,
                          width_ratios=[1.65, 1.0, 1.0], wspace=.32)
    imagegen_schematic(fig.add_subplot(gs[0, 0]))
    phase_map(fig.add_subplot(gs[0, 1]), phase)
    descriptors(fig.add_subplot(gs[0, 2]), phase)
    fig.suptitle("Finite contacts create a distinct SIP spectral regime",
                 fontsize=10, fontweight="bold", y=.965)
    save(fig, OUT / "Figure_1_finite_contact_dynamics_streamlined_v4_imagegen_a_colorbar")
    plt.close(fig)


def selected_phase_cases(data):
    targets = [0.0, 0.0795774715, 0.3978873577,
               0.7957747155, 1.9894367886, 397.8873577]
    chosen = []
    for target in targets:
        case = min(data["summaries"],
                   key=lambda r: abs(float(r["G_over_omega0CA"]) - target))
        if case["case_id"] not in [q["case_id"] for q in chosen]:
            chosen.append(case)
    return chosen


def spectra_panel(ax, phase):
    panel(ax, "a")
    rows = phase["rows"]
    cases = selected_phase_cases(phase)
    colors = ["#A8C8D9", "#5D9FC2", NAVY, ORANGE, RED, "#7F5A8F"]
    for col, case in zip(colors, cases):
        rr = sorted([r for r in rows if r["case_id"] == case["case_id"]],
                    key=lambda r: float(r["frequency_over_f0"]))
        f = np.array([float(r["frequency_over_f0"]) for r in rr])
        y = np.array([float(r["sigma_im_S_m"]) for r in rr]) * 1e3
        g = float(case["G_over_omega0CA"])
        label = ("disconnected" if g == 0 else
                 ("near-equipotential" if g > 100 else rf"$\gamma={g:.2g}$"))
        ax.semilogx(f, y, "o-", ms=2.8, lw=1.2, color=col, label=label)
    ax.axvspan(.1, 3, color=PALE_ORANGE, alpha=.30, lw=0)
    ax.set_xlabel(r"normalized frequency, $f/f_0$")
    ax.set_ylabel(r"$\sigma''$ (mS m$^{-1}$)")
    ax.set_title("Finite contacts reshape the SIP spectrum",
                 fontsize=7.2, pad=5)
    ax.legend(ncol=3, fontsize=5.6, loc="best")
    ax.grid(True, which="both", alpha=.22)


def complex_panel(ax, phase):
    panel(ax, "b")
    rows = phase["rows"]
    cases = selected_phase_cases(phase)
    colors = ["#A8C8D9", "#5D9FC2", NAVY, ORANGE, RED, "#7F5A8F"]
    for col, case in zip(colors, cases):
        rr = sorted([r for r in rows if r["case_id"] == case["case_id"]],
                    key=lambda r: float(r["frequency_over_f0"]))
        re = np.array([float(r["sigma_re_S_m"]) for r in rr])
        im = np.array([float(r["sigma_im_S_m"]) for r in rr])
        x = (re - re[0]) * 1e3
        y = im * 1e3
        ax.plot(x, y, "o-", ms=2.5, lw=1.05, color=col)
        # A small arrow makes frequency orientation explicit without adding
        # a second legend.
        if len(x) > 3:
            k = len(x) // 2
            ax.annotate("", xy=(x[k + 1], y[k + 1]), xytext=(x[k], y[k]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=.8))
    ax.set_xlabel(r"excess real conductivity, $\Delta\sigma'$ (mS m$^{-1}$)")
    ax.set_ylabel(r"$\sigma''$ (mS m$^{-1}$)")
    ax.set_title("Distinct complex-plane trajectories",
                 fontsize=7.2, pad=5)
    ax.text(.02, .04, "arrows: increasing frequency",
            transform=ax.transAxes, fontsize=5.5, color=GREY)
    ax.grid(True, alpha=.22)


def mechanism_maps(ax_v, ax_p, phase):
    rows = [r for r in phase["rows"]
            if np.isfinite(float(r["G_over_omega0CA"])) and
            float(r["G_over_omega0CA"]) > 0]
    gammas = sorted({float(r["G_over_omega0CA"]) for r in rows})
    freqs = sorted({float(r["frequency_over_f0"]) for r in rows})
    V = np.zeros((len(freqs), len(gammas)))
    P = np.zeros_like(V)
    for j, g in enumerate(gammas):
        rr = sorted([r for r in rows
                     if np.isclose(float(r["G_over_omega0CA"]), g)],
                    key=lambda r: float(r["frequency_over_f0"]))
        V[:, j] = [float(r["component_voltage_difference_V"]) for r in rr]
        P[:, j] = [float(r["contact_dissipation_W"]) for r in rr]
    extent = [np.log10(gammas[0]), np.log10(gammas[-1]),
              np.log10(freqs[0]), np.log10(freqs[-1])]

    panel(ax_v, "c")
    im = ax_v.imshow(V, origin="lower", aspect="auto", extent=extent,
                     cmap=CMAP_BLUE)
    ax_v.set_xlabel(r"$\log_{10}\gamma$")
    ax_v.set_ylabel(r"$\log_{10}(f/f_0)$")
    ax_v.set_title("Contact voltage collapses toward equipotentiality",
                   fontsize=7.2, pad=5)
    cb = plt.colorbar(im, ax=ax_v, fraction=.045, pad=.03)
    cb.set_label(r"$|\Delta V_m|$ (V)", fontsize=5.8)

    panel(ax_p, "d")
    pnorm = P / max(P.max(), 1e-30)
    im = ax_p.imshow(pnorm, origin="lower", aspect="auto", extent=extent,
                     cmap=CMAP_ORANGE, vmin=0, vmax=1)
    ax_p.set_xlabel(r"$\log_{10}\gamma$")
    ax_p.set_ylabel(r"$\log_{10}(f/f_0)$")
    ax_p.set_title("Voltage and conductance combine to localize dissipation",
                   fontsize=7.2, pad=5)
    cb = plt.colorbar(im, ax=ax_p, fraction=.045, pad=.03)
    cb.set_label("normalized contact dissipation", fontsize=5.8)


def voltage_panel(ax, phase):
    """Show the voltage-redistribution mechanism retained after Fig. 1b."""
    rows = [r for r in phase["rows"]
            if np.isfinite(float(r["G_over_omega0CA"])) and
            float(r["G_over_omega0CA"]) > 0]
    gammas = sorted({float(r["G_over_omega0CA"]) for r in rows})
    freqs = sorted({float(r["frequency_over_f0"]) for r in rows})
    V = np.zeros((len(freqs), len(gammas)))
    for j, g in enumerate(gammas):
        rr = sorted([r for r in rows
                     if np.isclose(float(r["G_over_omega0CA"]), g)],
                    key=lambda r: float(r["frequency_over_f0"]))
        V[:, j] = [float(r["component_voltage_difference_V"]) for r in rr]
    extent = [np.log10(gammas[0]), np.log10(gammas[-1]),
              np.log10(freqs[0]), np.log10(freqs[-1])]
    panel(ax, "c")
    im = ax.imshow(V, origin="lower", aspect="auto", extent=extent,
                   cmap=CMAP_BLUE)
    ax.set_xlabel(r"$\log_{10}\gamma$")
    ax.set_ylabel(r"$\log_{10}(f/f_0)$")
    ax.set_title("Contact voltage collapses toward equipotentiality",
                 fontsize=7.2, pad=5)
    cb = plt.colorbar(im, ax=ax, fraction=.045, pad=.03)
    cb.set_label(r"$|\Delta V_m|$ (V)", fontsize=5.8)


def figure2():
    phase = load_phase()
    fig = plt.figure(figsize=(183 / 25.4, 72 / 25.4))
    gs = fig.add_gridspec(1, 3, left=.055, right=.985, bottom=.20, top=.80,
                          width_ratios=[1.05, 1.05, 1.0], wspace=.48)
    spectra_panel(fig.add_subplot(gs[0, 0]), phase)
    complex_panel(fig.add_subplot(gs[0, 1]), phase)
    voltage_panel(fig.add_subplot(gs[0, 2]), phase)
    fig.suptitle("Spectral and voltage signatures of the finite-contact regime",
                 fontsize=9.2, fontweight="bold", y=.96)
    save(fig, OUT / "Figure_2_contact_spectral_anatomy_streamlined_v3_one_row")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    print("Created streamlined Figure 1 and Figure 2 in the submission figure directory.")
