#!/usr/bin/env python3
"""Create the missing key-result figures for the expanded JGR evidence chain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, FancyArrowPatch

from _paths import RESULTS


ROOT = RESULTS / "topology_research"
OUT = ROOT / "jgr_topology_submission" / "figures"

NAVY = "#1F5A85"; BLUE = "#3D83B5"; TEAL = "#2B978A"
ORANGE = "#D77835"; RED = "#BD5149"; DARK = "#30363A"
GREY = "#879198"; LIGHT = "#EEF3F5"; GRID = "#D8DEE2"
COLORS = ["#A8C8D9", "#5D9FC2", NAVY, ORANGE, RED, "#7F5A8F"]
CMAP_BLUE = LinearSegmentedColormap.from_list("blue", ["#F7FAFB", "#B5D4E3", NAVY])
CMAP_ORANGE = LinearSegmentedColormap.from_list("orange", ["#FCF8F4", "#EAB88E", "#A94A2F"])

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
    "axes.linewidth": 0.65, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def panel(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")


def selected_phase_cases(data):
    targets = [0.0, 0.0795774715, 0.3978873577, 0.7957747155, 1.9894367886, 397.8873577]
    summaries = data["summaries"]
    chosen = []
    for target in targets:
        chosen.append(min(summaries, key=lambda r: abs(float(r["G_over_omega0CA"]) - target)))
    # Preserve order while removing any nearest-neighbour duplicate.
    out = []
    for r in chosen:
        if r["case_id"] not in [q["case_id"] for q in out]: out.append(r)
    return out


def figure2():
    phase = json.loads((ROOT / "contact_phase_diagram/phase_diagram.json").read_text())
    mapping = json.loads((ROOT / "contact_geometry_mapping/mapping.json").read_text())
    neck = json.loads((ROOT / "explicit_neck_validation.json").read_text())
    conv = json.loads((ROOT / "explicit_neck_convergence.json").read_text())
    cases = selected_phase_cases(phase)
    rows = phase["rows"]
    disconnected = next(c for c in cases if float(c["G_over_omega0CA"]) == 0)

    fig = plt.figure(figsize=(183/25.4, 184/25.4))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.18, 1.05, 1.0],
                          left=.06, right=.985, bottom=.07, top=.97,
                          hspace=.55, wspace=1.15)

    ax = fig.add_subplot(gs[0, :7]); panel(ax, "a")
    for col, case in zip(COLORS, cases):
        rr = sorted([r for r in rows if r["case_id"] == case["case_id"]],
                    key=lambda r: float(r["frequency_over_f0"]))
        f = np.array([float(r["frequency_over_f0"]) for r in rr])
        y = np.array([float(r["sigma_im_S_m"]) for r in rr])
        g = float(case["G_over_omega0CA"])
        label = "disconnected" if g == 0 else ("near-equipotential" if g > 100 else rf"$\gamma={g:.2g}$")
        ax.semilogx(f, y*1e3, "o-", ms=2.8, lw=1.2, color=col, label=label)
    ax.axvspan(.1, 3, color="#F7E4D6", alpha=.35, lw=0)
    ax.set(xlabel=r"normalized frequency, $f/f_0$", ylabel=r"$\sigma''$ (mS m$^{-1}$)",
           title="Finite contact conductance reshapes the full SIP spectrum")
    ax.legend(ncol=3, fontsize=5.5, loc="best")
    ax.grid(True, which="both", alpha=.22)

    ax = fig.add_subplot(gs[0, 7:]); panel(ax, "b")
    for col, case in zip(COLORS, cases):
        rr = sorted([r for r in rows if r["case_id"] == case["case_id"]],
                    key=lambda r: float(r["frequency_over_f0"]))
        re = np.array([float(r["sigma_re_S_m"]) for r in rr])
        im = np.array([float(r["sigma_im_S_m"]) for r in rr])
        ax.plot((re-re[0])*1e3, im*1e3, "o-", ms=2.5, lw=1.05, color=col)
    ax.set(xlabel=r"$\sigma'-\sigma'_{low}$ (mS m$^{-1}$)",
           ylabel=r"$\sigma''$ (mS m$^{-1}$)", title="Complex-plane trajectories are not simple interpolations")
    ax.grid(True, alpha=.22)

    ax = fig.add_subplot(gs[1, :4]); panel(ax, "c")
    ss = [r for r in phase["summaries"] if float(r["G_over_omega0CA"]) > 0]
    g = np.array([float(r["G_over_omega0CA"]) for r in ss])
    fp = np.array([float(r["fp_interp_hz"]) for r in ss])/phase["normalization"]["f0_hz"]
    amp = np.array([float(r["peak_interp"]) for r in ss])/float(disconnected["peak_interp"])
    width = np.array([float(r["fwhm_decades"]) for r in ss])
    ax.semilogx(g, fp, "o-", color=NAVY, ms=3, label=r"peak $f/f_0$")
    ax.semilogx(g, amp, "s-", color=ORANGE, ms=3, label="peak/disconnected")
    ax.semilogx(g, width, "^-", color=TEAL, ms=3, label="FWHM (decades)")
    ax.axvspan(.1, 3, color="#F7E4D6", alpha=.55, lw=0)
    ax.set(xlabel=r"contact ratio, $\gamma$", ylabel="normalized descriptor",
           title="Spectral descriptors are non-monotonic")
    ax.legend(fontsize=5.3); ax.grid(True, which="both", alpha=.22)

    finite = [r for r in rows if np.isfinite(float(r["G_over_omega0CA"])) and float(r["G_over_omega0CA"]) > 0]
    gammas = sorted({float(r["G_over_omega0CA"]) for r in finite})
    freqs = sorted({float(r["frequency_over_f0"]) for r in finite})
    V = np.zeros((len(freqs), len(gammas))); P = np.zeros_like(V)
    for j, gg in enumerate(gammas):
        rr = sorted([r for r in finite if np.isclose(float(r["G_over_omega0CA"]), gg)],
                    key=lambda r: float(r["frequency_over_f0"]))
        V[:,j] = [float(r["component_voltage_difference_V"]) for r in rr]
        P[:,j] = [float(r["contact_dissipation_W"]) for r in rr]
    extent=[np.log10(gammas[0]),np.log10(gammas[-1]),np.log10(freqs[0]),np.log10(freqs[-1])]
    ax = fig.add_subplot(gs[1, 4:8]); panel(ax, "d")
    im=ax.imshow(V, origin="lower", aspect="auto", extent=extent, cmap=CMAP_BLUE)
    ax.set(xlabel=r"$\log_{10}\gamma$", ylabel=r"$\log_{10}(f/f_0)$",
           title="Contact voltage collapses")
    cb=fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.set_label(r"$|V_{m,1}-V_{m,2}|$ (V)", fontsize=5.5)

    ax = fig.add_subplot(gs[1, 8:]); panel(ax, "e")
    pnorm=P/max(P.max(),1e-30)
    im=ax.imshow(pnorm, origin="lower", aspect="auto", extent=extent, cmap=CMAP_ORANGE, vmin=0, vmax=1)
    ax.set(xlabel=r"$\log_{10}\gamma$", ylabel=r"$\log_{10}(f/f_0)$",
           title="Dissipation localizes the transition")
    cb=fig.colorbar(im, ax=ax, fraction=.045, pad=.03); cb.set_label("normalized contact dissipation", fontsize=5.5)

    ax = fig.add_subplot(gs[2, :7]); panel(ax, "f")
    sigmas = sorted({float(r["neck_conductivity_S_m"]) for r in mapping})
    lengths = sorted({float(r["neck_length_m"]) for r in mapping})
    marks=["o","s","^","D","v"]
    for col,sigma in zip(COLORS[1:],sigmas):
        for mark,L in zip(marks,lengths):
            rr=sorted([r for r in mapping if np.isclose(float(r["neck_conductivity_S_m"]),sigma) and np.isclose(float(r["neck_length_m"]),L)],key=lambda r:float(r["target_G_S"]))
            ax.loglog([float(r["target_G_S"]) for r in rr],[float(r["required_radius_um"]) for r in rr],marker=mark,color=col,lw=1,ms=3,label=rf"$\sigma_n={sigma:g}$ S/m, $L_n={L*1e3:g}$ mm")
    ax.set(xlabel=r"target contact conductance, $G_c$ (S)", ylabel=r"equivalent neck radius, $r_n$ ($\mu$m)",
           title="Contact window mapped to conductive-neck geometry")
    ax.legend(ncol=3,fontsize=4.8); ax.grid(True,which="both",alpha=.20)
    ax.text(.02,.05,f"21 explicit-neck solves: maximum conductance error {100*neck['max_relative_error']:.2f}%",
            transform=ax.transAxes,color=NAVY,fontweight="bold")

    ax = fig.add_subplot(gs[2, 7:]); panel(ax, "g")
    rr=conv["rows"]; x=np.array([float(r["radius_over_h"]) for r in rr]); err=100*np.array([float(r["relative_error"]) for r in rr])
    ax.plot(x,err,"o-",color=RED,lw=1.3,ms=4)
    for xx,yy,r in zip(x,err,rr): ax.text(xx,yy+.10,f"{int(r['nodes']):,}",ha="center",fontsize=5,color=GREY)
    ax.set(xlabel=r"neck resolution, $r_n/h$",ylabel="conductance error (%)",
           title="Explicit-neck solution converges")
    ax.text(.97,.93,rf"$G_c=\sigma_n\pi r_n^2/L_n$"+f"\nfinest error {err[-1]:.2f}%",transform=ax.transAxes,ha="right",va="top",fontweight="bold")
    ax.grid(True,alpha=.22)
    save(fig, OUT/"Figure_2_contact_spectral_anatomy"); plt.close(fig)


def chain_schematic(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    pts=np.array([[.17,.50],[.50,.50],[.83,.50]])
    ax.plot(pts[:,0],pts[:,1],color=RED,lw=3,solid_capstyle="round")
    for x,y in pts: ax.add_patch(Circle((x,y),.085,fc="#A9D0DF",ec=NAVY,lw=1))
    ax.add_patch(FancyArrowPatch((.15,.18),(.82,.18),arrowstyle="-|>",mutation_scale=8,color=DARK,lw=1.1))
    ax.text(.49,.06,"electric field",ha="center",fontsize=5.8)
    ax.text(.49,.78,r"rotate chain by $\theta$",ha="center",fontweight="bold")


def branch_schematic(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    pts=np.array([[.18,.35],[.48,.35],[.80,.35],[.48,.76]])
    edges=[(0,1),(1,2),(1,3)]
    for i,j in edges: ax.plot(pts[[i,j],0],pts[[i,j],1],color=RED if (i,j)==(1,3) else GREY,lw=3 if (i,j)==(1,3) else 1.4)
    ax.scatter(pts[:,0],pts[:,1],s=95,c=TEAL,edgecolor="white",linewidth=.6,zorder=3)
    ax.text(.62,.72,"tested branch",color=RED,fontweight="bold")
    ax.text(.50,.08,"0° and 90° calibrate; intermediate angles are held out",ha="center",fontsize=5.5)


def figure3():
    angle=json.loads((ROOT/"angle_sweep/projection_test.json").read_text())
    branch=json.loads((ROOT/"branched_network/tensor_cross_validation.json").read_text())
    contrast=json.loads((ROOT/"branched_network/topology_contrast_analysis.json").read_text())
    modal=json.loads((ROOT/"modal_predictor/three_particle_cross_validation.json").read_text())

    fig=plt.figure(figsize=(183/25.4,186/25.4))
    gs=fig.add_gridspec(3,12,height_ratios=[.58,1.18,1.08],left=.06,right=.985,bottom=.07,top=.97,hspace=.58,wspace=1.15)
    ax=fig.add_subplot(gs[0,:5]);panel(ax,"a",-.02,.94);chain_schematic(ax)
    ax=fig.add_subplot(gs[0,5:]);panel(ax,"d",-.02,.94);branch_schematic(ax)

    ax=fig.add_subplot(gs[1,:5]);panel(ax,"b")
    angles=sorted({float(r["angle_deg"]) for r in angle["rows"]})
    for col,a in zip(COLORS,angles):
        rr=sorted([r for r in angle["rows"] if float(r["angle_deg"])==a],key=lambda r:float(r["frequency_over_f0"]))
        f=np.array([float(r["frequency_over_f0"]) for r in rr]); y=np.array([abs(complex(float(r["delta_re"]),float(r["delta_im"]))) for r in rr])
        ax.loglog(f,y,"o-",color=col,lw=1.2,ms=3,label=f"{a:g}°")
        ax.set(xlabel=r"$f/f_0$",ylabel=r"$|\Delta\sigma_c|$ (S m$^{-1}$)",title="Rotation suppresses the entire contact spectrum")
    ax.legend(ncol=3,fontsize=5.5);ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[1,5:8]);panel(ax,"c",-.12,1.05)
    freqs=sorted({float(r["frequency_over_f0"]) for r in angle["rows"]})
    E=np.zeros((len(freqs),len(angles)))
    for i,f in enumerate(freqs):
        for j,a in enumerate(angles):
            q=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f) and np.isclose(float(r["angle_deg"]),a))
            q0=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f) and np.isclose(float(r["angle_deg"]),0.0))
            den=max(abs(complex(float(q0["delta_re"]),float(q0["delta_im"]))),1e-20)
            E[i,j]=100*float(q["prediction_error"])/den
    im=ax.imshow(E,origin="lower",aspect="auto",cmap=CMAP_BLUE,vmin=0,vmax=max(1,E.max()))
    ax.set_xticks(range(len(angles)),[f"{a:g}°" for a in angles]);ax.set_yticks(range(len(freqs)),[f"{f:g}" for f in freqs])
    ax.set(xlabel="chain angle",ylabel=r"$f/f_0$",title=r"$\cos^2\theta$ residual / parallel signal (%)")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):ax.text(j,i,f"{E[i,j]:.1f}",ha="center",va="center",fontsize=5.2,color="white" if E[i,j]>.6*E.max() else DARK)
    for sp in ax.spines.values():sp.set_visible(False)

    ax=fig.add_subplot(gs[1,8:]);panel(ax,"e",-.12,1.05)
    held=branch["rows"]
    actual=np.array([abs(complex(float(r["actual_delta_re"]),float(r["actual_delta_im"]))) for r in held])
    pred=np.array([abs(complex(float(r["predicted_delta_re"]),float(r["predicted_delta_im"]))) for r in held])
    c=np.array([float(r["angle_deg"]) for r in held])
    sc=ax.scatter(actual*1e3,pred*1e3,c=c,cmap="viridis",s=25,edgecolor="white",linewidth=.4)
    lo=min(actual.min(),pred.min())*1e3;hi=max(actual.max(),pred.max())*1e3;ax.plot([lo,hi],[lo,hi],"--",color=GREY,lw=.8)
    ax.set(xlabel="held-out full FEM (mS/m)",ylabel="endpoint prediction (mS/m)",title="Held-out branch spectra")
    cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.03);cb.set_label("angle (°)",fontsize=5.5)
    ax.text(.04,.95,f"median complex error\n{100*branch['median_held_out_relative_l2_error']:.2f}%",transform=ax.transAxes,va="top",color=NAVY,fontweight="bold")

    ax=fig.add_subplot(gs[2,:5]);panel(ax,"f")
    for col,a in zip((NAVY,RED),(0.,90.)):
        rr=sorted([r for r in contrast["rows"] if np.isclose(float(r["angle_deg"]),a)],key=lambda r:float(r["frequency_over_f0"]))
        ax.loglog([float(r["frequency_over_f0"]) for r in rr],[float(r["contrast_over_full_contact_increment"]) for r in rr],"o-",color=col,lw=1.3,ms=3.5,label=f"network orientation {a:g}°")
    ax.set(xlabel=r"$f/f_0$",ylabel="branch-deletion signal / full contact increment",title="One direction misses an essential branch")
    ax.legend(fontsize=5.5);ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[2,5:9]);panel(ax,"g")
    order=["strong_chain","transition_chain","weak_chain","bottleneck_chain","single_edge_0p02"]
    vals=[100*float(modal["metrics"][k]["relative_l2_error"]) for k in order]
    cols=[TEAL]+[ORANGE]*4
    ax.barh(range(len(order)),vals,color=cols,height=.65)
    ax.set_yticks(range(len(order)),[x.replace("_"," ") for x in order]);ax.invert_yaxis();ax.set_xlabel("complex spectral error (%)")
    ax.set_title("Connectivity-only modes fail on held-out topology")
    for i,v in enumerate(vals):ax.text(v+.8,i,f"{v:.1f}%",va="center",fontsize=5.7)
    ax.text(.98,.05,"strong chain = calibration case",transform=ax.transAxes,ha="right",fontsize=5.3,color=TEAL)

    ax=fig.add_subplot(gs[2,9:]);panel(ax,"h",-.10,1.05);ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.02,.98,"What closes here?",fontweight="bold",fontsize=7.3,va="top")
    cards=[("STRAIGHT CHAIN",r"field selection follows $\cos^2\theta$",NAVY),("BRANCHED NETWORK","endpoint law predicts held-out angles",TEAL),("TOPOLOGY LIMIT","connectivity alone misses spectra",ORANGE)]
    for k,(h,t,c0) in enumerate(cards):
        y=.76-k*.28;ax.add_patch(plt.Rectangle((.02,y-.14),.96,.20,fc=LIGHT,ec="none"));ax.add_patch(plt.Rectangle((.02,y-.14),.02,.20,fc=c0,ec="none"));ax.text(.08,y+.005,h,color=c0,fontweight="bold",fontsize=5.2);ax.text(.08,y-.075,t,fontsize=5.5,va="center")
    save(fig,OUT/"Figure_4_direction_topology_mechanism");plt.close(fig)


def figure3_clean():
    """Create a less crowded adopted layout while retaining panels a--g."""
    angle=json.loads((ROOT/"angle_sweep/projection_test.json").read_text())
    branch=json.loads((ROOT/"branched_network/tensor_cross_validation.json").read_text())
    contrast=json.loads((ROOT/"branched_network/topology_contrast_analysis.json").read_text())
    modal=json.loads((ROOT/"modal_predictor/three_particle_cross_validation.json").read_text())

    fig=plt.figure(figsize=(183/25.4,134/25.4))
    gs=fig.add_gridspec(2,4,height_ratios=[.62,1.0],left=.065,right=.985,bottom=.12,top=.87,hspace=.72,wspace=.62)

    # Top row: experiment design, continuous rotation, and residual.
    ax=fig.add_subplot(gs[0,0]);panel(ax,"a",-.04,.96);chain_schematic(ax)
    ax.set_title("Straight-chain test",fontsize=7.1,pad=3)

    ax=fig.add_subplot(gs[0,1:3]);panel(ax,"b")
    angles=sorted({float(r["angle_deg"]) for r in angle["rows"]})
    for col,a in zip(COLORS,angles):
        rr=sorted([r for r in angle["rows"] if float(r["angle_deg"])==a],key=lambda r:float(r["frequency_over_f0"]))
        f=np.array([float(r["frequency_over_f0"]) for r in rr]); y=np.array([abs(complex(float(r["delta_re"]),float(r["delta_im"]))) for r in rr])
        ax.loglog(f,y,"o-",color=col,lw=1.2,ms=3,label=f"{a:g}°")
    ax.set(xlabel=r"$f/f_0$",ylabel=r"$|\Delta\sigma_c|$ (S m$^{-1}$)",title="Angle suppresses the contact spectrum")
    ax.legend(ncol=5,fontsize=5.1,loc="lower left",columnspacing=.7,handlelength=1.3)
    ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[0,3]);panel(ax,"c",-.12,1.04)
    freqs=sorted({float(r["frequency_over_f0"]) for r in angle["rows"]})
    E=np.zeros((len(freqs),len(angles)))
    for i,f0 in enumerate(freqs):
        for j,a in enumerate(angles):
            q=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f0) and np.isclose(float(r["angle_deg"]),a))
            q0=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f0) and np.isclose(float(r["angle_deg"]),0.0))
            den=max(abs(complex(float(q0["delta_re"]),float(q0["delta_im"]))),1e-20)
            E[i,j]=100*float(q["prediction_error"])/den
    im=ax.imshow(E,origin="lower",aspect="auto",cmap=CMAP_BLUE,vmin=0,vmax=max(1,E.max()))
    ax.set_xticks(range(len(angles)),[f"{a:g}°" for a in angles],fontsize=5.2)
    ax.set_yticks(range(len(freqs)),[f"{f0:.3g}" for f0 in freqs],fontsize=5.2)
    ax.set(xlabel="chain angle",ylabel=r"$f/f_0$",title=r"$\cos^2\theta$ residual (%)")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j,i,f"{E[i,j]:.1f}",ha="center",va="center",fontsize=5.0,color="white" if E[i,j]>.6*E.max() else DARK)
    for sp in ax.spines.values():sp.set_visible(False)

    # Bottom row: branched-network test, visibility contrast, and negative control.
    ax=fig.add_subplot(gs[1,0]);panel(ax,"d",-.04,1.04);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    pts=np.array([[.16,.35],[.50,.35],[.84,.35],[.50,.75]])
    for i,j in [(0,1),(1,2)]: ax.plot(pts[[i,j],0],pts[[i,j],1],color=GREY,lw=1.5)
    ax.plot(pts[[1,3],0],pts[[1,3],1],color=RED,lw=3)
    ax.scatter(pts[:,0],pts[:,1],s=75,c=TEAL,edgecolor="white",linewidth=.6,zorder=3)
    ax.text(.50,.91,"T-network test",ha="center",fontweight="bold",fontsize=7.0)
    ax.text(.50,.09,"0°/90° calibrate;\nintermediate angles held out",ha="center",fontsize=5.2)
    ax.text(.68,.73,"tested branch",color=RED,fontweight="bold",fontsize=5.0)

    ax=fig.add_subplot(gs[1,1]);panel(ax,"e",-.04,1.08)
    held=branch["rows"]
    actual=np.array([abs(complex(float(r["actual_delta_re"]),float(r["actual_delta_im"]))) for r in held])
    pred=np.array([abs(complex(float(r["predicted_delta_re"]),float(r["predicted_delta_im"]))) for r in held])
    cc=np.array([float(r["angle_deg"]) for r in held])
    sc=ax.scatter(actual*1e3,pred*1e3,c=cc,cmap="viridis",s=22,edgecolor="white",linewidth=.35)
    lo=min(actual.min(),pred.min())*1e3;hi=max(actual.max(),pred.max())*1e3
    ax.plot([lo,hi],[lo,hi],"--",color=GREY,lw=.8)
    ax.set(xlabel="full FEM (mS/m)",ylabel="prediction (mS/m)",title="Held-out T-network angles")
    cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.03);cb.set_label("angle (°)",fontsize=5.0)
    ax.text(.04,.95,f"median error\n{100*branch['median_held_out_relative_l2_error']:.2f}%",transform=ax.transAxes,va="top",color=NAVY,fontweight="bold",fontsize=5.4)

    ax=fig.add_subplot(gs[1,2]);panel(ax,"f",-.12,1.04)
    for col,a in zip((NAVY,RED),(0.,90.)):
        rr=sorted([r for r in contrast["rows"] if np.isclose(float(r["angle_deg"]),a)],key=lambda r:float(r["frequency_over_f0"]))
        ax.loglog([float(r["frequency_over_f0"]) for r in rr],[float(r["contrast_over_full_contact_increment"]) for r in rr],"o-",color=col,lw=1.2,ms=3,label=f"{a:g}°")
    ax.set(xlabel=r"$f/f_0$",ylabel="deletion / full",title="Branch visibility")
    ax.legend(fontsize=5.2,loc="lower left",title="orientation",title_fontsize=5.0)
    ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[1,3]);panel(ax,"g",-.12,1.04)
    order=["strong_chain","transition_chain","weak_chain","bottleneck_chain","single_edge_0p02"]
    vals=[100*float(modal["metrics"][k]["relative_l2_error"]) for k in order]
    cols=[TEAL]+[ORANGE]*4
    ax.barh(range(len(order)),vals,color=cols,height=.60)
    ax.set_yticks(range(len(order)),["calibration","transition","weak","bottleneck","single edge"],fontsize=5.0)
    ax.invert_yaxis();ax.set_xlabel("complex error (%)",fontsize=5.5);ax.set_title("Connectivity-only error",fontsize=7.0,pad=3)
    ax.tick_params(axis="x",labelsize=5.0)
    for i,v in enumerate(vals):ax.text(v+1.0,i,f"{v:.1f}",va="center",fontsize=5.0)
    ax.grid(True,axis="x",alpha=.2)

    fig.suptitle("Field direction controls contact observability",fontsize=9.0,fontweight="bold",y=.965)
    save(fig,OUT/"Figure_4_direction_topology_mechanism_clean_v2");plt.close(fig)


def figure3_compact(heatmap=False):
    """Compact six-panel version; optionally use a frequency-resolved error heatmap."""
    angle=json.loads((ROOT/"angle_sweep/projection_test.json").read_text())
    branch=json.loads((ROOT/"branched_network/tensor_cross_validation.json").read_text())
    contrast=json.loads((ROOT/"branched_network/topology_contrast_analysis.json").read_text())
    modal=json.loads((ROOT/"modal_predictor/three_particle_cross_validation.json").read_text())

    fig=plt.figure(figsize=(183/25.4,126/25.4))
    gs=fig.add_gridspec(2,3,height_ratios=[.78,1.0],left=.065,right=.985,bottom=.13,top=.87,hspace=.74,wspace=.68)

    # Unified design panel: the two geometries are different tests of the same
    # field-selection mechanism, not unrelated particle models.
    ax=fig.add_subplot(gs[0,0]);panel(ax,"a",-.04,1.04);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(.25,.96,"3-particle\nchain",ha="center",va="top",fontweight="bold",fontsize=5.8,linespacing=.9)
    pts=np.array([[.08,.58],[.25,.58],[.42,.58]])
    ax.plot(pts[:,0],pts[:,1],color=RED,lw=2.4,solid_capstyle="round")
    ax.scatter(pts[:,0],pts[:,1],s=52,c="#A9D0DF",edgecolor=NAVY,linewidth=.8,zorder=3)
    ax.add_patch(FancyArrowPatch((.10,.37),(.40,.37),arrowstyle="-|>",mutation_scale=7,color=DARK,lw=.8))
    ax.text(.25,.28,"rotate relative to field",ha="center",fontsize=4.8)
    ax.text(.76,.96,"4-particle\nT network",ha="center",va="top",fontweight="bold",fontsize=5.8,linespacing=.9)
    pts=np.array([[.59,.58],[.76,.58],[.93,.58],[.76,.83]])
    ax.plot(pts[[0,1],0],pts[[0,1],1],color=GREY,lw=1.1)
    ax.plot(pts[[1,2],0],pts[[1,2],1],color=GREY,lw=1.1)
    ax.plot(pts[[1,3],0],pts[[1,3],1],color=RED,lw=2.2)
    ax.scatter(pts[:,0],pts[:,1],s=52,c="#A9D0DF",edgecolor=NAVY,linewidth=.6,zorder=3)
    ax.text(.76,.28,"0°/90° calibrate;\nintermediate angles held out",ha="center",fontsize=4.8)
    ax.text(.76,.72,"tested branch",ha="left",color=RED,fontweight="bold",fontsize=4.5)

    # Straight-chain spectrum and projection residual.
    ax=fig.add_subplot(gs[0,1]);panel(ax,"b",-.20,1.14)
    angles=sorted({float(r["angle_deg"]) for r in angle["rows"]})
    for col,a in zip(COLORS,angles):
        rr=sorted([r for r in angle["rows"] if float(r["angle_deg"])==a],key=lambda r:float(r["frequency_over_f0"]))
        f=np.array([float(r["frequency_over_f0"]) for r in rr]); y=np.array([abs(complex(float(r["delta_re"]),float(r["delta_im"]))) for r in rr])
        ax.loglog(f,y,"o-",color=col,lw=1.0,ms=2.5,label=f"{a:g}°")
    ax.set(xlabel=r"$f/f_0$",ylabel=r"$|\Delta\sigma_c|$ (S m$^{-1}$)")
    ax.set_title("Angle suppresses the contact spectrum",pad=10)
    ax.legend(ncol=3,fontsize=4.5,loc="lower left",columnspacing=.45,handlelength=1.1)
    ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[0,2]);panel(ax,"c",-.12,1.04)
    freqs=sorted({float(r["frequency_over_f0"]) for r in angle["rows"]})
    E=np.zeros((len(freqs),len(angles)))
    for i,f0 in enumerate(freqs):
        for j,a in enumerate(angles):
            q=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f0) and np.isclose(float(r["angle_deg"]),a))
            q0=next(r for r in angle["rows"] if np.isclose(float(r["frequency_over_f0"]),f0) and np.isclose(float(r["angle_deg"]),0.0))
            den=max(abs(complex(float(q0["delta_re"]),float(q0["delta_im"]))),1e-20)
            E[i,j]=100*float(q["prediction_error"])/den
    im=ax.imshow(E,origin="lower",aspect="auto",cmap=CMAP_BLUE,vmin=0,vmax=.8)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.03,ticks=[0,.4,.8])
    cb.set_label("error (%)",fontsize=4.6,labelpad=2)
    cb.ax.tick_params(labelsize=4.5,length=2)
    ax.set_xticks(range(len(angles)),[f"{a:g}°" for a in angles],fontsize=4.8)
    ax.set_yticks(range(len(freqs)),[f"{f0:.3g}" for f0 in freqs],fontsize=4.8)
    ax.set(xlabel="chain angle",ylabel=r"$f/f_0$",title=r"$\cos^2\theta$ residual (%)")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j,i,f"{E[i,j]:.1f}",ha="center",va="center",fontsize=4.6,color="white" if E[i,j]>.6*E.max() else DARK)
    for sp in ax.spines.values():sp.set_visible(False)

    # Held-out T-network prediction, branch visibility, and graph-only negative control.
    ax=fig.add_subplot(gs[1,0]);panel(ax,"d",-.10,1.04)
    held=branch["rows"]
    actual=np.array([abs(complex(float(r["actual_delta_re"]),float(r["actual_delta_im"]))) for r in held])
    pred=np.array([abs(complex(float(r["predicted_delta_re"]),float(r["predicted_delta_im"]))) for r in held])
    cc=np.array([float(r["angle_deg"]) for r in held])
    sc=ax.scatter(actual*1e3,pred*1e3,c=cc,cmap="viridis",s=19,edgecolor="white",linewidth=.3)
    lo=min(actual.min(),pred.min())*1e3;hi=max(actual.max(),pred.max())*1e3
    ax.plot([lo,hi],[lo,hi],"--",color=GREY,lw=.75)
    ax.set(xlabel="full FEM (mS/m)",ylabel="prediction (mS/m)",title="Held-out T-network angles")
    cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.03);cb.set_label("angle (°)",fontsize=4.8)
    ax.text(.04,.95,f"median error\n{100*branch['median_held_out_relative_l2_error']:.2f}%",transform=ax.transAxes,va="top",color=NAVY,fontweight="bold",fontsize=5.0)

    ax=fig.add_subplot(gs[1,1]);panel(ax,"e",-.10,1.04)
    for col,a in zip((NAVY,RED),(0.,90.)):
        rr=sorted([r for r in contrast["rows"] if np.isclose(float(r["angle_deg"]),a)],key=lambda r:float(r["frequency_over_f0"]))
        ax.loglog([float(r["frequency_over_f0"]) for r in rr],[float(r["contrast_over_full_contact_increment"]) for r in rr],"o-",color=col,lw=1.15,ms=2.8,label=f"{a:g}°")
    ax.set(xlabel=r"$f/f_0$",ylabel="deletion / full",title="Branch visibility")
    ax.legend(fontsize=4.8,loc="lower left",title="orientation",title_fontsize=4.8)
    ax.grid(True,which="both",alpha=.22)

    ax=fig.add_subplot(gs[1,2]);panel(ax,"f",-.10,1.04)
    if heatmap:
        order=["transition_chain","weak_chain","bottleneck_chain","single_edge_0p02"]
        labels=["transition","weak","bottleneck","single edge"]
        rows=modal["rows"]
        freq_hz=sorted({float(r["frequency_hz"]) for r in rows})
        f0_hz=min(freq_hz)/0.1
        f_rel=np.asarray(freq_hz)/f0_hz
        H=np.zeros((len(order),len(freq_hz)))
        for i,case in enumerate(order):
            for j,fhz in enumerate(freq_hz):
                r=next(r for r in rows if r["case_id"]==case and np.isclose(float(r["frequency_hz"]),fhz))
                actual=complex(float(r["actual_delta_re"]),float(r["actual_delta_im"]))
                predicted=complex(float(r["predicted_delta_re"]),float(r["predicted_delta_im"]))
                H[i,j]=100*abs(predicted-actual)/max(abs(actual),1e-30)
        im=ax.imshow(H,origin="upper",aspect="auto",cmap=CMAP_ORANGE,vmin=0,vmax=85)
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.03,ticks=[0,40,80])
        cb.set_label("complex error (%)",fontsize=4.6,labelpad=2)
        cb.ax.tick_params(labelsize=4.5,length=2)
        ax.set_xticks(range(len(f_rel)),[f"{x:.3g}" for x in f_rel],rotation=45,ha="right",fontsize=4.4)
        ax.set_yticks(range(len(order)),labels,fontsize=4.8)
        ax.set_xlabel(r"$f/f_0$",fontsize=5.2);ax.set_title("Frequency-resolved error",fontsize=6.8,pad=3)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j,i,f"{H[i,j]:.0f}",ha="center",va="center",fontsize=4.1,color="white" if H[i,j]>45 else DARK)
    else:
        order=["strong_chain","transition_chain","weak_chain","bottleneck_chain","single_edge_0p02"]
        vals=[100*float(modal["metrics"][k]["relative_l2_error"]) for k in order]
        cols=[TEAL]+[ORANGE]*4
        ax.barh(range(len(order)),vals,color=cols,height=.58)
        ax.set_yticks(range(len(order)),["calibration","transition","weak","bottleneck","single edge"],fontsize=4.8)
        ax.invert_yaxis();ax.set_xlabel("complex error (%)",fontsize=5.2);ax.set_title("Connectivity-only error",fontsize=6.8,pad=3)
        ax.tick_params(axis="x",labelsize=4.8)
        for i,v in enumerate(vals):ax.text(v+1.0,i,f"{v:.1f}",va="center",fontsize=4.8)
        ax.grid(True,axis="x",alpha=.2)

    fig.suptitle("Field direction controls contact observability",fontsize=8.8,fontweight="bold",y=.965)
    stem="Figure_4_direction_topology_mechanism_compact_v4_heatmap" if heatmap else "Figure_4_direction_topology_mechanism_compact_v3"
    save(fig,OUT/stem);plt.close(fig)


def figure6():
    raw=json.loads((ROOT/"rve_directional_scaling/raw_results.json").read_text())["rows"]
    analysis=json.loads((ROOT/"rve_directional_scaling/analysis.json").read_text())
    size=[r for r in raw if r["family"]=="size" and np.isclose(float(r["omega_over_GC"]),1.0)]
    ps=sorted({float(r["p_mean"]) for r in size}); ns=sorted({int(r["n_nodes"]) for r in size})
    fig=plt.figure(figsize=(183/25.4,180/25.4));gs=fig.add_gridspec(2,12,left=.06,right=.985,bottom=.08,top=.96,hspace=.58,wspace=1.25)

    ax=fig.add_subplot(gs[0,:5]);panel(ax,"a")
    p0=.45;show=[64,512,4096]
    vals=[]
    for n in show: vals.append([float(r["anisotropy_cv"]) for r in size if np.isclose(float(r["p_mean"]),p0) and int(r["n_nodes"])==n])
    vp=ax.violinplot(vals,positions=range(3),showmeans=False,showmedians=True,widths=.75)
    for body,col in zip(vp["bodies"],["#A9CADC",BLUE,NAVY]):body.set_facecolor(col);body.set_edgecolor("none");body.set_alpha(.8)
    vp["cmedians"].set_color(RED)
    for i,v in enumerate(vals):ax.scatter(i+np.linspace(-.16,.16,len(v)),v,s=6,color=DARK,alpha=.25)
    ax.set_xticks(range(3),[str(n) for n in show]);ax.set(xlabel="mineral nodes, N",ylabel="directional variability, CV",title="Directional variability contracts with volume")
    ax.text(.97,.95,"p = 0.45; 32 realizations\nred = median",transform=ax.transAxes,ha="right",va="top",fontsize=5.5,color=GREY)

    ax=fig.add_subplot(gs[0,5:9]);panel(ax,"b")
    for col,p in zip(COLORS,ps):
        rr=sorted([r for r in analysis["size_summary"] if np.isclose(float(r["p_mean"]),p)],key=lambda r:int(r["n_nodes"]))
        ax.loglog([int(r["n_nodes"]) for r in rr],[float(r["anisotropy_median"]) for r in rr],"o-",color=col,lw=1.2,ms=3,label=f"p={p:.2f}")
    ax.plot([64,4096],[.30,.30*(4096/64)**-.5],"--",color=DARK,lw=.8,label=r"$N^{-1/2}$")
    ax.set(xlabel="mineral nodes, N",ylabel="median directional CV",title="The size law persists across occupancy")
    ax.legend(ncol=2,fontsize=5.3);ax.grid(True,which="both",alpha=.20)
    ex=[float(r["anisotropy_size_exponent"]) for r in analysis["size_decay"]]
    ax.text(.97,.08,f"fitted exponent {min(ex):.2f} to {max(ex):.2f}",transform=ax.transAxes,ha="right",color=NAVY,fontweight="bold")

    ax=fig.add_subplot(gs[0,9:]);panel(ax,"c",-.12,1.05)
    rr=[r for r in size if int(r["n_nodes"])==4096]
    dat=[[float(r["min_over_max"]) for r in rr if np.isclose(float(r["p_mean"]),p)] for p in ps]
    ax.boxplot(dat,positions=range(len(ps)),widths=.6,patch_artist=True,showfliers=False,boxprops=dict(facecolor="#DCEAF2",edgecolor=NAVY),medianprops=dict(color=RED),whiskerprops=dict(color=GREY),capprops=dict(color=GREY))
    ax.set_xticks(range(len(ps)),[f"{p:.2f}" for p in ps],rotation=35);ax.set(xlabel="bond occupancy, p",ylabel="weakest / strongest direction",ylim=(.82,1.01),title="Largest RVE approaches isotropy")
    ax.axhline(1,color=GREY,ls="--",lw=.8)

    ax=fig.add_subplot(gs[1,:4]);panel(ax,"d")
    for col,p in zip(COLORS,ps):
        rr=sorted([r for r in analysis["size_summary"] if np.isclose(float(r["p_mean"]),p)],key=lambda r:int(r["n_nodes"]))
        ax.semilogx([int(r["n_nodes"]) for r in rr],[float(r["spanning_probability"]) for r in rr],"o-",color=col,lw=1.1,ms=3,label=f"p={p:.2f}")
    ax.set(xlabel="mineral nodes, N",ylabel="spanning probability",ylim=(-.03,1.03),title="Connectivity evolves over the size series")
    ax.grid(True,which="both",alpha=.20)

    ax=fig.add_subplot(gs[1,4:8]);panel(ax,"e")
    for col,p in zip(COLORS,ps):
        med=[]
        for n in ns:
            q=[float(r["neff_fraction_mean"]) for r in size if np.isclose(float(r["p_mean"]),p) and int(r["n_nodes"])==n]
            med.append(np.median(q))
        ax.semilogx(ns,med,"o-",color=col,lw=1.1,ms=3,label=f"p={p:.2f}")
    ax.set(xlabel="mineral nodes, N",ylabel="effective participating-contact fraction",title="Contacts remain unequally involved")
    ax.grid(True,which="both",alpha=.20)

    ax=fig.add_subplot(gs[1,8:]);panel(ax,"f")
    xr=[];yr=[];nr=[]
    for n0 in ns:
        for p0 in ps:
            rr=[r for r in size if int(r["n_nodes"])==n0 and np.isclose(float(r["p_mean"]),p0)]
            xx=np.array([float(r["top1pct_energy_mean"]) for r in rr]);yy=np.array([float(r["anisotropy_cv"]) for r in rr])
            xr.extend(xx-xx.mean());yr.extend(yy-yy.mean());nr.extend([n0]*len(rr))
    x=np.asarray(xr);y=np.asarray(yr);n=np.asarray(nr)
    sc=ax.scatter(x,y,c=np.log2(n),cmap=CMAP_BLUE,s=10,alpha=.55,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.7);ax.axvline(0,color=GREY,lw=.7)
    ax.set(xlabel="within-design deviation in top-1% response fraction",ylabel="within-design deviation in CV",title="Concentration is a weak within-design predictor")
    cb=fig.colorbar(sc,ax=ax,fraction=.045,pad=.03);cb.set_label(r"$\log_2 N$",fontsize=5.5)
    corr=float(np.corrcoef(x,y)[0,1])
    ax.text(.97,.95,f"after controlling N and p: r = {corr:.2f}",transform=ax.transAxes,ha="right",va="top",color=NAVY,fontweight="bold")
    save(fig,OUT/"Figure_7_self_averaging_closure");plt.close(fig)


def main():
    figure2(); figure3(); figure3_clean(); figure3_compact(); figure3_compact(heatmap=True); figure6()
    print(json.dumps({"created":["Figure_2_contact_spectral_anatomy","Figure_4_direction_topology_mechanism_compact_v3","Figure_4_direction_topology_mechanism_compact_v4_heatmap","Figure_7_self_averaging_closure"]},indent=2))


if __name__=="__main__": main()
