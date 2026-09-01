#!/usr/bin/env python3
"""Summarize full-FEM RVE endpoint validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _paths import RESULTS


OUT = RESULTS / "topology_research" / "rve_directional_scaling"


def main():
    data = json.loads((OUT / "fem_raw.json").read_text())
    rows = data["rows"]
    idx = {(r["network"], r["measured_axis"], r["state"], r["frequency_over_f0"]): r for r in rows}
    metrics = []
    for kind in ("isotropic", "axial_fabric"):
        for f in (0.1, 1.0, 10.0):
            vals = {}
            gps = {}
            for axis in ("x", "y", "z"):
                c = idx[(kind, axis, "complete", f)]
                d = idx[(kind, axis, "disconnected", f)]
                ds = complex(c["sigma_re_S_m"] - d["sigma_re_S_m"], c["sigma_im_S_m"] - d["sigma_im_S_m"])
                vals[axis] = abs(ds)
                gps[axis] = abs(complex(c["graph_prediction_re"], c["graph_prediction_im"]))
            a = np.array(list(vals.values()))
            gp = np.array(list(gps.values()))
            metrics.append({
                "network": kind, "frequency_over_f0": f,
                "delta_sigma_x": vals["x"], "delta_sigma_y": vals["y"], "delta_sigma_z": vals["z"],
                "fem_min_over_max": float(a.min() / a.max()),
                "fem_anisotropy_cv": float(a.std() / a.mean()),
                "fem_z_over_xy": float(vals["z"] / (0.5 * (vals["x"] + vals["y"]))),
                "graph_min_over_max": float(gp.min() / gp.max()),
                "graph_z_over_xy": float(gp[2] / (0.5 * (gp[0] + gp[1]))),
            })
    max_mismatch = max(r["electrode_mismatch"] for r in rows)
    iso = [r for r in metrics if r["network"] == "isotropic"]
    fabric = [r for r in metrics if r["network"] == "axial_fabric"]
    summary = {
        "n_full_fem_solves": len(rows),
        "max_electrode_mismatch": max_mismatch,
        "isotropic_min_over_max_range": [min(r["fem_min_over_max"] for r in iso), max(r["fem_min_over_max"] for r in iso)],
        "fabric_z_over_xy_range": [min(r["fem_z_over_xy"] for r in fabric), max(r["fem_z_over_xy"] for r in fabric)],
        "metrics": metrics,
    }
    (OUT / "fem_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "fem_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
