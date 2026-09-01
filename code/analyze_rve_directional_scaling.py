#!/usr/bin/env python3
"""Analyze RVE directional scaling and select full-FEM validation targets."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _paths import RESULTS


OUT = RESULTS / "topology_research" / "rve_directional_scaling"


def median(rows, key):
    return float(np.median([r[key] for r in rows]))


def quantiles(rows, key):
    x = np.asarray([r[key] for r in rows], float)
    return [float(v) for v in np.quantile(x, [0.1, 0.5, 0.9])]


def group(rows, keys):
    out = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        out.setdefault(k, []).append(r)
    return out


def slope_loglog(x, y):
    x, y = np.asarray(x), np.asarray(y)
    good = (x > 0) & (y > 0)
    if good.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(x[good]), np.log(y[good]), 1)[0])


def main():
    data = json.loads((OUT / "raw_results.json").read_text())
    rows = data["rows"]
    # The characteristic frequency is the primary inference plane.
    mid = [r for r in rows if r["omega_over_GC"] == 1.0]

    size_summary = []
    for (p, L), rr in sorted(group([r for r in mid if r["family"] == "size"], ("p_mean", "L")).items()):
        size_summary.append({
            "p_mean": p, "L": L, "n_nodes": L**3, "n": len(rr),
            "anisotropy_q10": quantiles(rr, "anisotropy_cv")[0],
            "anisotropy_median": quantiles(rr, "anisotropy_cv")[1],
            "anisotropy_q90": quantiles(rr, "anisotropy_cv")[2],
            "min_over_max_median": median(rr, "min_over_max"),
            "largest_component_median": median(rr, "largest_component_fraction"),
            "spanning_probability": float(np.mean([r["span_x"] or r["span_y"] or r["span_z"] for r in rr])),
            "neff_fraction_median": median(rr, "neff_fraction_mean"),
            "top1pct_energy_median": median(rr, "top1pct_energy_mean"),
        })

    decay = []
    for p in sorted(set(r["p_mean"] for r in size_summary)):
        rr = [r for r in size_summary if r["p_mean"] == p]
        decay.append({
            "p_mean": p,
            "anisotropy_size_exponent": slope_loglog([r["n_nodes"] for r in rr], [r["anisotropy_median"] for r in rr]),
            "L16_anisotropy_median": [r["anisotropy_median"] for r in rr if r["L"] == 16][0],
            "L16_min_over_max_median": [r["min_over_max_median"] for r in rr if r["L"] == 16][0],
        })

    phase_rows = [r for r in mid if r["L"] == 12]
    # Deduplicate isotropic rows: use size family for a=0, fabric family otherwise.
    phase_rows = [r for r in phase_rows if (r["fabric_input"] == 0 and r["family"] == "size") or r["family"] == "fabric"]
    phase_summary = []
    for (p, a), rr in sorted(group(phase_rows, ("p_mean", "fabric_input")).items()):
        phase_summary.append({
            "p_mean": p, "fabric_input": a, "n": len(rr),
            "fabric_realized_median": median(rr, "fabric_realized"),
            "anisotropy_median": median(rr, "anisotropy_cv"),
            "anisotropy_q10": quantiles(rr, "anisotropy_cv")[0],
            "anisotropy_q90": quantiles(rr, "anisotropy_cv")[2],
            "min_over_max_median": median(rr, "min_over_max"),
            "z_over_xy_median": float(np.median([r["response_z"] / max(0.5 * (r["response_x"] + r["response_y"]), 1e-30) for r in rr])),
            "largest_component_median": median(rr, "largest_component_fraction"),
            "span_x_probability": float(np.mean([r["span_x"] for r in rr])),
            "span_y_probability": float(np.mean([r["span_y"] for r in rr])),
            "span_z_probability": float(np.mean([r["span_z"] for r in rr])),
            "neff_fraction_median": median(rr, "neff_fraction_mean"),
            "top1pct_energy_median": median(rr, "top1pct_energy_mean"),
        })

    frequency_summary = []
    for (omega, fabric), rr in sorted(group([
        r for r in rows if r["L"] == 12 and r["p_mean"] == 0.45
        and ((r["fabric_input"] == 0 and r["family"] == "size") or r["family"] == "fabric")
    ], ("omega_over_GC", "fabric_input")).items()):
        frequency_summary.append({
            "omega_over_GC": omega, "fabric_input": fabric, "n": len(rr),
            "anisotropy_median": median(rr, "anisotropy_cv"),
            "min_over_max_median": median(rr, "min_over_max"),
            "z_over_xy_median": float(np.median([
                r["response_z"] / max(0.5 * (r["response_x"] + r["response_y"]), 1e-30) for r in rr
            ])),
        })

    # Relationship between poor self-averaging and directional variability.
    x = np.asarray([r["top1pct_energy_mean"] for r in mid])
    y = np.asarray([r["anisotropy_cv"] for r in mid])
    rho = float(np.corrcoef(x, y)[0, 1])

    # Select representative full-FEM target classes from the phase map, not
    # individual lattice realizations: dense isotropic, near-threshold
    # isotropic, and near-threshold fabric.
    targets = [
        {"name": "dense_isotropic", "p_mean": 0.65, "fabric_input": 0.0},
        {"name": "near_threshold_isotropic", "p_mean": 0.25, "fabric_input": 0.0},
        {"name": "near_threshold_fabric", "p_mean": 0.25, "fabric_input": 0.6},
    ]
    result = {
        "validation": data["validation"],
        "n_frequency_rows": len(rows),
        "n_networks": len(rows) // len(data["frequencies"]),
        "size_summary": size_summary,
        "size_decay": decay,
        "phase_summary": phase_summary,
        "frequency_summary": frequency_summary,
        "correlation_top1pct_energy_vs_anisotropy": rho,
        "fem_target_classes": targets,
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    for name, recs in (("size_summary.csv", size_summary), ("phase_summary.csv", phase_summary),
                       ("frequency_summary.csv", frequency_summary)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=recs[0].keys())
            w.writeheader(); w.writerows(recs)
    print(json.dumps({"networks": result["n_networks"], "correlation": rho, "decay": decay}, indent=2))


if __name__ == "__main__":
    main()
