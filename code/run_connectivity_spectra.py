#!/usr/bin/env python3
"""Frequency spectra across disconnected, finite-contact, and ideal limits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from _paths import MESHES, RESULTS
import run_connectivity_pilot as pilot
import run_scenario0 as rs


def prepare_case():
    length = 47e-3
    gap = 5e-3
    rs.LX = rs.LY = rs.LZ = length
    rs.XC = rs.YC = rs.ZC = 0.5 * length
    rs.V_RIGHT = rs.E0 * length
    x1 = 0.5 * length - 0.5 * gap - rs.R
    x2 = 0.5 * length + 0.5 * gap + rs.R
    centres = [np.array([x1, rs.YC, rs.ZC]), np.array([x2, rs.YC, rs.ZC])]
    mesh_path = MESHES / "connectivity_pilot_L47_d5.msh"
    if not mesh_path.exists():
        pilot.main()
    mesh, facets = rs.load_msh_with_tags(mesh_path, centres)
    return length, centres, mesh, facets


def main() -> None:
    length, centres, mesh, facets = prepare_case()
    frequencies = rs.FP * np.logspace(-1.0, 1.0, 9)
    # At f0, one-grain interface susceptance is about 0.0194 S.
    cases = (
        ("disconnected", [0, 1], None),
        ("contact_0p002S", [0, 1], [(0, 1, 0.002)]),
        ("contact_0p02S", [0, 1], [(0, 1, 0.02)]),
        ("contact_0p2S", [0, 1], [(0, 1, 0.2)]),
        ("ideal_connected", [0, 0], None),
    )
    rows = []
    for case_id, labels, contacts in cases:
        for frequency in frequencies:
            basis, ur, ui, potentials = rs.solve(
                mesh,
                facets,
                v_right=rs.V_RIGHT,
                centres=centres,
                vim_electrode="dirichlet",
                component_ids=labels,
                omega=2.0 * math.pi * float(frequency),
                contact_edges=contacts,
            )
            left = rs.extract_sigma_electrode(
                mesh, ur, ui, e0=rs.E0, L=length, side="left"
            )
            right = rs.extract_sigma_electrode(
                mesh, ur, ui, e0=rs.E0, L=length, side="right"
            )
            sigma = 0.5 * (left + right)
            rows.append(
                {
                    "case_id": case_id,
                    "contact_conductance_S": 0.0 if not contacts else contacts[0][2],
                    "ideal_connected": labels == [0, 0],
                    "frequency_hz": float(frequency),
                    "frequency_over_f0": float(frequency / rs.FP),
                    "sigma_re_S_m": sigma.real,
                    "sigma_im_S_m": sigma.imag,
                    "phase_deg": float(np.degrees(np.angle(sigma))),
                    "n_components": len(potentials),
                    "left_right_relative_mismatch": abs(left - right)
                    / max(abs(sigma), 1e-30),
                }
            )

    summaries = []
    for case_id, _, _ in cases:
        group = [row for row in rows if row["case_id"] == case_id]
        peak = max(group, key=lambda row: abs(row["sigma_im_S_m"]))
        summaries.append(
            {
                "case_id": case_id,
                "sampled_peak_frequency_hz": peak["frequency_hz"],
                "sampled_peak_sigma_im_S_m": peak["sigma_im_S_m"],
                "sampled_peak_phase_deg": peak["phase_deg"],
                "max_current_mismatch": max(
                    row["left_right_relative_mismatch"] for row in group
                ),
            }
        )

    outdir = RESULTS / "topology_research" / "connectivity_spectra_v1"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "spectra.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "model": "same two-particle geometry; component potentials plus contact graph",
        "frequencies_over_f0": [float(f / rs.FP) for f in frequencies],
        "summaries": summaries,
        "rows": rows,
    }
    (outdir / "spectra.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(outdir), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

