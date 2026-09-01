#!/usr/bin/env python3
"""Dense conductance-frequency map for the two-particle contact mode."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math

import numpy as np

from _paths import RESULTS
from analyze_passivation_spectra import log_peak_and_width
from run_connectivity_spectra import prepare_case
import run_scenario0 as rs


def main():
    length, centres, mesh, facets = prepare_case()
    frequencies = rs.FP * np.logspace(math.log10(0.08), math.log10(12.0), 15)
    conductances = (0.0, 1e-4, 5e-4, 0.002, 0.01, 0.02, 0.05, 0.2, 1.0, 5.0)
    area = 4.0 * math.pi * rs.R**2
    omega0 = 2.0 * math.pi * rs.FP
    interface_b0 = omega0 * rs.C0 * area
    cases = [(f"G_{g:g}S", [0, 1], None if g == 0 else [(0, 1, g)], g) for g in conductances]
    cases.append(("ideal_connected", [0, 0], None, math.inf))

    rows, summaries = [], []
    for case_id, labels, contacts, conductance in cases:
        group = []
        for frequency in frequencies:
            with contextlib.redirect_stdout(io.StringIO()):
                _, ur, ui, potentials = rs.solve(
                    mesh,
                    facets,
                    v_right=rs.V_RIGHT,
                    centres=centres,
                    vim_electrode="dirichlet",
                    component_ids=labels,
                    omega=2.0 * math.pi * float(frequency),
                    contact_edges=contacts,
                )
            left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="left")
            right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="right")
            sigma = 0.5 * (left + right)
            delta_v = 0.0 if len(potentials) == 1 else abs(potentials[0] - potentials[1])
            finite_g = 0.0 if not np.isfinite(conductance) else conductance
            contact_current = finite_g * delta_v
            row = {
                "case_id": case_id,
                "contact_conductance_S": conductance,
                "G_over_omega0CA": conductance / interface_b0 if np.isfinite(conductance) else math.inf,
                "frequency_hz": float(frequency),
                "frequency_over_f0": float(frequency / rs.FP),
                "sigma_re_S_m": sigma.real,
                "sigma_im_S_m": sigma.imag,
                "phase_deg": float(np.degrees(np.angle(sigma))),
                "component_voltage_difference_V": delta_v,
                "contact_current_A": contact_current,
                "contact_dissipation_W": 0.5 * finite_g * delta_v**2,
                "electrode_mismatch": abs(left - right) / max(abs(sigma), 1e-30),
            }
            rows.append(row); group.append(row)
        peak = log_peak_and_width(
            [row["frequency_hz"] for row in group],
            [row["sigma_im_S_m"] for row in group],
        )
        contact_peak = max(group, key=lambda row: row["contact_dissipation_W"])
        summaries.append(
            {
                "case_id": case_id,
                "contact_conductance_S": conductance,
                "G_over_omega0CA": conductance / interface_b0 if np.isfinite(conductance) else math.inf,
                **peak,
                "contact_dissipation_peak_frequency_hz": contact_peak["frequency_hz"],
                "max_contact_dissipation_W": contact_peak["contact_dissipation_W"],
                "max_electrode_mismatch": max(row["electrode_mismatch"] for row in group),
            }
        )
        print(case_id, summaries[-1])

    outdir = RESULTS / "topology_research" / "contact_phase_diagram"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, data in (("phase_diagram.csv", rows), ("summaries.csv", summaries)):
        with (outdir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    result = {
        "normalization": {
            "particle_area_m2": area,
            "f0_hz": rs.FP,
            "omega0_C_A_S": interface_b0,
            "dimensionless_contact": "G/(omega0*C0*A_particle)",
        },
        "summaries": summaries,
        "rows": rows,
    }
    (outdir / "phase_diagram.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(outdir), "normalization": result["normalization"]}, indent=2))


if __name__ == "__main__":
    main()

