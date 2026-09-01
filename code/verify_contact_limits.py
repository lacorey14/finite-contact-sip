#!/usr/bin/env python3
"""Integration checks for zero- and large-conductance contact limits."""

from __future__ import annotations

import json
import math

import numpy as np

from _paths import RESULTS
from run_connectivity_spectra import prepare_case
import run_scenario0 as rs


def solve_sigma(mesh, facets, centres, length, frequency, labels, contacts):
    _, ur, ui, potentials = rs.solve(
        mesh,
        facets,
        v_right=rs.V_RIGHT,
        centres=centres,
        vim_electrode="dirichlet",
        component_ids=labels,
        omega=2.0 * math.pi * frequency,
        contact_edges=contacts,
    )
    left = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="left")
    right = rs.extract_sigma_electrode(mesh, ur, ui, e0=rs.E0, L=length, side="right")
    return 0.5 * (left + right), potentials, abs(left - right) / abs(0.5 * (left + right))


def main():
    length, centres, mesh, facets = prepare_case()
    rows = []
    for ratio in (0.3, 1.0, 3.0):
        frequency = ratio * rs.FP
        disconnected, _, mismatch_d = solve_sigma(
            mesh, facets, centres, length, frequency, [0, 1], None
        )
        zero_contact, _, mismatch_z = solve_sigma(
            mesh, facets, centres, length, frequency, [0, 1], [(0, 1, 0.0)]
        )
        tiny_contact, _, mismatch_t = solve_sigma(
            mesh, facets, centres, length, frequency, [0, 1], [(0, 1, 1e-8)]
        )
        strong_contact, strong_v, mismatch_s = solve_sigma(
            mesh, facets, centres, length, frequency, [0, 1], [(0, 1, 20.0)]
        )
        ideal, _, mismatch_i = solve_sigma(
            mesh, facets, centres, length, frequency, [0, 0], None
        )
        rows.append(
            {
                "frequency_over_f0": ratio,
                "zero_vs_disconnected_rel": abs(zero_contact - disconnected) / abs(disconnected),
                "tiny_vs_disconnected_rel": abs(tiny_contact - disconnected) / abs(disconnected),
                "strong_vs_ideal_rel": abs(strong_contact - ideal) / abs(ideal),
                "strong_component_voltage_difference_V": abs(strong_v[0] - strong_v[1]),
                "max_electrode_mismatch": max(
                    mismatch_d, mismatch_z, mismatch_t, mismatch_s, mismatch_i
                ),
            }
        )
    checks = {
        "exact_zero_contact": max(row["zero_vs_disconnected_rel"] for row in rows) < 1e-12,
        "tiny_contact_limit": max(row["tiny_vs_disconnected_rel"] for row in rows) < 1e-5,
        "strong_contact_limit": max(row["strong_vs_ideal_rel"] for row in rows) < 2e-3,
        "electrode_conservation": max(row["max_electrode_mismatch"] for row in rows) < 5e-4,
    }
    result = {"checks": checks, "rows": rows, "passed": all(checks.values())}
    out = RESULTS / "topology_research" / "contact_limit_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), **result}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

