#!/usr/bin/env python3
"""Calibrate and cross-validate a graph-modal predictor against three-grain FEM.

The electronic network response is represented by

    q(omega) = b.T L (L + i omega Cn I)^-1 b,

on the mean-free subspace.  Here ``b`` is the background potential sampled at
particle centres, L is the contact conductance Laplacian, and Cn is one global
effective nodal capacitance.  A complex transfer coefficient K(omega), learned
from one reference topology only, maps q to the FEM conductivity increment.
All remaining finite-contact cases are held out for prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from _paths import RESULTS
from electronic_components import contact_laplacian


EDGES = {
    "single_edge_0p02": [(0, 1, 0.02)],
    "weak_chain": [(0, 1, 0.002), (1, 2, 0.002)],
    "transition_chain": [(0, 1, 0.02), (1, 2, 0.02)],
    "strong_chain": [(0, 1, 0.2), (1, 2, 0.2)],
    "bottleneck_chain": [(0, 1, 0.2), (1, 2, 0.002)],
}


def modal_response(edges, omega, capacitance, background):
    lap = contact_laplacian(len(background), edges)
    values, vectors = np.linalg.eigh(lap)
    b = np.asarray(background, dtype=float)
    b -= b.mean()
    projections = vectors.T @ b
    terms = []
    response = 0j
    for value, projection in zip(values, projections):
        weight = float(projection**2)
        factor = 0j if value < 1e-14 else value / (value + 1j * omega * capacitance)
        response += weight * factor
        terms.append({"lambda_S": float(value), "weight_V2": weight})
    return response, terms


def load_parallel():
    source = RESULTS / "topology_research" / "three_particle_topology" / "results.json"
    data = json.loads(source.read_text())
    grouped = {}
    for row in data["rows"]:
        if row["orientation"] != "parallel":
            continue
        grouped.setdefault(row["case_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["frequency_hz"])
    return grouped


def predict_error(log10_capacitance, grouped, calibration="strong_chain"):
    capacitance = 10.0**log10_capacitance
    base = grouped["disconnected"]
    cal = grouped[calibration]
    background = np.array([-0.75, 0.0, 0.75])  # E0 * 15 mm spacing
    total_error = 0.0
    count = 0
    for index, (base_row, cal_row) in enumerate(zip(base, cal)):
        omega = 2 * np.pi * base_row["frequency_hz"]
        qcal, _ = modal_response(EDGES[calibration], omega, capacitance, background)
        ds_cal = complex(cal_row["sigma_re_S_m"] - base_row["sigma_re_S_m"],
                         cal_row["sigma_im_S_m"] - base_row["sigma_im_S_m"])
        transfer = ds_cal / qcal
        scale = max(abs(ds_cal), 1e-7)
        for case in ("single_edge_0p02", "weak_chain", "transition_chain", "bottleneck_chain"):
            row = grouped[case][index]
            actual = complex(row["sigma_re_S_m"] - base_row["sigma_re_S_m"],
                             row["sigma_im_S_m"] - base_row["sigma_im_S_m"])
            q, _ = modal_response(EDGES[case], omega, capacitance, background)
            predicted = transfer * q
            total_error += abs(predicted - actual) ** 2 / scale**2
            count += 1
    return total_error / count


def main():
    grouped = load_parallel()
    fit = minimize_scalar(
        lambda x: predict_error(x, grouped), bounds=(-7.0, 0.0), method="bounded",
        options={"xatol": 1e-7},
    )
    capacitance = 10.0**fit.x
    base = grouped["disconnected"]
    cal = grouped["strong_chain"]
    background = np.array([-0.75, 0.0, 0.75])
    rows = []
    for index, (base_row, cal_row) in enumerate(zip(base, cal)):
        frequency = base_row["frequency_hz"]
        omega = 2 * np.pi * frequency
        qcal, _ = modal_response(EDGES["strong_chain"], omega, capacitance, background)
        ds_cal = complex(cal_row["sigma_re_S_m"] - base_row["sigma_re_S_m"],
                         cal_row["sigma_im_S_m"] - base_row["sigma_im_S_m"])
        transfer = ds_cal / qcal
        for case in EDGES:
            actual_row = grouped[case][index]
            actual = complex(actual_row["sigma_re_S_m"] - base_row["sigma_re_S_m"],
                             actual_row["sigma_im_S_m"] - base_row["sigma_im_S_m"])
            q, terms = modal_response(EDGES[case], omega, capacitance, background)
            predicted = transfer * q
            rows.append({
                "case_id": case, "frequency_hz": frequency,
                "actual_delta_re": actual.real, "actual_delta_im": actual.imag,
                "predicted_delta_re": predicted.real, "predicted_delta_im": predicted.imag,
                "absolute_error": abs(predicted - actual),
                "modal_response_re": q.real, "modal_response_im": q.imag,
                "modes": terms,
            })
    metrics = {}
    for case in EDGES:
        subset = [row for row in rows if row["case_id"] == case]
        actual = np.array([complex(r["actual_delta_re"], r["actual_delta_im"]) for r in subset])
        pred = np.array([complex(r["predicted_delta_re"], r["predicted_delta_im"]) for r in subset])
        denom = np.sum(np.abs(actual - actual.mean())**2)
        metrics[case] = {
            "relative_l2_error": float(np.linalg.norm(pred-actual) / max(np.linalg.norm(actual), 1e-30)),
            "complex_r2": float(1 - np.sum(np.abs(pred-actual)**2) / max(denom, 1e-30)),
        }
    output = {
        "model": "delta_sigma = K(omega) b^T L(L+i omega Cn I)^-1 b",
        "calibration_case": "strong_chain",
        "held_out_cases": [k for k in EDGES if k != "strong_chain"],
        "effective_nodal_capacitance_F": capacitance,
        "objective": float(fit.fun), "metrics": metrics, "rows": rows,
    }
    outdir = RESULTS / "topology_research" / "modal_predictor"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "three_particle_cross_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("effective_nodal_capacitance_F", "objective", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
