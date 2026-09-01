#!/usr/bin/env python3
"""
Post-process sipFoam multi-grain fields: remove affine V_im gauge V≈a+b x,
then correct volume-mean σ*_im and rewrite bias rows.

Does NOT modify sipFoam. Diagnostic / corrected metrics only.

Usage:
  .venv/bin/python postprocessing/gauge_correct_of_sigma.py
  .venv/bin/python postprocessing/gauge_correct_of_sigma.py --cases scenario_H_phi1p00_d5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "postprocessing"))

from interpretation_bias import bias_at_frequency, usable_band  # noqa: E402

SIGMA0 = 0.2
E0 = 50.0
F_HZ = 154.48591479 / (2 * np.pi)


def correct_case(case: Path) -> dict:
    import fluidfoam

    meta = {}
    for line in (case / "case_meta.txt").read_text().splitlines():
        k, v = line.split("=", 1)
        try:
            meta[k] = float(v)
        except ValueError:
            meta[k] = v

    vim = np.asarray(fluidfoam.readfield(str(case), "6000", "V_im")).ravel()
    jre = fluidfoam.readfield(str(case), "6000", "J_re")
    jim = fluidfoam.readfield(str(case), "6000", "J_im")
    xs, _, _ = fluidfoam.readmesh(str(case), verbose=False)
    n = min(len(vim), len(xs), jre.shape[1])
    vim, xs = vim[:n], xs[:n]
    coef = np.linalg.lstsq(np.column_stack([np.ones(n), xs]), vim, rcond=None)[0]
    a, b = float(coef[0]), float(coef[1])
    sig_raw = complex(-float(np.mean(jre[0, :n])) / E0, -float(np.mean(jim[0, :n])) / E0)
    sig_corr = complex(sig_raw.real, sig_raw.imag - SIGMA0 * b / E0)
    phi = float(meta.get("volume_fraction", meta.get("target_phi")))
    br = bias_at_frequency(sig_raw, F_HZ, phi)
    bc = bias_at_frequency(sig_corr, F_HZ, phi)
    return {
        "case_id": case.name,
        "phi_true": phi,
        "affine_a": a,
        "affine_b": b,
        "sigma_raw_re": sig_raw.real,
        "sigma_raw_im": sig_raw.imag,
        "sigma_corr_re": sig_corr.real,
        "sigma_corr_im": sig_corr.imag,
        "phase_error_raw_deg": br.phase_error_true_deg,
        "phase_error_corr_deg": bc.phase_error_true_deg,
        "phi_inv_raw": br.phi_inv,
        "phi_inv_corr": bc.phi_inv,
        "band_raw": usable_band(br.phi_bias_rel, br.phase_error_true_deg, reliable=br.inversion_reliable),
        "band_corr": usable_band(bc.phi_bias_rel, bc.phase_error_true_deg, reliable=bc.inversion_reliable),
        "affine_frac_of_sigma_im": abs(SIGMA0 * b / E0) / max(abs(sig_raw.imag), 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        nargs="*",
        default=[
            "scenario_H_phi1p00_d5",
            "scenario_H_L70_d5",
            "scenario_H_phi1p50_d5",
            "scenario_Aprime_d6",
            "scenario_B_N2_d5",
        ],
    )
    args = ap.parse_args()
    rows = []
    for name in args.cases:
        case = ROOT / "simulations" / name
        if not (case / "6000" / "V_im").exists():
            print("skip", name)
            continue
        row = correct_case(case)
        rows.append(row)
        print(
            f"{name}: |Δφ| {row['phase_error_raw_deg']:.2f}° → {row['phase_error_corr_deg']:.2f}°  "
            f"(affine frac of σ_im={row['affine_frac_of_sigma_im']:.2f})"
        )
    out = ROOT / "results" / "of_gauge_corrected_sigma.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (ROOT / "results" / "of_gauge_corrected_sigma.json").write_text(json.dumps(rows, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
