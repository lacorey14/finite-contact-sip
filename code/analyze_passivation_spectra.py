#!/usr/bin/env python3
"""Publication-oriented post-processing of the Phase-1 spectra."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from _paths import RESULTS
import run_passivation as rp
import run_scenario0 as rs


OUT = RESULTS / "passivation_phase1"


def log_peak_and_width(freq, response):
    freq = np.asarray(freq, float)
    y = np.abs(np.asarray(response, float))
    x = np.log10(freq)
    i = int(np.argmax(y))
    fp, yp = float(freq[i]), float(y[i])
    if 0 < i < len(y) - 1:
        coef = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
        if coef[0] < 0:
            xp = float(-coef[1] / (2.0 * coef[0]))
            if x[i - 1] <= xp <= x[i + 1]:
                fp = 10.0**xp
                yp = float(np.polyval(coef, xp))

    half = 0.5 * yp
    left = right = None
    for j in range(i - 1, -1, -1):
        if y[j] <= half <= y[j + 1]:
            left = float(np.interp(half, [y[j], y[j + 1]], [x[j], x[j + 1]]))
            break
    for j in range(i, len(y) - 1):
        if y[j] >= half >= y[j + 1]:
            right = float(np.interp(half, [y[j + 1], y[j]], [x[j + 1], x[j]]))
            break
    return {
        "fp_interp_hz": fp,
        "peak_interp": yp,
        "fwhm_decades": right - left if left is not None and right is not None else None,
        "f_half_low_hz": 10.0**left if left is not None else None,
        "f_half_high_hz": 10.0**right if right is not None else None,
    }


def analyze_case(case_id: str):
    src = OUT / case_id / "spectrum.json"
    data = json.loads(src.read_text())
    rows = data["spectrum"]
    freq = np.asarray([r["frequency_hz"] for r in rows])
    sre = np.asarray([r["sigma_re"] for r in rows])
    sim = np.asarray([r["sigma_im"] for r in rows])
    phase = np.asarray([r["phase_deg"] for r in rows])
    peak = log_peak_and_width(freq, sim)
    phase_peak = log_peak_and_width(freq, phase)
    iref = int(np.argmin(np.abs(np.log(freq / rs.FP))))
    sigma_high = float(sre[-1])
    m_relax = (
        (sigma_high - float(sre[0])) / sigma_high
        if abs(sigma_high) > 0
        else float("nan")
    )
    return {
        "case_id": case_id,
        "f_active_geom": data["f_active_geom_mesh"],
        "fp_sigma_im_interp_hz": peak["fp_interp_hz"],
        "tau_interp_s": 1.0 / (2.0 * math.pi * peak["fp_interp_hz"]),
        "peak_sigma_im_interp": peak["peak_interp"],
        "fwhm_sigma_im_decades": peak["fwhm_decades"],
        "fp_phase_interp_hz": phase_peak["fp_interp_hz"],
        "max_phase_interp_deg": phase_peak["peak_interp"],
        "m_relax_delta_sigma": m_relax,
        "Seff_ref_J_frac_at_reference_fp": rows[iref].get("Seff_ref_J_frac"),
        "Seff_ref_q_frac_at_reference_fp": rows[iref].get("Seff_ref_q_frac"),
        "reference_weight_frequency_hz": float(freq[iref]),
        "max_net_I_rel_l1": data.get("max_net_I_rel_l1"),
    }


def corr(rows, xkey, ykey, *, equal_area_only=False):
    pairs = [
        (r.get(xkey), r.get(ykey))
        for r in rows
        if r["case_id"] != "full_passive"
        and (not equal_area_only or 0.49 < r["f_active_geom"] < 0.51)
        and r.get(xkey) is not None
        and r.get(ykey) is not None
    ]
    x, y = np.asarray(pairs, float).T
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 else None


def main():
    case_ids = [cid for cid in rp.GEOMETRIES if (OUT / cid / "spectrum.json").is_file()]
    rows = [analyze_case(cid) for cid in case_ids]
    result = {
        "reference_fp_hz": rs.FP,
        "cases": rows,
        "correlations": {
            "peak_sigma_im_vs_f_geom": corr(rows, "f_active_geom", "peak_sigma_im_interp"),
            "peak_sigma_im_vs_Seff_ref_J": corr(
                rows, "Seff_ref_J_frac_at_reference_fp", "peak_sigma_im_interp"
            ),
            "m_relax_vs_f_geom": corr(rows, "f_active_geom", "m_relax_delta_sigma"),
            "m_relax_vs_Seff_ref_J": corr(
                rows, "Seff_ref_J_frac_at_reference_fp", "m_relax_delta_sigma"
            ),
            "equal_area_peak_sigma_im_vs_Seff_ref_J": corr(
                rows,
                "Seff_ref_J_frac_at_reference_fp",
                "peak_sigma_im_interp",
                equal_area_only=True,
            ),
            "equal_area_m_relax_vs_Seff_ref_J": corr(
                rows,
                "Seff_ref_J_frac_at_reference_fp",
                "m_relax_delta_sigma",
                equal_area_only=True,
            ),
        },
    }
    (OUT / "spectral_analysis.json").write_text(json.dumps(result, indent=2))
    with (OUT / "spectral_analysis.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Phase 1 spectral analysis",
        "",
        "| case | f_active | fp (Hz) | tau (ms) | peak sigma_im | FWHM (decades) | m_relax | Seff_ref,J |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        width = r["fwhm_sigma_im_decades"]
        lines.append(
            f"| {r['case_id']} | {r['f_active_geom']:.4f} | "
            f"{r['fp_sigma_im_interp_hz']:.2f} | {1e3*r['tau_interp_s']:.3f} | "
            f"{r['peak_sigma_im_interp']:.6g} | "
            f"{width:.3f} | " if width is not None else
            f"| {r['case_id']} | {r['f_active_geom']:.4f} | "
            f"{r['fp_sigma_im_interp_hz']:.2f} | {1e3*r['tau_interp_s']:.3f} | "
            f"{r['peak_sigma_im_interp']:.6g} | n/a | "
        )
        lines[-1] += (
            f"{r['m_relax_delta_sigma']:.6g} | "
            f"{r['Seff_ref_J_frac_at_reference_fp'] if r['Seff_ref_J_frac_at_reference_fp'] is not None else 'n/a'} |"
        )
    lines += ["", "## Correlations", "", f"```json\n{json.dumps(result['correlations'], indent=2)}\n```", ""]
    (OUT / "spectral_analysis.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
