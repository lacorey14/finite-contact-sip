#!/usr/bin/env python3
"""Test whether the finite-contact conductivity increment follows cos^2(theta)."""

from __future__ import annotations

import json

import numpy as np

from _paths import RESULTS


def main():
    root = RESULTS / "topology_research"
    raw = json.loads((root / "angle_sweep" / "raw_results.json").read_text())["rows"]
    old = json.loads((root / "three_particle_topology" / "results.json").read_text())["rows"]
    records = []
    # New intermediate angles have matched disconnected/contacted meshes.
    for angle in (22.5, 45.0, 67.5):
        by_case = {}
        for row in raw:
            if float(row["angle_deg"]) == angle:
                by_case.setdefault(row["case_id"], {})[round(row["frequency_over_f0"], 8)] = row
        for key, base in by_case["disconnected"].items():
            contact = by_case["transition_chain"][key]
            records.append((angle, key, complex(contact["sigma_re_S_m"]-base["sigma_re_S_m"],
                                                  contact["sigma_im_S_m"]-base["sigma_im_S_m"]),
                            max(base["electrode_mismatch"], contact["electrode_mismatch"])))
    # Existing endpoint meshes, interpolate their nine-point spectra to new five-point frequencies.
    for angle, orientation in ((0.0, "parallel"), (90.0, "transverse")):
        groups = {}
        for row in old:
            if row["orientation"] == orientation and row["case_id"] in ("disconnected", "transition_chain"):
                groups.setdefault(row["case_id"], []).append(row)
        for values in groups.values():
            values.sort(key=lambda r: r["frequency_over_f0"])
        x = np.log10([r["frequency_over_f0"] for r in groups["disconnected"]])
        targets = np.log10(np.logspace(-1, 1, 5))
        deltas = []
        for part in ("sigma_re_S_m", "sigma_im_S_m"):
            y = np.array([c[part]-b[part] for b,c in zip(groups["disconnected"], groups["transition_chain"])])
            deltas.append(np.interp(targets, x, y))
        for target, re, im in zip(10**targets, *deltas):
            mismatch=max(max(r["electrode_mismatch"] for r in groups[k]) for k in groups)
            records.append((angle, round(float(target),8), complex(re,im),mismatch))

    metrics, detailed = {}, []
    for ratio in sorted(set(r[1] for r in records)):
        subset = sorted((r for r in records if r[1] == ratio), key=lambda r:r[0])
        angles=np.array([r[0] for r in subset]); actual=np.array([r[2] for r in subset])
        x=np.cos(np.radians(angles))**2
        # Strict zero-intercept prediction calibrated only at theta=0.
        endpoint=actual[angles==0][0]
        predicted=endpoint*x
        rel=np.linalg.norm(predicted-actual)/max(np.linalg.norm(actual),1e-30)
        corr=float(np.corrcoef(x, np.abs(actual))[0,1])
        metrics[str(ratio)]={"relative_l2_error_cos2":float(rel),"corr_cos2_vs_abs_delta":corr}
        for a,xx,z,p,m in zip(angles,x,actual,predicted,[r[3] for r in subset]):
            detailed.append({"angle_deg":float(a),"frequency_over_f0":ratio,"cos2":float(xx),
                             "delta_re":z.real,"delta_im":z.imag,
                             "prediction_re":p.real,"prediction_im":p.imag,
                             "prediction_error":abs(p-z),"electrode_mismatch":m})
    summary={
        "hypothesis":"Delta sigma(theta,omega) = Delta sigma(0,omega) cos^2(theta)",
        "angles_deg":[0,22.5,45,67.5,90],
        "median_relative_l2_error":float(np.median([v["relative_l2_error_cos2"] for v in metrics.values()])),
        "max_relative_l2_error":float(max(v["relative_l2_error_cos2"] for v in metrics.values())),
        "median_correlation":float(np.median([v["corr_cos2_vs_abs_delta"] for v in metrics.values()])),
        "max_electrode_mismatch":float(max(r[3] for r in records)),
        "by_frequency":metrics,"rows":detailed,
    }
    out=root/"angle_sweep"/"projection_test.json"
    out.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("median_relative_l2_error","max_relative_l2_error","median_correlation","max_electrode_mismatch")},indent=2))


if __name__=="__main__": main()
