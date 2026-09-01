#!/usr/bin/env python3
"""Cross-validate the complex SIP response tensor of the T network."""

from __future__ import annotations

import json

import numpy as np

from _paths import RESULTS


def main():
    path = RESULTS / "topology_research" / "branched_network" / "raw_results.json"
    data = json.loads(path.read_text())
    grouped = {}
    for row in data["rows"]:
        key = (float(row["angle_deg"]), round(float(row["frequency_over_f0"]), 8))
        grouped.setdefault(key, {})[row["case_id"]] = row
    increments = {}
    for key, cases in grouped.items():
        base, contact = cases["disconnected"], cases["T_contact"]
        increments[key] = complex(contact["sigma_re_S_m"] - base["sigma_re_S_m"],
                                  contact["sigma_im_S_m"] - base["sigma_im_S_m"])

    rows, by_frequency = [], {}
    ratios = sorted(set(key[1] for key in increments))
    held_out = (22.5, 45.0, 67.5)
    for ratio in ratios:
        dxx = increments[(0.0, ratio)]
        dyy = increments[(90.0, ratio)]
        errors, actuals = [], []
        for angle in held_out:
            theta = np.radians(angle)
            predicted = dxx * np.cos(theta)**2 + dyy * np.sin(theta)**2
            actual = increments[(angle, ratio)]
            errors.append(abs(predicted-actual)); actuals.append(abs(actual))
            rows.append({"frequency_over_f0":ratio,"angle_deg":angle,
                         "actual_delta_re":actual.real,"actual_delta_im":actual.imag,
                         "predicted_delta_re":predicted.real,"predicted_delta_im":predicted.imag,
                         "absolute_error":abs(predicted-actual)})
        relative_l2=float(np.linalg.norm(errors)/max(np.linalg.norm(actuals),1e-30))
        by_frequency[str(ratio)]={
            "delta_x_re":dxx.real,"delta_x_im":dxx.imag,
            "delta_y_re":dyy.real,"delta_y_im":dyy.imag,
            "amplitude_anisotropy_abs_dx_over_dy":float(abs(dxx)/max(abs(dyy),1e-30)),
            "held_out_relative_l2_error":relative_l2,
        }

    # Geometry-only graph visibility has the same second-rank tensor form.
    geometry=data["geometry_modal_data"]
    v0=geometry["0.0"]["bLb_W"]; v90=geometry["90.0"]["bLb_W"]
    visibility_errors=[]; visibility_rows=[]
    for angle in held_out:
        theta=np.radians(angle)
        prediction=v0*np.cos(theta)**2+v90*np.sin(theta)**2
        actual=geometry[str(angle)]["bLb_W"]
        visibility_errors.append(abs(prediction-actual)/max(abs(actual),1e-30))
        visibility_rows.append({"angle_deg":angle,"actual_bLb":actual,
                                "predicted_bLb":float(prediction),
                                "relative_error":float(visibility_errors[-1])})

    all_mismatch=max(float(row["electrode_mismatch"]) for row in data["rows"])
    freq_errors=[v["held_out_relative_l2_error"] for v in by_frequency.values()]
    anis=[v["amplitude_anisotropy_abs_dx_over_dy"] for v in by_frequency.values()]
    output={
        "prediction":"Delta sigma(theta)=Delta sigma_x cos^2(theta)+Delta sigma_y sin^2(theta)",
        "calibration_angles_deg":[0,90],"held_out_angles_deg":list(held_out),
        "median_held_out_relative_l2_error":float(np.median(freq_errors)),
        "maximum_held_out_relative_l2_error":float(max(freq_errors)),
        "anisotropy_range_abs_dx_over_dy":[float(min(anis)),float(max(anis))],
        "max_electrode_mismatch":all_mismatch,
        "graph_visibility_max_relative_error":float(max(visibility_errors)),
        "by_frequency":by_frequency,"graph_visibility_rows":visibility_rows,"rows":rows,
    }
    out=path.parent/"tensor_cross_validation.json"
    out.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:output[k] for k in (
        "median_held_out_relative_l2_error","maximum_held_out_relative_l2_error",
        "anisotropy_range_abs_dx_over_dy","max_electrode_mismatch",
        "graph_visibility_max_relative_error")},indent=2))


if __name__=="__main__": main()
