#!/usr/bin/env python3
"""Quantify directional blindness to a missing branch contact."""

from __future__ import annotations

import json
import numpy as np

from _paths import RESULTS


def z(row):
    return complex(row["sigma_re_S_m"],row["sigma_im_S_m"])


def main():
    root=RESULTS/"topology_research"/"branched_network"
    full=json.loads((root/"raw_results.json").read_text())["rows"]
    partial=json.loads((root/"topology_contrast_raw.json").read_text())["rows"]
    lookup={}
    for row in full:
        lookup[(float(row["angle_deg"]),round(row["frequency_over_f0"],8),row["case_id"])]=row
    rows=[]
    for row in partial:
        angle=float(row["angle_deg"]); ratio=round(row["frequency_over_f0"],8)
        base=lookup[(angle,ratio,"disconnected")]; complete=lookup[(angle,ratio,"T_contact")]
        contrast=z(complete)-z(row); total_increment=z(complete)-z(base)
        rows.append({"angle_deg":angle,"frequency_over_f0":ratio,
                     "contrast_re":contrast.real,"contrast_im":contrast.imag,
                     "contrast_over_full_contact_increment":abs(contrast)/max(abs(total_increment),1e-30),
                     "contrast_over_bulk_sigma":abs(contrast)/max(abs(z(complete)),1e-30),
                     "full_contact_increment_abs":abs(total_increment),
                     "electrode_mismatch":row["electrode_mismatch"]})
    summary={}
    for angle in (0.0,90.0):
        subset=[r for r in rows if r["angle_deg"]==angle]
        summary[str(angle)]={
            "median_contrast_over_full_contact_increment":float(np.median([r["contrast_over_full_contact_increment"] for r in subset])),
            "range_contrast_over_full_contact_increment":[float(min(r["contrast_over_full_contact_increment"] for r in subset)),float(max(r["contrast_over_full_contact_increment"] for r in subset))],
            "median_contrast_over_bulk_sigma":float(np.median([r["contrast_over_bulk_sigma"] for r in subset])),
            "max_contrast_over_bulk_sigma":float(max(r["contrast_over_bulk_sigma"] for r in subset)),
        }
    output={
        "comparison":"full T contacts versus identical geometry with branch edge (1,3) deleted",
        "directional_metrics":summary,
        "orthogonal_to_parallel_median_visibility_ratio":summary["90.0"]["median_contrast_over_full_contact_increment"] / max(summary["0.0"]["median_contrast_over_full_contact_increment"],1e-30),
        "max_electrode_mismatch":float(max(r["electrode_mismatch"] for r in rows)),
        "rows":rows,
    }
    (root/"topology_contrast_analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({k:output[k] for k in ("directional_metrics","orthogonal_to_parallel_median_visibility_ratio","max_electrode_mismatch")},indent=2))


if __name__=="__main__":main()
