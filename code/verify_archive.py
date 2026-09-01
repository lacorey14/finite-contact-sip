#!/usr/bin/env python3
"""Verify the frozen manuscript-to-figure-to-data correspondence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "topology_research"
FIGURES = ROOT / "figures"

MAP = {
    1: (["Figure_1_finite_contact_dynamics_streamlined_v4_imagegen_a_colorbar"], [
        "contact_phase_diagram/phase_diagram.json",
        "jgr_topology_submission/figures/finite_contact_model_schematic_imagegen_v5_pore_matches_interface_font.png",
    ]),
    2: (["Figure_2_contact_spectral_anatomy_streamlined_v3_one_row"], [
        "contact_phase_diagram/phase_diagram.json",
    ]),
    3: (["Figure_3_three_particle_topology_benchmark"], [
        "three_particle_topology/results.json",
    ]),
    4: (["Figure_4_direction_topology_mechanism_compact_v4_heatmap"], [
        "angle_sweep/projection_test.json",
        "branched_network/tensor_cross_validation.json",
        "branched_network/topology_contrast_analysis.json",
        "modal_predictor/three_particle_cross_validation.json",
    ]),
    5: (["Figure_5_directional_blindness"], [
        "random_networks/irregular_fem_raw.json",
        "random_networks/visibility_ensemble.json",
        "random_networks/irregular_mesh_verification.json",
        "random_networks/figure_source_data/Fig_main_random_network_edges.csv",
    ]),
    6: (["Figure_6_numerical_closure"], [
        "fem_ensemble/raw_results.json",
        "fem_ensemble/ensemble_metrics.csv",
        "fem_ensemble/ensemble_analysis.json",
        "fem_ensemble/mesh_verification.json",
        "scale_similarity/similarity_metrics.csv",
    ]),
    7: (["Figure_7_self_averaging_closure"], [
        "rve_directional_scaling/raw_results.json",
        "rve_directional_scaling/analysis.json",
    ]),
    8: (["Figure_8_micro_to_macro_schematic_v4"], [
        "rve_directional_scaling/phase_summary.csv",
        "rve_directional_scaling/fem_metrics.csv",
        "rve_directional_scaling/fem_mesh_verification.json",
    ]),
}


def main() -> None:
    missing = []
    for number, (stems, sources) in MAP.items():
        for stem in stems:
            for ext in ("png", "pdf", "svg"):
                path = FIGURES / f"{stem}.{ext}"
                if not path.is_file():
                    missing.append(f"Figure {number}: {path.relative_to(ROOT)}")
        for source in sources:
            path = DATA / source
            if not path.is_file():
                missing.append(f"Figure {number}: {path.relative_to(ROOT)}")
    if missing:
        raise SystemExit("Missing archive files:\n" + "\n".join(missing))
    print("PASS: all eight adopted figures and mapped source-data files are present.")


if __name__ == "__main__":
    main()
