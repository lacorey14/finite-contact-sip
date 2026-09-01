# Main-Figure Source-Data Map

All quantitative panels are generated directly from archived numerical output;
no digitization or manually entered plotting values are used.

| Figure | Main source data | Generating script |
|---|---|---|
| 1 | `contact_phase_diagram/phase_diagram.json`; `angle_sweep/projection_test.json`; `branched_network/tensor_cross_validation.json`; `explicit_neck_convergence.json` | `plot_jgr_topology_figure1.py` |
| 2 | `contact_phase_diagram/phase_diagram.json`; `contact_geometry_mapping/mapping.json`; `explicit_neck_validation.json`; `explicit_neck_convergence.json` | `plot_jgr_added_key_figures.py` |
| 3 | `angle_sweep/projection_test.json`; `branched_network/tensor_cross_validation.json`; `branched_network/topology_contrast_analysis.json`; `modal_predictor/three_particle_cross_validation.json` | `plot_jgr_added_key_figures.py` |
| 4 | `random_networks/irregular_fem_raw.json`; `random_networks/visibility_ensemble.json`; CSV files under `random_networks/figure_source_data/` | `plot_directional_blindness_atlas.py` |
| 5 | `fem_ensemble/raw_results.json`; `fem_ensemble/ensemble_metrics.csv`; `fem_ensemble/mesh_verification.json`; `scale_similarity/similarity_metrics.csv` | `plot_numerical_closure_atlas.py` |
| 6 | `rve_directional_scaling/raw_results.json`; `rve_directional_scaling/analysis.json` | `plot_jgr_added_key_figures.py` |
| 7 | `rve_directional_scaling/size_summary.csv`; `rve_directional_scaling/phase_summary.csv`; `rve_directional_scaling/frequency_summary.csv`; `rve_directional_scaling/fem_metrics.csv`; `rve_directional_scaling/fem_mesh_verification.json` | `plot_rve_directional_atlas.py` |

The public repository should preserve these relative paths or provide a
machine-readable manifest mapping archived paths to the names above.

## Supporting figures

| Figure | Main source data | Generating script |
|---|---|---|
| S1 | `contact_phase_diagram/phase_diagram.json`; `contact_limit_verification.json`; `contact_mesh_domain_verification.json`; `explicit_neck_convergence.json` | `plot_jgr_topology_supporting_figures.py` |
| S2 | `three_particle_topology/results.json`; `modal_predictor/three_particle_cross_validation.json` | `plot_jgr_topology_supporting_figures.py` |
| S3 | `random_networks/irregular_fem_raw.json`; `random_networks/irregular_fem_analysis.json`; `random_networks/visibility_ensemble.json`; `random_networks/irregular_mesh_verification.json` | `plot_random_network_publication_figures.py`; copied without data alteration by `plot_jgr_topology_supporting_figures.py` |
| S4 | `rve_directional_scaling/analysis.json`; `rve_directional_scaling/fem_analysis.json` | `plot_jgr_topology_supporting_figures.py` |

Every archived numeric file should retain units, complex-number convention,
case identifiers, and the script version that created it. Raster previews are
not source data; editable PDF/SVG figures and the underlying JSON/CSV files are.
