# Main-figure source-data map

All quantitative panels are generated from archived numerical output. No
values were digitized from plotted figures or entered manually.

| Figure | Manuscript figure stem | Main source data | Generating script |
|---|---|---|---|
| 1 | `Figure_1_finite_contact_dynamics_streamlined_v4_imagegen_a_colorbar` | `contact_phase_diagram/phase_diagram.json`; schematic asset `finite_contact_model_schematic_imagegen_v5_pore_matches_interface_font.png` | `plot_jgr_streamlined_figures.py` |
| 2 | `Figure_2_contact_spectral_anatomy_streamlined_v3_one_row` | `contact_phase_diagram/phase_diagram.json` | `plot_jgr_streamlined_figures.py` |
| 3 | `Figure_3_three_particle_topology_benchmark` | `three_particle_topology/results.json` | `plot_jgr_topology_benchmark_main.py` |
| 4 | `Figure_4_direction_topology_mechanism_compact_v4_heatmap` | `angle_sweep/projection_test.json`; `branched_network/tensor_cross_validation.json`; `branched_network/topology_contrast_analysis.json`; `modal_predictor/three_particle_cross_validation.json` | `plot_jgr_added_key_figures.py` |
| 5 | `Figure_5_directional_blindness` | `random_networks/irregular_fem_raw.json`; `random_networks/visibility_ensemble.json`; `random_networks/irregular_mesh_verification.json`; CSV files in `random_networks/figure_source_data/` | `plot_directional_blindness_atlas.py` |
| 6 | `Figure_6_numerical_closure` | `fem_ensemble/raw_results.json`; `fem_ensemble/ensemble_metrics.csv`; `fem_ensemble/ensemble_analysis.json`; `fem_ensemble/mesh_verification.json`; `scale_similarity/similarity_metrics.csv` | `plot_numerical_closure_atlas.py` |
| 7 | `Figure_7_self_averaging_closure` | `rve_directional_scaling/raw_results.json`; `rve_directional_scaling/analysis.json` | `plot_jgr_added_key_figures.py` |
| 8 | `Figure_8_micro_to_macro_schematic_v4` | `rve_directional_scaling/phase_summary.csv`; `rve_directional_scaling/fem_metrics.csv`; `rve_directional_scaling/fem_mesh_verification.json` | `plot_figure8_with_schematic_v4.py` |

Paths above are relative to `results/topology_research/`. Regeneration scripts
write new files to their original paths under `results/topology_research/`.
