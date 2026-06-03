# CSR Blocked Gap Guarded Replay

- status: `passed`
- shadow_only: `True`
- runtime_selector_changed: `False`
- replay_rows: `5`
- actual_quality_gate_blocks: `5`
- actual_runtime_selector_changes: `0`
- fallback_chain_enforced_rows: `5`
- learned_success_predictions: `5`
- learned_oracle_predictions: `4`
- learned_differs_from_artifact_rows: `1`
- max_learned_regret_vs_artifact_ms: `1098.04`
- counterfactual_eligible_gate_promotions: `5`
- acceptance_note: `Actual replay requires the D20 model to remain blocked by quality gate; counterfactual promotion rows only verify existing profiled-success and fallback-chain enforcement.`
- next_step: `add a transformer handoff bundle that packages tensors, D20/D21 summaries, and guard thresholds for external model training`

| matrix | learned | artifact | blocked_status | counterfactual_status | regret_ms |
|---|---|---|---|---|---:|
| suitesparse:FIDAP/ex5 | taichi_csr_pcg_symmetric_equilibration_float64 | taichi_csr_bicgstab_ilu0_float64 | blocked_quality_gate | promoted | 1098.04 |
| suitesparse:Grund/b1_ss | taichi_csr_bicgstab_row_column_equilibration_float64 | taichi_csr_bicgstab_row_column_equilibration_float64 | blocked_quality_gate | promoted | 0 |
| suitesparse:HB/bcsstk01 | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | blocked_quality_gate | promoted | 0 |
| suitesparse:HB/curtis54 | taichi_csr_bicgstab_row_column_equilibration_float64 | taichi_csr_bicgstab_row_column_equilibration_float64 | blocked_quality_gate | promoted | 0 |
| suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_pcg_symmetric_equilibration_float64 | taichi_csr_pcg_symmetric_equilibration_float64 | blocked_quality_gate | promoted | 0 |
