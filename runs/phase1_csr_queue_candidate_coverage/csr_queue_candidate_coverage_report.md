# CSR Queue Candidate Coverage

- status: `passed`
- schema_version: `phase1_csr_queue_candidate_coverage_v1`
- runtime_selector_changed: `False`
- executes_gpu: `False`
- completed_queue_batches: `10`
- next_pending_batch_id: `batch_00011`
- latest_batch_id: `batch_00010`
- latest_batch_outcome: `screen_only_no_oracle`
- recent_window_batches: `['batch_00009', 'batch_00010']`
- recent_success_rate: `0.16666666666666666`
- trigger_candidate_coverage: `True`
- trigger_reason: `screen_only_after_low_success_batch`
- current_queue_solvers: `['bicgstab', 'cg', 'gmres', 'pcg', 'richardson']`
- current_queue_preconditioners: `['jacobi', 'none']`
- queue_missing_supported_solvers: `['chebyshev']`
- queue_missing_supported_preconditioners: `['block_jacobi', 'ilu0', 'row_column_equilibration', 'symmetric_equilibration']`
- planned_gap_count: `6`
- queue_ready_gap_count: `2`
- cpu_screen_blocked_gap_count: `4`
- next_step: `run_queue_ready_gmres_restart_probe_then_integrate_cpu_screens_for_ilu0_equilibration_chebyshev`

## Recent Batches

| batch | outcome | jobs | success | screened | oracle | success_rate |
|---|---|---:|---:|---:|---:|---:|
| batch_00003 | profiled_with_gpu_success | 24 | 5 | 19 | 3 | 0.208 |
| batch_00004 | profiled_with_gpu_success | 24 | 3 | 21 | 2 | 0.125 |
| batch_00005 | profiled_with_gpu_success | 24 | 1 | 23 | 1 | 0.042 |
| batch_00006 | profiled_with_gpu_success | 24 | 1 | 23 | 1 | 0.042 |
| batch_00007 | profiled_with_gpu_success | 24 | 16 | 8 | 6 | 0.667 |
| batch_00008 | profiled_with_gpu_success | 24 | 12 | 12 | 5 | 0.500 |
| batch_00009 | profiled_with_gpu_success | 24 | 8 | 16 | 4 | 0.333 |
| batch_00010 | screen_only_no_oracle | 24 | 0 | 24 | 0 | 0.000 |

## Candidate Gaps

| priority | gap | candidate | queue_ready | blocker | next_action |
|---|---|---|---|---|---|
| high | general_gmres_jacobi_restart32 | taichi_csr_gmres_jacobi_restart32_float64 | True |  | add_to_candidate_coverage_probe_queue |
| medium | general_gmres_jacobi_restart64 | taichi_csr_gmres_jacobi_restart64_float64 | True |  | add_to_candidate_coverage_probe_queue_after_restart32 |
| high | general_bicgstab_ilu0 | taichi_csr_bicgstab_ilu0_float64 | False | generic_queue_cpu_screen_for_ilu0_not_integrated | probe_with_existing_ilu0_coverage_path_before_queue_merge |
| medium | general_bicgstab_row_column_equilibration | taichi_csr_bicgstab_row_column_equilibration_float64 | False | generic_queue_cpu_screen_for_row_column_equilibration_not_integrated | add_scaled_cpu_screen_before_queue_merge |
| medium | symmetric_chebyshev_jacobi | taichi_csr_chebyshev_jacobi_float64 | False | generic_queue_cpu_screen_for_chebyshev_not_integrated | add_chebyshev_cpu_screen_before_queue_merge |
| medium | symmetric_pcg_symmetric_equilibration | taichi_csr_pcg_symmetric_equilibration_float64 | False | generic_queue_cpu_screen_for_symmetric_equilibration_not_integrated | add_scaled_cpu_screen_before_queue_merge |
