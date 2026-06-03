# CSR Blocked Gap Probe

- status: `passed`
- schema_version: `phase1_csr_blocked_gap_probe_v1`
- runtime_selector_changed: `False`
- executes_gpu: `True`
- coverage_source_latest_batch_id: `batch_00010`
- coverage_outcome: `screen_only_no_oracle`
- selected_matrices: `8`
- candidate_jobs: `16`
- gpu_success_rows: `0`
- cpu_screened_out_rows: `16`
- gpu_failed_rows: `0`
- selector_oracle_rows: `0`
- queue_merge_ready: `False`
- queue_merge_ready_gap_ids: `[]`
- next_step: `keep_blocked_gaps_out_of_generic_queue_and_search_better_formulations_or_matrices`

## By Gap

| gap | success | screened_out | failed |
|---|---:|---:|---:|
| general_bicgstab_ilu0 | 0 | 7 | 0 |
| general_bicgstab_row_column_equilibration | 0 | 7 | 0 |
| symmetric_chebyshev_jacobi | 0 | 1 | 0 |
| symmetric_pcg_symmetric_equilibration | 0 | 1 | 0 |
