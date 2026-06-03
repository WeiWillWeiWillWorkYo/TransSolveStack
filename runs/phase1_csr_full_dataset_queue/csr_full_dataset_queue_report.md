# CSR Full Dataset Queue

- status: `passed`
- schema_version: `phase1_csr_full_dataset_queue_v1`
- queue_id: `phase1_csr_full_dataset_queue_m83`
- runtime_selector_changed: `False`
- executes_gpu: `False`
- imports_matrices: `False`
- resumable: `True`
- external_drive_required: `True`
- external_drive_path: `/mnt/tss_external`
- index_matrices: `2904`
- index_present_archives: `2904`
- already_profiled_matrices: `12`
- eligible_matrices: `1557`
- queued_matrices: `1545`
- queued_jobs: `4635`
- queued_batches: `194`
- planned_gpu_solve_attempts: `4635`
- total_queued_nnz: `158605773`
- estimated_total_nnz_visits: `181114205268`
- by_candidate_profile: `{'symmetric': 786, 'general': 759}`
- by_solver: `{'cg': 786, 'pcg': 786, 'richardson': 786, 'bicgstab': 1518, 'gmres': 759}`
- first_pending_batch_id: `batch_00001`
- next_step: `execute_queue_batches_with_explicit_import_cpu_screen_and_taichi_gpu_benchmark`

## Batches

| batch | matrices | jobs | gpu_attempts | total_nnz | archive_mb | status |
|---|---:|---:|---:|---:|---:|---|
| batch_00001 | 8 | 24 | 24 | 15342 | 0.074 | pending |
| batch_00002 | 8 | 24 | 24 | 27563 | 0.056 | pending |
| batch_00003 | 8 | 24 | 24 | 49029 | 0.140 | pending |
| batch_00004 | 8 | 24 | 24 | 387102 | 1.299 | pending |
| batch_00005 | 8 | 24 | 24 | 950580 | 3.787 | pending |
| batch_00006 | 8 | 24 | 24 | 1929053 | 4.907 | pending |
| batch_00007 | 8 | 24 | 24 | 10186 | 0.044 | pending |
| batch_00008 | 8 | 24 | 24 | 69445 | 0.236 | pending |
| batch_00009 | 8 | 24 | 24 | 102992 | 0.568 | pending |
| batch_00010 | 8 | 24 | 24 | 43089 | 0.164 | pending |
| batch_00011 | 8 | 24 | 24 | 24543 | 0.033 | pending |
| batch_00012 | 8 | 24 | 24 | 31404 | 0.041 | pending |
| batch_00013 | 8 | 24 | 24 | 36200 | 0.044 | pending |
| batch_00014 | 8 | 24 | 24 | 40100 | 0.051 | pending |
| batch_00015 | 8 | 24 | 24 | 27070 | 0.036 | pending |
| batch_00016 | 8 | 24 | 24 | 64170 | 0.075 | pending |
| batch_00017 | 8 | 24 | 24 | 22441 | 0.224 | pending |
| batch_00018 | 8 | 24 | 24 | 88996 | 0.778 | pending |
| batch_00019 | 8 | 24 | 24 | 19714 | 0.058 | pending |
| batch_00020 | 8 | 24 | 24 | 14373 | 0.055 | pending |
| batch_00021 | 8 | 24 | 24 | 64489 | 0.084 | pending |
| batch_00022 | 8 | 24 | 24 | 60835 | 0.358 | pending |
| batch_00023 | 8 | 24 | 24 | 96064 | 0.138 | pending |
| batch_00024 | 8 | 24 | 24 | 50098 | 0.106 | pending |
| ... | 170 more | | | | | |

## Execution Boundary

- This artifact is a queue only.
- Matrix import, CPU screening, and Taichi GPU solves remain separate explicit steps.
- The default runtime selector is not changed by this queue.
