# CSR Unresolved Matrix Diagnostics

- status: `passed`
- diagnosed_matrices: `2`
- source_gpu_probe_ready_candidates: `0`
- executes_gpu: `False`
- runtime_selector_changed: `False`
- next_step: `implement_and_test_preconditioner_or_formulation_candidates_before_gpu_probe`

| matrix | route | zero_diag | asym | cholesky | ritz_min | ritz_max | best_res | best_sol_err |
|---|---|---:|---:|---|---:|---:|---:|---:|
| suitesparse:Gset/G17 | formulation_diagnostic_required | 800 | 0 | failed | -10.4526 | 22.6699 | 9.96165e-06 | 0.0082299 |
| suitesparse:Zitney/extr1b | ilu_or_nonsymmetric_scaling_preconditioner | 2834 | 1.41421 | skipped_nonsymmetric | -423859 | 423859 | 0.864132 | 4198.65 |
