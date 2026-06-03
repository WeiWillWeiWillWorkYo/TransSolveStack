# CSR Micro-Campaign

- status: `passed`
- imported_matrices: `8`
- candidate_jobs: `24`
- gpu_success_rows: `5`
- cpu_screened_out_rows: `19`
- gpu_failed_rows: `0`
- selector_rows: `24`
- selector_oracle_rows: `3`
- max_final_relative_residual: `8.41872e-06`
- max_cpu_recomputed_relative_residual: `8.41872e-06`
- max_solution_relative_error: `0.00436904`

| matrix | candidate | status | solver | preconditioner | repeats | median_ms | iters | rel_res | sol_err |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| suitesparse:HB/bcspwr10 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 2 | 0.163044 | 0.345204 |
| suitesparse:HB/bcspwr10 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 2 | 0.163044 | 0.345204 |
| suitesparse:HB/bcspwr10 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 8.35394e+09 | 1.20478e+10 |
| suitesparse:HB/bcsstk01 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 31 | 7.17601e-06 | 0.557175 |
| suitesparse:HB/bcsstk01 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 33 | 6.07376e-06 | 0.113325 |
| suitesparse:HB/bcsstk01 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 105 | 9.97229e-06 | 0.755891 |
| suitesparse:HB/bcsstk02 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 655.122 | 46 | 5.73295e-06 | 1.62737e-06 |
| suitesparse:HB/bcsstk02 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 805.373 | 40 | 1.36088e-06 | 3.7795e-07 |
| suitesparse:HB/bcsstk02 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.00286737 | 0.617056 |
| suitesparse:HB/bcsstk03 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 68 | 9.30613e-06 | 0.595813 |
| suitesparse:HB/bcsstk03 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 947.337 | 125 | 7.40291e-06 | 0.00436904 |
| suitesparse:HB/bcsstk03 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.000565104 | 2.01236 |
| suitesparse:HB/bcsstk04 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 82 | 7.28504e-06 | 0.578403 |
| suitesparse:HB/bcsstk04 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 49 | 3.48464e-06 | 0.0750214 |
| suitesparse:HB/bcsstk04 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 54 | 9.9413e-06 | 0.949327 |
| suitesparse:HB/bcsstk05 | taichi_csr_cg_none_float64 | success | cg | none | 1 | 1974.22 | 267 | 8.41872e-06 | 3.44912e-05 |
| suitesparse:HB/bcsstk05 | taichi_csr_pcg_jacobi_float64 | success | pcg | jacobi | 1 | 917.48 | 125 | 6.39843e-06 | 5.25257e-06 |
| suitesparse:HB/bcsstk05 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 0.0291065 | 0.618368 |
| suitesparse:HB/bcsstk06 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 248 | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk06 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 103 | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk06 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.06253e-05 | 0.839226 |
| suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 | screened_out | cg | none | 1 |  | 248 | 8.94851e-06 | 0.526635 |
| suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 | screened_out | pcg | jacobi | 1 |  | 103 | 7.19328e-06 | 0.0808606 |
| suitesparse:HB/bcsstk07 | taichi_csr_richardson_jacobi_float64 | screened_out | richardson | jacobi | 1 |  | 300 | 1.06253e-05 | 0.839226 |
