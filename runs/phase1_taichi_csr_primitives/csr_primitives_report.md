# Taichi CSR Primitive Smoke

- status: `passed`
- matrices: `4`
- success: `4`
- failed: `0`
- total_csr_nnz: `5292`
- max_matvec_relative_l2_error: `1.33292e-07`
- max_residual_relative_norm: `1.47892e-07`
- max_dot_relative_error: `5.81917e-08`
- max_norm_relative_error: `2.90958e-08`

| matrix | status | shape | nnz | rel_l2 | residual_rel | dot_rel | norm_rel |
|---|---|---:|---:|---:|---:|---:|---:|
| suitesparse:HB/curtis54 | success | 54x54 | 291 | 0 | 0 | 0 | 0 |
| suitesparse:HB/fs_183_1 | success | 183x183 | 998 | 8.79078e-09 | 1.32827e-09 | 5.81917e-08 | 2.90958e-08 |
| suitesparse:HB/young3c | success | 841x841 | 3988 | 1.33292e-07 | 1.47892e-07 | 4.69336e-08 | 2.34668e-08 |
| suitesparse:Grund/b1_ss | success | 7x7 | 15 | 1.07759e-08 | 0 | 2.0659e-09 | 1.03295e-09 |
