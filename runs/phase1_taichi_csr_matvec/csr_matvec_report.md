# Taichi CSR Matvec Smoke

- status: `passed`
- matrices: `4`
- success: `4`
- failed: `0`
- total_csr_nnz: `5292`
- max_abs_error: `9.112`
- max_relative_l2_error: `1.33292e-07`

| matrix | status | shape | nnz | field | symmetry | max_abs | rel_l2 | matvec_ms |
|---|---|---:|---:|---|---|---:|---:|---:|
| suitesparse:HB/curtis54 | success | 54x54 | 291 | pattern | general | 0 | 0 | 155.424 |
| suitesparse:HB/fs_183_1 | success | 183x183 | 998 | real | general | 9.112 | 8.79078e-09 | 198.689 |
| suitesparse:HB/young3c | success | 841x841 | 3988 | real | general | 1.80664e-05 | 1.33292e-07 | 152.672 |
| suitesparse:Grund/b1_ss | success | 7x7 | 15 | real | general | 2.60063e-08 | 1.07759e-08 | 200.553 |
