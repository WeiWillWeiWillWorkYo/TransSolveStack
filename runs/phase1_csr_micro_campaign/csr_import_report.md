# SuiteSparse CSR Import Boundary

- status: `passed`
- eligible_rows: `8`
- attempted_imports: `8`
- imported_matrices: `8`
- failed_imports: `0`
- total_csr_nnz: `82789`
- source_selection_path: `runs/phase1_csr_benchmark_expansion_plan/csr_benchmark_matrix_queue.jsonl`
- source_header_probe_path: `None`

| rank | matrix | status | shape | stored_entries | csr_nnz | field | symmetry | duplicates | zeros_dropped |
|---:|---|---|---:|---:|---:|---|---|---:|---:|
| 13 | suitesparse:MathWorks/tomography | success | 500x500 | 28726 | 28726 | real | general | 0 | 0 |
| 14 | suitesparse:HB/jgl009 | success | 9x9 | 50 | 50 | pattern | general | 0 | 0 |
| 15 | suitesparse:HB/ibm32 | success | 32x32 | 126 | 126 | pattern | general | 0 | 0 |
| 16 | suitesparse:SNAP/email-Eu-core | success | 1005x1005 | 25571 | 25571 | pattern | general | 0 | 0 |
| 28 | suitesparse:Oberwolfach/t2dal_e | success | 4257x4257 | 4257 | 4257 | real | symmetric | 0 | 0 |
| 30 | suitesparse:HB/bcsstk07 | success | 420x420 | 4140 | 7860 | real | symmetric | 0 | 0 |
| 31 | suitesparse:HB/lshp1009 | success | 1009x1009 | 3937 | 6865 | pattern | symmetric | 0 | 0 |
| 32 | suitesparse:Gset/G17 | success | 800x800 | 4667 | 9334 | pattern | symmetric | 0 | 0 |
