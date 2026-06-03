# SuiteSparse CSR Import Boundary

- status: `passed`
- eligible_rows: `8`
- attempted_imports: `8`
- imported_matrices: `8`
- failed_imports: `0`
- total_csr_nnz: `15342`
- source_selection_path: `runs/phase1_csr_queue_batch_00001/csr_queue_batch_matrix_queue.jsonl`
- source_header_probe_path: `None`

| rank | matrix | status | shape | stored_entries | csr_nnz | field | symmetry | duplicates | zeros_dropped |
|---:|---|---|---:|---:|---:|---|---|---:|---:|
| 1 | suitesparse:HB/1138_bus | success | 1138x1138 | 2596 | 4054 | real | symmetric | 0 | 0 |
| 2 | suitesparse:HB/494_bus | success | 494x494 | 1080 | 1666 | real | symmetric | 0 | 0 |
| 3 | suitesparse:HB/662_bus | success | 662x662 | 1568 | 2474 | real | symmetric | 0 | 0 |
| 4 | suitesparse:HB/685_bus | success | 685x685 | 1967 | 3249 | real | symmetric | 0 | 0 |
| 6 | suitesparse:HB/arc130 | success | 130x130 | 1282 | 1037 | real | general | 0 | 245 |
| 8 | suitesparse:HB/ash292 | success | 292x292 | 1250 | 2208 | pattern | symmetric | 0 | 0 |
| 11 | suitesparse:HB/ash85 | success | 85x85 | 304 | 523 | pattern | symmetric | 0 | 0 |
| 13 | suitesparse:HB/bcspwr01 | success | 39x39 | 85 | 131 | pattern | symmetric | 0 | 0 |
