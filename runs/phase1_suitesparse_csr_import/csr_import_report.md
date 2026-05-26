# SuiteSparse CSR Import Boundary

- status: `passed`
- eligible_rows: `48`
- attempted_imports: `12`
- imported_matrices: `12`
- failed_imports: `0`
- total_csr_nnz: `118312`
- source_selection_path: `runs/phase1_suitesparse_selection/selected_matrices.jsonl`
- source_header_probe_path: `runs/phase1_suitesparse_header_probe/archive_header_probe.jsonl`

| rank | matrix | status | shape | stored_entries | csr_nnz | field | symmetry | duplicates | zeros_dropped |
|---:|---|---|---:|---:|---:|---|---|---:|---:|
| 1 | suitesparse:HB/curtis54 | success | 54x54 | 291 | 291 | pattern | general | 0 | 0 |
| 2 | suitesparse:HB/fs_183_1 | success | 183x183 | 1069 | 998 | real | general | 0 | 71 |
| 3 | suitesparse:HB/young3c | success | 841x841 | 3988 | 3988 | real | general | 0 | 0 |
| 4 | suitesparse:Grund/b1_ss | success | 7x7 | 15 | 15 | real | general | 0 | 0 |
| 5 | suitesparse:Zitney/extr1b | success | 2836x2836 | 11404 | 10965 | real | general | 0 | 439 |
| 6 | suitesparse:Hamrle/Hamrle1 | success | 32x32 | 98 | 98 | real | general | 0 | 0 |
| 7 | suitesparse:Sandia/oscil_dcop_01 | success | 430x430 | 1544 | 1544 | real | general | 0 | 0 |
| 8 | suitesparse:JGD_Trefethen/Trefethen_20b | success | 19x19 | 83 | 147 | integer | symmetric | 0 | 0 |
| 9 | suitesparse:Negre/dendrimer | success | 730x730 | 31877 | 63024 | real | symmetric | 0 | 0 |
| 10 | suitesparse:Goodwin/Goodwin_010 | success | 1182x1182 | 32282 | 32282 | real | general | 0 | 0 |
| 11 | suitesparse:FIDAP/ex5 | success | 27x27 | 153 | 279 | real | symmetric | 0 | 0 |
| 12 | suitesparse:Bai/cdde1 | success | 961x961 | 4681 | 4681 | real | general | 0 | 0 |
