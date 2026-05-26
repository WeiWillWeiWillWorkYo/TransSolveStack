# Dataset Download Plan

- catalog: `phase1_external_small`
- data_root: `data/raw`
- dry_run: `True`
- matrices: `3`
- estimated_download_gb: `0.003`

| matrix | source | format | shape | nnz | gb | required | local_path |
|---|---|---|---:|---:|---:|---:|---|
| suitesparse/HB/west0479 | SuiteSparse Matrix Collection | matrix_market_tar_gz | 479x479 | 1910 | 0.001 | yes | data/raw/suitesparse_matrix_collection/HB/west0479.tar.gz |
| suitesparse/HB/bcsstk01 | SuiteSparse Matrix Collection | matrix_market_tar_gz | 48x48 | 400 | 0.001 | yes | data/raw/suitesparse_matrix_collection/HB/bcsstk01.tar.gz |
| nist/pores_1 | NIST Matrix Market | matrix_market_gz | 30x30 | 107 | 0.001 | yes | data/raw/nist_matrix_market/NIST/pores_1.mtx.gz |
