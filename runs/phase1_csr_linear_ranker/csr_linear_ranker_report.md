# CSR Linear Ranker

- status: `passed`
- schema_version: `phase1_csr_linear_ranker_v1`
- model_family: `pairwise_linear_ranker_v1`
- model_id: `csr_pairwise_linear_ranker_v1`
- model_trained: `True`
- runtime_selector_changed: `False`
- requests: `12`
- predictions: `12`
- train/eval requests: `9` / `3`
- train/eval oracle requests: `2` / `2`
- global_candidates: `9`
- features: `374`
- epochs: `80`
- pairwise_constraints: `110`
- pairwise_updates: `148`
- final_train_pairwise_loss: `1.7549`
- train_oracle_top1_accuracy: `0.5`
- eval_oracle_top1_accuracy: `0`
- eval_oracle_top1_accuracy_all_requests: `0`
- eval_profiled_success_selection_rate: `0`
- eval_non_success_selection_count: `3`
- eval_mean_regret_ms: ``
- eval_max_regret_ms: ``

| split | matrix | selected | oracle | status | oracle_rank | regret_ms |
|---|---|---|---|---|---:|---:|
| eval | suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 4 |  |
| eval | suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart8_float64 | taichi_csr_bicgstab_none_float64 | non_success_selected | 4 |  |
| eval | suitesparse:Zitney/extr1b | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Bai/cdde1 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Goodwin/Goodwin_010 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart8_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 2 | 12.9477 |
| train | suitesparse:HB/fs_183_1 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/young3c | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Hamrle/Hamrle1 | taichi_csr_gmres_jacobi_restart8_float64 |  | no_oracle_target |  |  |
| train | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:Negre/dendrimer | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Sandia/oscil_dcop_01 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
