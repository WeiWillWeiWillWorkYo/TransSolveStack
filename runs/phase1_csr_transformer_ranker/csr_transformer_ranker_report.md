# CSR Transformer Ranker

- status: `passed`
- schema_version: `phase1_csr_transformer_ranker_v1`
- model_family: `masked_self_attention_ranker_v1`
- model_id: `csr_masked_self_attention_ranker_v1`
- model_trained: `True`
- runtime_selector_changed: `False`
- encoder_training: `deterministic_masked_self_attention_encoder_with_trained_pairwise_head`
- requests: `20`
- train/eval requests: `15` / `5`
- train/eval oracle requests: `4` / `4`
- global_candidates: `9`
- token_feature_dim: `37`
- d_model: `24`
- attention_heads: `4`
- feedforward_dim: `48`
- scorer_feature_dim: `62`
- epochs: `160`
- pairwise_constraints: `116`
- pairwise_updates: `3264`
- final_train_pairwise_loss: `15.7667`
- train_oracle_top1_accuracy: `0.5`
- eval_oracle_top1_accuracy: `0.5`
- eval_profiled_success_selection_rate: `0.6`
- eval_non_success_selection_count: `2`
- eval_mean_regret_ms: `36.5636`
- eval_max_regret_ms: `109.691`

| split | matrix | selected | oracle | status | oracle_rank | regret_ms |
|---|---|---|---|---|---:|---:|
| eval | suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 3 |  |
| eval | suitesparse:HB/curtis54 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | oracle_match | 1 | 0 |
| eval | suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 3 | 109.691 |
| eval | suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | oracle_match | 1 | 0 |
| eval | suitesparse:Zitney/extr1b | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart8_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Goodwin/Goodwin_010 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart8_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 3 | 12.9477 |
| train | suitesparse:Gset/G17 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk07 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/fs_183_1 | taichi_csr_gmres_jacobi_restart8_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/lshp1009 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/young3c | taichi_csr_gmres_jacobi_restart16_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Hamrle/Hamrle1 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:Negre/dendrimer | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Oberwolfach/t2dal_e | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 2 | 154.067 |
| train | suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Sandia/oscil_dcop_01 | taichi_csr_gmres_jacobi_restart16_float64 |  | no_oracle_target |  |  |
