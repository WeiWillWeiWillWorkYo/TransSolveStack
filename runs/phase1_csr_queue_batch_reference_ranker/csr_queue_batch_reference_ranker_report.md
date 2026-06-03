# CSR Transformer Ranker

- status: `passed`
- schema_version: `phase1_csr_transformer_ranker_v1`
- model_family: `masked_self_attention_ranker_v1`
- model_id: `csr_queue_batch_shadow_ranker_v1`
- model_trained: `True`
- runtime_selector_changed: `False`
- encoder_training: `deterministic_masked_self_attention_encoder_with_trained_pairwise_head`
- requests: `100`
- train/eval requests: `75` / `25`
- train/eval oracle requests: `28` / `4`
- global_candidates: `9`
- token_feature_dim: `37`
- d_model: `24`
- attention_heads: `4`
- feedforward_dim: `48`
- scorer_feature_dim: `62`
- epochs: `160`
- pairwise_constraints: `155`
- pairwise_updates: `6323`
- final_train_pairwise_loss: `43.9943`
- train_oracle_top1_accuracy: `0.678571`
- eval_oracle_top1_accuracy: `0.5`
- eval_profiled_success_selection_rate: `0.12`
- eval_non_success_selection_count: `22`
- eval_mean_regret_ms: `81.5147`
- eval_max_regret_ms: `244.544`

| split | matrix | selected | oracle | status | oracle_rank | regret_ms |
|---|---|---|---|---|---:|---:|
| eval | suitesparse:FIDAP/ex5 | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 9 |  |
| eval | suitesparse:HB/bcsstk10 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 1 | 0 |
| eval | suitesparse:HB/bcsstm09 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| eval | suitesparse:HB/bcsstm13 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bcsstm25 | taichi_csr_pcg_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | profiled_success_non_oracle | 3 | 244.544 |
| eval | suitesparse:HB/bcsstm27 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/blckhole | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_0 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_1000 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_1200 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_1400 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_1600 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_200 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_400 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_600 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/bp_800 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/can_1054 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/fs_183_1 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/lshp1009 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:HB/young3c | taichi_csr_gmres_jacobi_restart16_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:Hamrle/Hamrle1 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:Negre/dendrimer | taichi_csr_gmres_jacobi_restart32_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:SNAP/email-Eu-core | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:Sandia/oscil_dcop_01 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| eval | suitesparse:Zitney/extr1b | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Bai/cdde1 | taichi_csr_gmres_jacobi_restart32_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Goodwin/Goodwin_010 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:Grund/b1_ss | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | oracle_match | 1 | 0 |
| train | suitesparse:Gset/G17 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/1138_bus | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/494_bus | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/662_bus | taichi_csr_richardson_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 2 |  |
| train | suitesparse:HB/685_bus | taichi_csr_richardson_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 2 |  |
| train | suitesparse:HB/arc130 | taichi_csr_bicgstab_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/ash292 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/ash85 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr01 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr02 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr03 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr04 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr05 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr06 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr07 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr08 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr09 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcspwr10 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk01 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk02 | taichi_csr_cg_none_float64 | taichi_csr_cg_none_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstk03 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstk04 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk05 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstk06 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk07 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk08 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk09 | taichi_csr_pcg_jacobi_float64 | taichi_csr_cg_none_float64 | profiled_success_non_oracle | 2 | 260.539 |
| train | suitesparse:HB/bcsstk11 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk12 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk13 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk14 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk15 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk16 | taichi_csr_richardson_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | non_success_selected | 2 |  |
| train | suitesparse:HB/bcsstk17 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk18 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk19 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk20 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk21 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk22 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk23 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk24 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk25 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk26 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk27 | taichi_csr_pcg_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstk28 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk29 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstk33 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm01 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm02 | taichi_csr_richardson_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | profiled_success_non_oracle | 2 | 58.014 |
| train | suitesparse:HB/bcsstm03 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm04 | taichi_csr_richardson_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm05 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm06 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm07 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm08 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm10 | taichi_csr_pcg_jacobi_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm11 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm12 | taichi_csr_cg_none_float64 |  | no_oracle_target |  |  |
| train | suitesparse:HB/bcsstm19 | taichi_csr_richardson_jacobi_float64 | taichi_csr_pcg_jacobi_float64 | profiled_success_non_oracle | 2 | 115.075 |
| train | suitesparse:HB/bcsstm20 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm21 | taichi_csr_richardson_jacobi_float64 | taichi_csr_cg_none_float64 | profiled_success_non_oracle | 3 | 57.9079 |
| train | suitesparse:HB/bcsstm22 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm23 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm24 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/bcsstm26 | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/curtis54 | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_bicgstab_none_float64 | profiled_success_non_oracle | 5 | 4753.09 |
| train | suitesparse:HB/ibm32 | taichi_csr_bicgstab_none_float64 | taichi_csr_bicgstab_none_float64 | oracle_match | 1 | 0 |
| train | suitesparse:HB/jgl009 | taichi_csr_bicgstab_none_float64 | taichi_csr_gmres_jacobi_restart16_float64 | profiled_success_non_oracle | 3 | 109.691 |
| train | suitesparse:JGD_Trefethen/Trefethen_20b | taichi_csr_chebyshev_jacobi_float64 | taichi_csr_chebyshev_jacobi_float64 | oracle_match | 1 | 0 |
| train | suitesparse:MathWorks/tomography | taichi_csr_gmres_jacobi_restart16_float64 | taichi_csr_gmres_jacobi_restart16_float64 | oracle_match | 1 | 0 |
| train | suitesparse:Oberwolfach/t2dal_e | taichi_csr_richardson_jacobi_float64 | taichi_csr_richardson_jacobi_float64 | oracle_match | 1 | 0 |
