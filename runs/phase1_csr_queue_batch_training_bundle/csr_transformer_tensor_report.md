# CSR Training Tensor Export

- status: `passed`
- schema_version: `phase1_csr_training_tensors_v1`
- source_contract_schema_version: `phase1_csr_model_contract_v1`
- storage_format: `json_numeric_arrays_v1`
- requests: `100`
- train_requests: `75`
- eval_requests: `25`
- global_candidates: `9`
- active_candidate_slots: `372`
- matrix_feature_dim: `21`
- candidate_feature_dim: `16`
- oracle_targets: `32`
- requests_without_oracle: `68`
- label_class_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 223, 'success_non_oracle': 39, 'success_oracle': 32}`
- target_status_counts: `{'not_applicable': 34, 'not_profiled': 44, 'screened_out': 223, 'success': 71}`
- has_missing_candidate_slots: `True`
- model_required: `False`
- runtime_selector_changed: `False`

| row | split | matrix | oracle_index | active_candidates |
|---:|---|---|---:|---:|
| 0 | eval | suitesparse:FIDAP/ex5 | 7 | 9 |
| 1 | eval | suitesparse:HB/bcsstk10 | 7 | 3 |
| 2 | eval | suitesparse:HB/bcsstm09 | 8 | 3 |
| 3 | eval | suitesparse:HB/bcsstm13 | -1 | 3 |
| 4 | eval | suitesparse:HB/bcsstm25 | 8 | 3 |
| 5 | eval | suitesparse:HB/bcsstm27 | -1 | 3 |
| 6 | eval | suitesparse:HB/blckhole | -1 | 3 |
| 7 | eval | suitesparse:HB/bp_0 | -1 | 3 |
| 8 | eval | suitesparse:HB/bp_1000 | -1 | 3 |
| 9 | eval | suitesparse:HB/bp_1200 | -1 | 3 |
| 10 | eval | suitesparse:HB/bp_1400 | -1 | 3 |
| 11 | eval | suitesparse:HB/bp_1600 | -1 | 3 |
| 12 | eval | suitesparse:HB/bp_200 | -1 | 3 |
| 13 | eval | suitesparse:HB/bp_400 | -1 | 3 |
| 14 | eval | suitesparse:HB/bp_600 | -1 | 3 |
| 15 | eval | suitesparse:HB/bp_800 | -1 | 3 |
| 16 | eval | suitesparse:HB/can_1054 | -1 | 3 |
| 17 | eval | suitesparse:HB/fs_183_1 | -1 | 9 |
| 18 | eval | suitesparse:HB/lshp1009 | -1 | 3 |
| 19 | eval | suitesparse:HB/young3c | -1 | 9 |
| 20 | eval | suitesparse:Hamrle/Hamrle1 | -1 | 9 |
| 21 | eval | suitesparse:Negre/dendrimer | -1 | 9 |
| 22 | eval | suitesparse:SNAP/email-Eu-core | -1 | 3 |
| 23 | eval | suitesparse:Sandia/oscil_dcop_01 | -1 | 9 |
| 24 | eval | suitesparse:Zitney/extr1b | -1 | 9 |
| 25 | train | suitesparse:Bai/cdde1 | -1 | 9 |
| 26 | train | suitesparse:Goodwin/Goodwin_010 | -1 | 9 |
| 27 | train | suitesparse:Grund/b1_ss | 4 | 9 |
| 28 | train | suitesparse:Gset/G17 | -1 | 3 |
| 29 | train | suitesparse:HB/1138_bus | -1 | 3 |
| 30 | train | suitesparse:HB/494_bus | -1 | 3 |
| 31 | train | suitesparse:HB/662_bus | 7 | 3 |
| 32 | train | suitesparse:HB/685_bus | 7 | 3 |
| 33 | train | suitesparse:HB/arc130 | -1 | 3 |
| 34 | train | suitesparse:HB/ash292 | -1 | 3 |
| 35 | train | suitesparse:HB/ash85 | -1 | 3 |
| 36 | train | suitesparse:HB/bcspwr01 | -1 | 3 |
| 37 | train | suitesparse:HB/bcspwr02 | -1 | 3 |
| 38 | train | suitesparse:HB/bcspwr03 | -1 | 3 |
| 39 | train | suitesparse:HB/bcspwr04 | -1 | 3 |
| 40 | train | suitesparse:HB/bcspwr05 | -1 | 3 |
| 41 | train | suitesparse:HB/bcspwr06 | -1 | 3 |
| 42 | train | suitesparse:HB/bcspwr07 | -1 | 3 |
| 43 | train | suitesparse:HB/bcspwr08 | -1 | 3 |
| 44 | train | suitesparse:HB/bcspwr09 | -1 | 3 |
| 45 | train | suitesparse:HB/bcspwr10 | -1 | 3 |
| 46 | train | suitesparse:HB/bcsstk01 | -1 | 3 |
| 47 | train | suitesparse:HB/bcsstk02 | 2 | 3 |
| 48 | train | suitesparse:HB/bcsstk03 | 7 | 3 |
| 49 | train | suitesparse:HB/bcsstk04 | -1 | 3 |
| 50 | train | suitesparse:HB/bcsstk05 | 7 | 3 |
| 51 | train | suitesparse:HB/bcsstk06 | -1 | 3 |
| 52 | train | suitesparse:HB/bcsstk07 | -1 | 3 |
| 53 | train | suitesparse:HB/bcsstk07 | -1 | 3 |
| 54 | train | suitesparse:HB/bcsstk08 | -1 | 3 |
| 55 | train | suitesparse:HB/bcsstk09 | 2 | 3 |
| 56 | train | suitesparse:HB/bcsstk11 | -1 | 3 |
| 57 | train | suitesparse:HB/bcsstk12 | -1 | 3 |
| 58 | train | suitesparse:HB/bcsstk13 | -1 | 3 |
| 59 | train | suitesparse:HB/bcsstk14 | -1 | 3 |
| 60 | train | suitesparse:HB/bcsstk15 | -1 | 3 |
| 61 | train | suitesparse:HB/bcsstk16 | 7 | 3 |
| 62 | train | suitesparse:HB/bcsstk17 | -1 | 3 |
| 63 | train | suitesparse:HB/bcsstk18 | -1 | 3 |
| 64 | train | suitesparse:HB/bcsstk19 | -1 | 3 |
| 65 | train | suitesparse:HB/bcsstk20 | -1 | 3 |
| 66 | train | suitesparse:HB/bcsstk21 | -1 | 3 |
| 67 | train | suitesparse:HB/bcsstk22 | -1 | 3 |
| 68 | train | suitesparse:HB/bcsstk23 | -1 | 3 |
| 69 | train | suitesparse:HB/bcsstk24 | -1 | 3 |
| 70 | train | suitesparse:HB/bcsstk25 | -1 | 3 |
| 71 | train | suitesparse:HB/bcsstk26 | -1 | 3 |
| 72 | train | suitesparse:HB/bcsstk27 | 7 | 3 |
| 73 | train | suitesparse:HB/bcsstk28 | -1 | 3 |
| 74 | train | suitesparse:HB/bcsstk29 | -1 | 3 |
| 75 | train | suitesparse:HB/bcsstk33 | -1 | 3 |
| 76 | train | suitesparse:HB/bcsstm01 | -1 | 3 |
| 77 | train | suitesparse:HB/bcsstm02 | 7 | 3 |
| 78 | train | suitesparse:HB/bcsstm03 | -1 | 3 |
| 79 | train | suitesparse:HB/bcsstm04 | -1 | 3 |
| 80 | train | suitesparse:HB/bcsstm05 | 8 | 3 |
| 81 | train | suitesparse:HB/bcsstm06 | 8 | 3 |
| 82 | train | suitesparse:HB/bcsstm07 | 8 | 3 |
| 83 | train | suitesparse:HB/bcsstm08 | 8 | 3 |
| 84 | train | suitesparse:HB/bcsstm10 | -1 | 3 |
| 85 | train | suitesparse:HB/bcsstm11 | 8 | 3 |
| 86 | train | suitesparse:HB/bcsstm12 | -1 | 3 |
| 87 | train | suitesparse:HB/bcsstm19 | 7 | 3 |
| 88 | train | suitesparse:HB/bcsstm20 | 8 | 3 |
| 89 | train | suitesparse:HB/bcsstm21 | 2 | 3 |
| 90 | train | suitesparse:HB/bcsstm22 | 8 | 3 |
| 91 | train | suitesparse:HB/bcsstm23 | 8 | 3 |
| 92 | train | suitesparse:HB/bcsstm24 | 8 | 3 |
| 93 | train | suitesparse:HB/bcsstm26 | 8 | 3 |
| 94 | train | suitesparse:HB/curtis54 | 1 | 9 |
| 95 | train | suitesparse:HB/ibm32 | 1 | 3 |
| 96 | train | suitesparse:HB/jgl009 | 4 | 3 |
| 97 | train | suitesparse:JGD_Trefethen/Trefethen_20b | 3 | 9 |
| 98 | train | suitesparse:MathWorks/tomography | 4 | 3 |
| 99 | train | suitesparse:Oberwolfach/t2dal_e | 8 | 3 |
