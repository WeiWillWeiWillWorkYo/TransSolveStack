# CSR Transformer Model Replay

- status: `passed`
- schema_version: `phase1_csr_transformer_model_replay_v1`
- model_family: `masked_self_attention_ranker_v1`
- model_id: `csr_masked_self_attention_ranker_v1`
- model_loaded: `True`
- model_trained: `True`
- runtime_selector_changed: `False`
- predictions: `20`
- reference_predictions: `20`
- exact_replay: `True`
- selected_candidate_mismatch_count: `0`
- ranked_order_mismatch_count: `0`
- evaluation_status_mismatch_count: `0`
- missing_reference_count: `0`
- max_abs_score_delta: `0`
- prediction_contract_valid: `True`

| request | selected_match | rank_match | status_match | max_score_delta |
|---|---|---|---|---:|
| eval:suitesparse:FIDAP/ex5:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:HB/curtis54:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:HB/jgl009:phase1_csr_micro_campaign | True | True | True | 0 |
| eval:suitesparse:MathWorks/tomography:phase1_csr_micro_campaign | True | True | True | 0 |
| eval:suitesparse:Zitney/extr1b:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Bai/cdde1:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Goodwin/Goodwin_010:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Grund/b1_ss:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Gset/G17:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/bcsstk07:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/fs_183_1:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:HB/ibm32:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/lshp1009:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/young3c:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Hamrle/Hamrle1:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:JGD_Trefethen/Trefethen_20b:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Negre/dendrimer:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Oberwolfach/t2dal_e:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:SNAP/email-Eu-core:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:Sandia/oscil_dcop_01:phase1_csr_selector | True | True | True | 0 |
