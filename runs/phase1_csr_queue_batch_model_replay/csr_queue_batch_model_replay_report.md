# CSR Transformer Model Replay

- status: `passed`
- schema_version: `phase1_csr_transformer_model_replay_v1`
- model_family: `masked_self_attention_ranker_v1`
- model_id: `csr_queue_batch_shadow_ranker_v1`
- model_loaded: `True`
- model_trained: `True`
- runtime_selector_changed: `False`
- predictions: `100`
- reference_predictions: `100`
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
| eval:suitesparse:HB/bcsstk10:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| eval:suitesparse:HB/bcsstm09:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| eval:suitesparse:HB/bcsstm13:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| eval:suitesparse:HB/bcsstm25:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| eval:suitesparse:HB/bcsstm27:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| eval:suitesparse:HB/blckhole:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| eval:suitesparse:HB/bp_0:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| eval:suitesparse:HB/bp_1000:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| eval:suitesparse:HB/bp_1200:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_1400:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_1600:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_200:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_400:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_600:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/bp_800:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/can_1054:phase1_csr_queue_batch_00010 | True | True | True | 0 |
| eval:suitesparse:HB/fs_183_1:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:HB/lshp1009:phase1_csr_micro_campaign | True | True | True | 0 |
| eval:suitesparse:HB/young3c:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:Hamrle/Hamrle1:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:Negre/dendrimer:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:SNAP/email-Eu-core:phase1_csr_micro_campaign | True | True | True | 0 |
| eval:suitesparse:Sandia/oscil_dcop_01:phase1_csr_selector | True | True | True | 0 |
| eval:suitesparse:Zitney/extr1b:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Bai/cdde1:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Goodwin/Goodwin_010:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Grund/b1_ss:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:Gset/G17:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/1138_bus:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/494_bus:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/662_bus:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/685_bus:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/arc130:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/ash292:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/ash85:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr01:phase1_csr_queue_batch_00001 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr02:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr03:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr04:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr05:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr06:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr07:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr08:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr09:phase1_csr_queue_batch_00002 | True | True | True | 0 |
| train:suitesparse:HB/bcspwr10:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk01:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk02:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk03:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk04:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk05:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk06:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk07:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/bcsstk07:phase1_csr_queue_batch_00003 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk08:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk09:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk11:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk12:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk13:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk14:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk15:phase1_csr_queue_batch_00004 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk16:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk17:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk18:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk19:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk20:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk21:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk22:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk23:phase1_csr_queue_batch_00005 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk24:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk25:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk26:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk27:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk28:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk29:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstk33:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm01:phase1_csr_queue_batch_00006 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm02:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm03:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm04:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm05:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm06:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm07:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm08:phase1_csr_queue_batch_00007 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm10:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm11:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm12:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm19:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm20:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm21:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm22:phase1_csr_queue_batch_00008 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm23:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm24:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| train:suitesparse:HB/bcsstm26:phase1_csr_queue_batch_00009 | True | True | True | 0 |
| train:suitesparse:HB/curtis54:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:HB/ibm32:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:HB/jgl009:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:JGD_Trefethen/Trefethen_20b:phase1_csr_selector | True | True | True | 0 |
| train:suitesparse:MathWorks/tomography:phase1_csr_micro_campaign | True | True | True | 0 |
| train:suitesparse:Oberwolfach/t2dal_e:phase1_csr_micro_campaign | True | True | True | 0 |
