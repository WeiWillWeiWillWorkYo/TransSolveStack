# CSR Transformer Handoff Bundle

- status: `passed`
- handoff_ready: `True`
- model_family: `csr_transformer_policy_v2`
- model_training_required: `True`
- model_trained: `False`
- runtime_selector_changed: `False`
- executes_gpu: `False`
- source_files: `24`
- training_model_requests: `105`
- training_global_candidates: `12`
- positive_success_rows: `8`
- ranker_model_id: `csr_blocked_gap_augmented_shadow_ranker_v1`
- ranker_eval_non_success_selection_count: `23`
- guard_actual_quality_gate_blocks: `5`
- guard_actual_runtime_selector_changes: `0`
- submission_shadow_ready: `True`
- submission_runtime_promotion_ready: `False`
- validation_error_count: `0`
- next_step: `run external large-scale Transformer training against this handoff bundle, then submit the resulting model through artifact, replay, quality-gate, acceptance, and guarded runtime checks`

| label | kind | exists | size_bytes |
|---|---|---|---:|
| training_integration_summary | source_metadata | True | 2634 |
| training_tensor_summary | source_metadata | True | 853 |
| training_tensors | training_input | True | 468432 |
| request_index | training_input | True | 36949 |
| model_requests | training_input | True | 181335 |
| model_targets | training_input | True | 122708 |
| selector_rows | training_input | True | 718035 |
| positive_membership | source_metadata | True | 3627 |
| training_manifest | source_metadata | True | 3413 |
| augmented_ranker_summary | shadow_baseline | True | 3895 |
| augmented_ranker_model | shadow_baseline | True | 185911 |
| augmented_ranker_predictions | shadow_baseline | True | 102920 |
| augmented_ranker_positive_predictions | shadow_baseline | True | 6747 |
| augmented_ranker_manifest | shadow_baseline | True | 2372 |
| guarded_replay_summary | runtime_safety | True | 2281 |
| guarded_replay_rows | runtime_safety | True | 9975 |
| guarded_replay_blocked_quality_gate | runtime_safety | True | 722 |
| guarded_replay_counterfactual_quality_gate | runtime_safety | True | 669 |
| guarded_replay_manifest | runtime_safety | True | 2563 |
| model_submission_summary | contribution_contract | True | 1379 |
| model_submission_schema | contribution_contract | True | 654 |
| model_submission_manifest | contribution_contract | True | 4059 |
| model_contribution_terms | contribution_contract | True | 1791 |
| contributor_license_agreement | contribution_contract | True | 1490 |
