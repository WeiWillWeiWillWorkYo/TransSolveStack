# CSR Transformer Training Entrypoint

- status: `passed`
- schema_version: `phase1_csr_transformer_training_entrypoint_v1`
- training_entrypoint_ready: `True`
- model_training_required: `True`
- model_trained: `False`
- runtime_selector_changed: `False`
- transformer_connectable: `True`
- current_quality_gate_runtime_eligible: `False`
- current_quality_gate_failures: `['below_min_profiled_success_rate', 'below_baseline_oracle_top1', 'below_baseline_profiled_success_rate', 'non_success_eval_selections']`
- selector_rows: `132`
- matrices: `20`
- model_requests: `20`
- tensor_requests: `20`
- active_candidate_slots: `132`
- matrix_feature_dim: `21`
- candidate_feature_dim: `16`
- validation_error_count: `0`
- next_step: `run_large_scale_transformer_training_then_quality_gate`
