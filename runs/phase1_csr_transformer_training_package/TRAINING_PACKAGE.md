# CSR Transformer Training Package

- package_id: `csr_transformer_policy_v2_augmented_handoff_package`
- status: `passed`
- package_ready: `True`
- model_trained: `False`
- runtime_selector_changed: `False`

## Inputs

- `tensor_arrays`: `runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_tensors.json`
- `request_index`: `runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_request_index.jsonl`
- `label_free_model_requests`: `runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_model_requests.jsonl`
- `offline_targets`: `runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_model_targets.jsonl`
- `selector_rows`: `runs/phase1_csr_blocked_gap_training_integration/combined_csr_selector_rows.jsonl`
- `handoff_summary`: `runs/phase1_csr_transformer_handoff_bundle/csr_transformer_handoff_bundle_summary.json`
- `training_spec`: `runs/phase1_csr_transformer_handoff_bundle/csr_transformer_handoff_training_spec.json`
- `guard_contract`: `runs/phase1_csr_transformer_handoff_bundle/csr_transformer_handoff_guard_contract.json`
- `handoff_schema`: `runs/phase1_csr_transformer_handoff_bundle/csr_transformer_handoff_bundle_schema.json`
- `handoff_manifest`: `runs/phase1_csr_transformer_handoff_bundle/artifact_manifest.json`
- `shadow_ranker_summary`: `runs/phase1_csr_blocked_gap_augmented_ranker/csr_blocked_gap_augmented_ranker_summary.json`
- `shadow_ranker_predictions`: `runs/phase1_csr_blocked_gap_augmented_ranker/csr_blocked_gap_augmented_ranker_predictions.jsonl`
- `guarded_replay_summary`: `runs/phase1_csr_blocked_gap_guarded_replay/csr_blocked_gap_guarded_replay_summary.json`
- `model_submission_contract`: `runs/phase1_csr_policy_model_submission/csr_policy_model_submission_schema.json`
- `model_contribution_terms`: `MODEL_CONTRIBUTION_TERMS.md`
- `contributor_license_agreement`: `CONTRIBUTOR_LICENSE_AGREEMENT.md`

## Validation Steps

1. `saved_model_replay`: `.venv-tss/bin/python scripts/tss_csr_transformer_model_replay.py --model future_csr_transformer_policy_model.json --tensors runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_tensors.json --request-index runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_request_index.jsonl --reference-predictions future_csr_transformer_policy_predictions.jsonl --out runs/future_csr_transformer_model_replay`
2. `quality_gate`: `.venv-tss/bin/python scripts/tss_csr_selector_model_eval.py --baseline-summary runs/phase1_csr_learning_readiness/csr_learning_summary.json --baseline-predictions runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl --ranker-summary future_csr_transformer_quality_gate_summary.json --ranker-predictions future_csr_transformer_policy_predictions.jsonl --out runs/future_csr_selector_model_eval`
3. `policy_model_artifact`: `.venv-tss/bin/python scripts/tss_csr_policy_model_artifact.py --model future_csr_transformer_policy_model.json --tensors runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_tensors.json --request-index runs/phase1_csr_blocked_gap_training_integration/csr_blocked_gap_training_request_index.jsonl --quality-gate-summary future_csr_transformer_quality_gate_summary.json --replay-summary future_csr_transformer_policy_replay_summary.json --out runs/future_csr_policy_model_artifact`
4. `policy_model_acceptance`: `.venv-tss/bin/python scripts/tss_csr_policy_model_acceptance.py --model-artifact future_csr_policy_model_artifact.json --out runs/future_csr_policy_model_acceptance`
5. `policy_model_submission`: `.venv-tss/bin/python scripts/tss_csr_policy_model_submission.py --model-artifact future_csr_policy_model_artifact.json --acceptance-summary runs/future_csr_policy_model_acceptance/csr_policy_model_acceptance_summary.json --acceptance-rows runs/future_csr_policy_model_acceptance/csr_policy_model_acceptance_rows.jsonl --contributor-name '<name>' --contribution-name '<model-name>' --out runs/future_csr_policy_model_submission`
6. `guarded_shadow_smoke`: `.venv-tss/bin/python scripts/tss_csr_learned_guard_smoke.py --learned-model-artifact future_csr_policy_model_artifact.json --out runs/future_csr_learned_guard`

## Boundary

- This package does not train a model.
- This package does not change runtime selection.
- A trained model must pass every validation step before runtime use.
