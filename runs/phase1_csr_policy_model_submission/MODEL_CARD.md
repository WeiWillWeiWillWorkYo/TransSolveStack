# CSR Policy Model Submission

## Identity

- submission_id: `wei_cui_reference:csr_masked_self_attention_ranker_v1_reference_submission:phase1-reference`
- contributor: `Wei CUI`
- contribution_name: `csr_masked_self_attention_ranker_v1_reference_submission`
- contribution_version: `phase1-reference`

## Model

- model_id: `csr_masked_self_attention_ranker_v1`
- model_family: `masked_self_attention_ranker_v1`
- adapter: `csr_transformer_ranker_saved_model_v1`
- training_statement: Reference CSR Transformer ranker trained on the current Phase 1 Transformer-ready SuiteSparse subset.

## Acceptance

- accepted_for_shadow: `True`
- accepted_for_runtime_promotion: `False`
- guarded_gpu_shadow_smoke_checked: `True`
- guarded_gpu_smoke_successes: `2`

## Promotion Blockers

- `below_min_profiled_success_rate`
- `below_baseline_oracle_top1`
- `below_baseline_profiled_success_rate`
- `non_success_eval_selections`
- `quality_gate_not_runtime_eligible`

## Runtime Boundary

- default_runtime_mode: `shadow`
- guard_required: `True`
- promotion_requires_acceptance_gate: `True`
- promotion_requires_quality_gate: `True`
- runtime_selector_changed: `False`

## Files

- `policy_model_artifact`: `runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json` sha256 `1ddbc0c59b01c3778be9ff81f9ea30cdaf6e6a2e6c03d2e7c2ac1e303f3c1163`
- `saved_model`: `runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json` sha256 `6adba9457a65c06fe94a54d43e0c732ef0bcbaa7efc161febb6635eeaf5272bd`
- `transformer_tensors`: `runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json` sha256 `27c3c99dd7706e596f40ea260bd0a33765fb4eddae13a7a470200a04417134be`
- `request_index`: `runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl` sha256 `7a76140b19e8d9ff33b753193aa6011e2f6ee392b96a3f3feb179731963a1b27`
- `quality_gate_summary`: `runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json` sha256 `d7353ef33e5809d8681e218737b9ab9d2e55f6a2c78022bb23dd90aa7a2c8c9c`
- `replay_summary`: `runs/phase1_csr_transformer_model_replay/csr_transformer_model_replay_summary.json` sha256 `d8147f7a66ba620a81a49f8d9433d0f80b9885046ccc528d8def2d2464449575`
- `acceptance_summary`: `runs/phase1_csr_policy_model_acceptance/csr_policy_model_acceptance_summary.json` sha256 `9339712d346d7c86478ec9c00f45fc3c79b01b4b7d5195cc63c5e524d5ee51aa`
- `acceptance_rows`: `runs/phase1_csr_policy_model_acceptance/csr_policy_model_acceptance_rows.jsonl` sha256 `e029c599311bc912cdc3f646b5343eb520a20fa5029cf890b9e83d4025cfc03d`

## Terms

- terms_acknowledged: `True`
- required_terms_documents: `MODEL_CONTRIBUTION_TERMS.md`, `CONTRIBUTOR_LICENSE_AGREEMENT.md`
