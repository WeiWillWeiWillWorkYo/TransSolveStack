# CSR External Model Intake Review

- status: `passed`
- intake_ready: `True`
- shadow_submission_ready: `True`
- runtime_promotion_ready: `False`
- default_runtime_mode: `shadow`
- num_predictions: `20`
- guarded_gpu_shadow_smoke_checked: `True`
- guarded_gpu_smoke_successes: `2`
- runtime_selector_changed: `False`

## Stages

- adapter: `passed`
- replay: `passed`
- quality_gate: `passed`
- policy_model_artifact: `passed`
- learned_guard: `passed`
- acceptance: `passed`
- submission: `passed`

## Promotion Blockers

- `below_min_profiled_success_rate`
- `below_baseline_oracle_top1`
- `below_baseline_profiled_success_rate`
- `non_success_eval_selections`
- `quality_gate_not_runtime_eligible`
