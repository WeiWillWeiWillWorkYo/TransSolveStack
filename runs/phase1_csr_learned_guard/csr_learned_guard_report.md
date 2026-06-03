# CSR Learned Runtime Guard

- status: `passed`
- schema_version: `phase1_csr_learned_runtime_guard_v1`
- runtime_selector_changed: `False`
- preemptive_gpu_kill_supported: `False`
- saved_model_loaded_checks: `2`

| check | status | guard_status | runtime_source |
|---|---|---|---|
| actual_shadow_mode | passed | shadow_only | artifact |
| actual_quality_gate_blocks_promotion | passed | blocked_quality_gate | artifact |
| eligible_high_confidence_promotion_fixture | passed | promoted | learned |
| runtime_timeout_fallback_fixture | passed | success |  |
