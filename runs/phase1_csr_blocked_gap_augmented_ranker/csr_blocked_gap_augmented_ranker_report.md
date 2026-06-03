# CSR Blocked Gap Augmented Ranker

- status: `passed`
- shadow_only: `True`
- runtime_selector_changed: `False`
- model_id: `csr_blocked_gap_augmented_shadow_ranker_v1`
- ranker_requests: `105`
- ranker_global_candidates: `12`
- blocked_gap_prediction_rows: `5`
- blocked_gap_success_predictions: `5`
- blocked_gap_oracle_matches: `4`
- blocked_gap_eval_success_selection_rate: `1`
- blocked_gap_positive_candidate_coverage_complete: `True`
- ranker_eval_profiled_success_selection_rate: `0.115385`
- ranker_eval_non_success_selection_count: `23`
- acceptance_note: `This artifact gates only blocked-gap training-shard behavior; runtime promotion remains behind learned guard acceptance.`
- next_step: `compare augmented ranker replay against guarded fallback policy and prepare a transformer training handoff that treats this model as a shadow baseline, not a production policy`
