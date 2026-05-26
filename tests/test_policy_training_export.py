from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.policies.training_data import build_policy_training_rows


def test_policy_training_rows_export_current_evaluation():
    workload = load_workload_config("configs/workloads/phase1_smoke.yaml")
    rows = build_policy_training_rows(
        workload,
        "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    assert len(rows) == 21
    assert sum(1 for row in rows if row.label_is_oracle) == 7
    assert {row.schema_version for row in rows} == {"phase1_policy_features_v1"}
    assert all(row.target_status == "success" for row in rows)
    assert all("estimated_effective_nnz" in row.features for row in rows)
    assert all("candidate_id" in row.features for row in rows)
