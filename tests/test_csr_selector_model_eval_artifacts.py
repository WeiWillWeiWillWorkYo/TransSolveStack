import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_selector_model_eval_artifacts_are_valid():
    root = Path("runs/phase1_csr_selector_model_eval")
    rows = read_jsonl(root / "csr_selector_model_eval_rows.jsonl")
    summary = json.loads((root / "csr_selector_model_eval_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_selector_model_eval_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_selector_model_quality_gate"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_selector_model_eval_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["integration_boundary"]["status"] == "offline_quality_gate_only"
    assert schema["runtime_selector_changed"] is False
    assert summary["runtime_selector_changed"] is False
    assert summary["baseline_model_id"] == "candidate_prior_success_median_v1"
    assert summary["challenger_model_id"] == "csr_pairwise_linear_ranker_v1"
    assert summary["best_offline_model_id"] == "candidate_prior_success_median_v1"
    assert summary["runtime_selected_model_id"] is None
    assert summary["num_models"] == 2
    assert summary["num_eval_predictions"] == 3
    assert summary["num_eval_oracle_requests"] == 2
    assert summary["challenger_beats_baseline"] is False
    assert summary["challenger_runtime_eligible"] is False
    assert set(summary["challenger_gate_failures"]) == {
        "insufficient_eval_oracle_requests:2<10",
        "below_min_oracle_top1",
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }

    rows_by_role = {row["comparison_role"]: row for row in rows}
    assert set(rows_by_role) == {"baseline", "challenger"}
    baseline = rows_by_role["baseline"]
    challenger = rows_by_role["challenger"]
    assert baseline["model_id"] == summary["baseline_model_id"]
    assert challenger["model_id"] == summary["challenger_model_id"]
    assert baseline["model_trained"] is False
    assert challenger["model_trained"] is True
    assert baseline["runtime_eligible"] is False
    assert challenger["runtime_eligible"] is False
    assert challenger["beats_baseline"] is False
    assert baseline["eval_oracle_top1_accuracy"] == 1.0 / 3.0
    assert baseline["eval_profiled_success_selection_rate"] == 2.0 / 3.0
    assert baseline["eval_non_success_selection_count"] == 1
    assert challenger["eval_oracle_top1_accuracy"] == 0.0
    assert challenger["eval_profiled_success_selection_rate"] == 0.0
    assert challenger["eval_non_success_selection_count"] == 3
