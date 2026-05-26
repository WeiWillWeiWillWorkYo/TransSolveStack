import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_transformer_quality_gate_artifacts_are_valid():
    root = Path("runs/phase1_csr_transformer_quality_gate")
    rows = read_jsonl(root / "csr_transformer_quality_gate_rows.jsonl")
    summary = json.loads(
        (root / "csr_transformer_quality_gate_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "csr_transformer_quality_gate_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_transformer_quality_gate"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_selector_model_eval_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["baseline_model_id"] == "candidate_prior_success_median_v1"
    assert summary["challenger_model_id"] == "csr_masked_self_attention_ranker_v1"
    assert summary["best_offline_model_id"] == "candidate_prior_success_median_v1"
    assert summary["challenger_beats_baseline"] is False
    assert summary["challenger_runtime_eligible"] is False
    assert summary["runtime_selected_model_id"] is None
    assert summary["runtime_selector_changed"] is False
    assert set(summary["challenger_gate_failures"]) == {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    assert summary["recommendation"] == "keep_runtime_artifact_backed_and_expand_benchmark_coverage"
    assert summary["num_eval_predictions"] == 5
    assert summary["num_eval_oracle_requests"] == 4
    assert summary["min_required_eval_oracle_requests"] == 4
    assert summary["min_runtime_oracle_top1_accuracy"] == 0.5
    assert summary["min_runtime_profiled_success_rate"] == 0.8

    assert len(rows) == 2
    rows_by_role = {row["comparison_role"]: row for row in rows}
    assert set(rows_by_role) == {"baseline", "challenger"}
    baseline = rows_by_role["baseline"]
    challenger = rows_by_role["challenger"]
    assert baseline["runtime_eligible"] is False
    assert challenger["runtime_eligible"] is False
    assert challenger["beats_baseline"] is False
    assert baseline["eval_oracle_top1_accuracy"] == 0.6
    assert challenger["eval_oracle_top1_accuracy"] == 0.5
    assert baseline["eval_profiled_success_selection_rate"] == 0.8
    assert challenger["eval_profiled_success_selection_rate"] == 0.6
    assert challenger["eval_non_success_selection_count"] == 2
    assert set(challenger["gate_failures"]) == set(summary["challenger_gate_failures"])

    assert manifest.metadata["challenger_runtime_eligible"] is False
    assert manifest.metadata["best_offline_model_id"] == summary["best_offline_model_id"]
