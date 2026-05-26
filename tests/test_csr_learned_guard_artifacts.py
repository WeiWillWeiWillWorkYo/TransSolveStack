import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_csr_learned_guard_artifacts_are_valid():
    root = Path("runs/phase1_csr_learned_guard")
    rows = read_jsonl(root / "csr_learned_guard_rows.jsonl")
    summary = json.loads((root / "csr_learned_guard_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "csr_learned_guard_schema.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "csr_learned_runtime_guard"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_csr_learned_runtime_guard_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert schema["default_mode"] == "shadow"
    assert schema["runtime_guard"]["preemptive_gpu_kill_supported"] is False
    assert summary["num_checks"] == 4
    assert summary["shadow_mode_checked"] is True
    assert summary["quality_gate_blocks_checked"] is True
    assert summary["confidence_threshold_checked"] is True
    assert summary["fallback_chain_enforced_checked"] is True
    assert summary["runtime_fallback_checked"] is True
    assert summary["timeout_guard_checked"] is True
    assert summary["runtime_selector_changed"] is False

    rows_by_id = {row["check_id"]: row for row in rows}
    assert rows_by_id["actual_shadow_mode"]["guard_status"] == "shadow_only"
    assert rows_by_id["actual_quality_gate_blocks_promotion"]["guard_status"] == "blocked_quality_gate"
    assert rows_by_id["eligible_high_confidence_promotion_fixture"]["guard_status"] == "promoted"
    assert rows_by_id["runtime_timeout_fallback_fixture"]["result_status"] == "fallback_success"
    assert manifest.metadata["runtime_selector_changed"] is False
