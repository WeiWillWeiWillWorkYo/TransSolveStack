from pathlib import Path
import json

from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_phase1_campaign_artifacts_are_valid():
    root = Path("runs/phase1_campaign")
    summary = json.loads((root / "campaign_summary.json").read_text(encoding="utf-8"))
    manifest = read_manifest(root / "artifact_manifest.json")
    assert summary["campaign_id"] == "phase1_readiness"
    assert summary["status"] == "passed"
    stage_ids = {stage["stage_id"] for stage in summary["stages"]}
    queue_stage_ids = {
        stage_id
        for stage_id in stage_ids
        if stage_id.startswith("csr_queue_batch_")
        and stage_id.removeprefix("csr_queue_batch_").isdigit()
    }
    assert len(summary["stages"]) == 74 + len(queue_stage_ids)
    assert manifest.artifact_kind == "phase1_campaign_report"
    assert manifest.metadata["status"] == "passed"
    assert manifest.metadata["num_stages"] == len(summary["stages"])
    assert verify_manifest_hashes(manifest) == ()
    assert all(stage["status"] == "passed" for stage in summary["stages"])
