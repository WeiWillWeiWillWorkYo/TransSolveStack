import json
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def test_public_release_hygiene_artifacts_are_valid():
    root = Path("runs/phase1_public_release_hygiene")
    rows = read_jsonl(root / "public_release_hygiene_rows.jsonl")
    summary = json.loads(
        (root / "public_release_hygiene_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / "public_release_hygiene_schema.json").read_text(encoding="utf-8")
    )
    manifest = read_manifest(root / "artifact_manifest.json")

    assert manifest.artifact_kind == "public_release_hygiene"
    assert verify_manifest_hashes(manifest) == ()
    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_public_release_hygiene_v1"
    assert schema["schema_version"] == summary["schema_version"]
    assert summary["public_release_hygiene_ready"] is True
    git_initialized = Path(".git").exists()
    assert summary["github_upload_ready"] is git_initialized
    assert summary["project_owner"] == "Wei CUI"
    assert summary["citation_author"] == "Wei CUI"
    assert summary["license_route_selected"] is True
    assert summary["formal_license_finalized"] is True
    assert summary["source_code_license"] == "PolyForm Noncommercial License 1.0.0"
    assert summary["documentation_license_direction"] == "CC BY-NC 4.0"
    assert summary["model_terms_documented"] is True
    assert summary["contributor_terms_documented"] is True
    assert summary["commercial_use_requires_permission"] is True
    assert summary["license_policy_documented"] is True
    assert summary["permissive_license_family_rejected"] is True
    assert summary["license_upload_blockers"] == (
        [] if git_initialized else ["git_init_pending"]
    )
    assert summary["runtime_selector_changed"] is False
    assert summary["git_repository_initialized"] is git_initialized
    assert summary["git_init_required"] is (not git_initialized)
    assert summary["missing_gitignore_patterns"] == []
    assert summary["large_unignored_file_count"] == 0
    assert summary["secret_like_file_count"] == 0
    assert summary["validation_error_count"] == 0
    assert len(rows) >= summary["public_source_path_count"]
    assert all(row["blocks_hygiene"] is False for row in rows)
    assert {
        "LICENSE",
        "docs/PROJECT_OVERVIEW.md",
        "LICENSE_POLICY.md",
        "COMMERCIAL_USE.md",
        "MODEL_CONTRIBUTION_TERMS.md",
        "CONTRIBUTOR_LICENSE_AGREEMENT.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
    }.issubset(set(summary["required_release_docs"]))
    assert {
        "ADVICE.md",
        "MILESTONE_LOG.md",
        "FOOTAGE_SPEC.md",
        "PUBLIC_RELEASE.md",
        "GITHUB_RELEASE_GATE.md",
        "footage/",
        "docs/toms_paper/",
        "reports/",
    }.issubset(set(summary["required_gitignore_patterns"]))
    assert "ADVICE.md" not in set(summary["public_source_paths"])
    assert "MILESTONE_LOG.md" not in set(summary["public_source_paths"])
    assert "FOOTAGE_SPEC.md" not in set(summary["public_source_paths"])
    assert "footage" not in set(summary["public_source_paths"])
    assert "docs" not in set(summary["public_source_paths"])
    assert "release_required_document" in {row["row_kind"] for row in rows}
    assert "license_decision" in {row["row_kind"] for row in rows}
    assert "citation" in {row["row_kind"] for row in rows}
    assert (root / "public_release_checklist.md").exists()
    assert (root / "public_release_hygiene_report.md").exists()
    assert manifest.metadata["public_release_hygiene_ready"] is True
    assert manifest.metadata["runtime_selector_changed"] is False
