from pathlib import Path

from transsolvestack.profiling.public_release_hygiene import (
    REQUIRED_GITIGNORE_PATTERNS,
    build_public_release_hygiene_artifact,
)


def test_public_release_hygiene_scans_release_boundary(tmp_path):
    export = build_public_release_hygiene_artifact(output_dir=tmp_path)
    summary = export["summary"]
    git_initialized = Path(".git").exists()

    assert summary["status"] == "passed"
    assert summary["schema_version"] == "phase1_public_release_hygiene_v1"
    assert summary["public_release_hygiene_ready"] is True
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
        () if git_initialized else ("git_init_pending",)
    )
    assert summary["runtime_selector_changed"] is False
    assert summary["git_repository_initialized"] is git_initialized
    assert summary["git_init_required"] is (not git_initialized)
    assert summary["missing_gitignore_patterns"] == ()
    assert summary["large_unignored_file_count"] == 0
    assert summary["secret_like_file_count"] == 0
    assert summary["validation_error_count"] == 0
    assert set(REQUIRED_GITIGNORE_PATTERNS).issubset(
        set(summary["required_gitignore_patterns"])
    )
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

    rows_by_kind = {}
    for row in export["rows"]:
        rows_by_kind.setdefault(row["row_kind"], []).append(row)
        assert row["blocks_hygiene"] is False
    assert "gitignore_required_pattern" in rows_by_kind
    assert "public_source_path" in rows_by_kind
    assert "release_required_document" in rows_by_kind
    assert "license_decision" in rows_by_kind
    assert "citation" in rows_by_kind
    assert any(
        row["path"] == ".venv-tss"
        for row in rows_by_kind.get("local_only_dir_present", [])
    )
    assert export["schema"]["runtime_selector_changed"] is False
    assert (tmp_path / "public_release_checklist.md").exists()
    assert (tmp_path / "public_release_hygiene_report.md").exists()
