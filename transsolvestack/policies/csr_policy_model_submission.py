"""Submission package contract for external CSR policy models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.policies.csr_policy_model_artifact import (
    csr_policy_model_artifact_paths,
    load_csr_policy_model_artifact,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_POLICY_MODEL_SUBMISSION_SCHEMA_VERSION = "phase1_csr_policy_model_submission_v1"
CSR_POLICY_MODEL_SUBMISSION_KIND = "csr_policy_model_submission"
REQUIRED_TERMS_DOCUMENTS = (
    "MODEL_CONTRIBUTION_TERMS.md",
    "CONTRIBUTOR_LICENSE_AGREEMENT.md",
)


def build_csr_policy_model_submission_from_files(
    model_artifact_path: str | Path = (
        "runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json"
    ),
    acceptance_summary_path: str | Path = (
        "runs/phase1_csr_policy_model_acceptance/"
        "csr_policy_model_acceptance_summary.json"
    ),
    acceptance_rows_path: str | Path = (
        "runs/phase1_csr_policy_model_acceptance/"
        "csr_policy_model_acceptance_rows.jsonl"
    ),
    *,
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    contribution_name: str = "csr_masked_self_attention_ranker_v1_reference_submission",
    contribution_version: str = "phase1-reference",
    training_statement: str = (
        "Reference CSR Transformer ranker trained on the current Phase 1 "
        "Transformer-ready SuiteSparse subset."
    ),
    contributor_terms_acknowledged: bool = True,
) -> dict[str, Any]:
    """Build a portable submission package around a policy model artifact.

    The package is intentionally a review/shadow boundary. It records the
    model files, checksums, contribution terms, acceptance-gate outcome, and
    runtime boundary without promoting the submitted model into production.
    """

    model_artifact_path = Path(model_artifact_path)
    acceptance_summary_path = Path(acceptance_summary_path)
    acceptance_rows_path = Path(acceptance_rows_path)
    artifact = load_csr_policy_model_artifact(model_artifact_path)
    acceptance_summary = _read_json(acceptance_summary_path)
    acceptance_rows = tuple(read_jsonl(acceptance_rows_path))
    source_files = _source_files(
        model_artifact_path=model_artifact_path,
        artifact=artifact,
        acceptance_summary_path=acceptance_summary_path,
        acceptance_rows_path=acceptance_rows_path,
    )
    file_checksums = tuple(
        _file_descriptor(label=label, path=Path(path)) for label, path in source_files
    )
    validation_errors = _validation_errors(
        artifact=artifact,
        acceptance_summary=acceptance_summary,
        acceptance_rows=acceptance_rows,
        file_checksums=file_checksums,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        contribution_name=contribution_name,
        contributor_terms_acknowledged=contributor_terms_acknowledged,
    )
    summary = _summary(
        artifact=artifact,
        acceptance_summary=acceptance_summary,
        file_checksums=file_checksums,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        contribution_name=contribution_name,
        contribution_version=contribution_version,
        contributor_terms_acknowledged=contributor_terms_acknowledged,
        validation_errors=validation_errors,
    )
    submission = _submission_manifest(
        summary=summary,
        artifact=artifact,
        model_artifact_path=model_artifact_path,
        acceptance_summary_path=acceptance_summary_path,
        acceptance_rows_path=acceptance_rows_path,
        file_checksums=file_checksums,
        training_statement=training_statement,
    )
    rows = _rows(
        summary=summary,
        submission=submission,
        acceptance_summary=acceptance_summary,
        file_checksums=file_checksums,
    )
    schema = _schema(summary)
    model_card = _model_card(
        summary=summary,
        submission=submission,
        training_statement=training_statement,
    )
    report = _report_text(summary)
    return {
        "submission": submission,
        "summary": summary,
        "schema": schema,
        "rows": rows,
        "model_card": model_card,
        "report": report,
    }


def write_csr_policy_model_submission_manifest(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["submission"], Path(path))


def write_csr_policy_model_submission_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    return write_jsonl(rows, path)


def write_csr_policy_model_submission_summary(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["summary"], Path(path))


def write_csr_policy_model_submission_schema(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    return _write_json(data["schema"], Path(path))


def write_csr_policy_model_submission_model_card(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(data["model_card"]), encoding="utf-8")
    return output


def write_csr_policy_model_submission_report(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(data["report"]), encoding="utf-8")
    return output


def _source_files(
    *,
    model_artifact_path: Path,
    artifact: dict[str, Any],
    acceptance_summary_path: Path,
    acceptance_rows_path: Path,
) -> tuple[tuple[str, str], ...]:
    paths = csr_policy_model_artifact_paths(artifact)
    return (
        ("policy_model_artifact", str(model_artifact_path)),
        ("saved_model", paths["model_path"]),
        ("transformer_tensors", paths["tensor_path"]),
        ("request_index", paths["request_index_path"]),
        ("quality_gate_summary", str(artifact["quality_gate"]["summary_path"])),
        ("replay_summary", str(artifact["replay"]["summary_path"])),
        ("acceptance_summary", str(acceptance_summary_path)),
        ("acceptance_rows", str(acceptance_rows_path)),
    )


def _file_descriptor(*, label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validation_errors(
    *,
    artifact: dict[str, Any],
    acceptance_summary: dict[str, Any],
    acceptance_rows: tuple[dict[str, Any], ...],
    file_checksums: tuple[dict[str, Any], ...],
    contributor_id: str,
    contributor_name: str,
    contribution_name: str,
    contributor_terms_acknowledged: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    if not contributor_id.strip():
        errors.append("missing_contributor_id")
    if not contributor_name.strip():
        errors.append("missing_contributor_name")
    if not contribution_name.strip():
        errors.append("missing_contribution_name")
    if contributor_terms_acknowledged is not True:
        errors.append("contributor_terms_not_acknowledged")
    if artifact["runtime_contract"]["guard_required"] is not True:
        errors.append("artifact_guard_not_required")
    if artifact["runtime_contract"]["runtime_selector_changed"] is not False:
        errors.append("artifact_changed_runtime_selector")
    if acceptance_summary.get("status") != "passed":
        errors.append("acceptance_not_passed")
    if acceptance_summary.get("accepted_for_shadow") is not True:
        errors.append("acceptance_shadow_not_ready")
    if acceptance_summary.get("guarded_gpu_shadow_smoke_checked") is not True:
        errors.append("acceptance_missing_guarded_gpu_shadow_smoke")
    if acceptance_summary.get("runtime_selector_changed") is not False:
        errors.append("acceptance_changed_runtime_selector")
    if not acceptance_rows:
        errors.append("empty_acceptance_rows")
    if any(item["exists"] is not True for item in file_checksums):
        errors.append("missing_submission_file")
    if len({item["label"] for item in file_checksums}) != len(file_checksums):
        errors.append("duplicate_submission_file_label")
    return tuple(errors)


def _summary(
    *,
    artifact: dict[str, Any],
    acceptance_summary: dict[str, Any],
    file_checksums: tuple[dict[str, Any], ...],
    contributor_id: str,
    contributor_name: str,
    contribution_name: str,
    contribution_version: str,
    contributor_terms_acknowledged: bool,
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    accepted_for_shadow = bool(acceptance_summary.get("accepted_for_shadow"))
    accepted_for_runtime_promotion = bool(
        acceptance_summary.get("accepted_for_runtime_promotion")
    )
    submission_ready = not validation_errors
    shadow_submission_ready = submission_ready and accepted_for_shadow
    runtime_promotion_ready = shadow_submission_ready and accepted_for_runtime_promotion
    default_runtime_mode = "shadow" if shadow_submission_ready else "rejected"
    return {
        "status": "passed" if submission_ready else "failed",
        "schema_version": CSR_POLICY_MODEL_SUBMISSION_SCHEMA_VERSION,
        "submission_kind": CSR_POLICY_MODEL_SUBMISSION_KIND,
        "submission_ready": submission_ready,
        "shadow_submission_ready": shadow_submission_ready,
        "runtime_promotion_ready": runtime_promotion_ready,
        "default_runtime_mode": default_runtime_mode,
        "contributor_id": contributor_id,
        "contributor_name": contributor_name,
        "contribution_name": contribution_name,
        "contribution_version": contribution_version,
        "terms_acknowledged": contributor_terms_acknowledged,
        "required_terms_documents": REQUIRED_TERMS_DOCUMENTS,
        "adapter": artifact["adapter"],
        "model_id": artifact["model"]["model_id"],
        "model_family": artifact["model"]["model_family"],
        "model_artifact_ready": True,
        "acceptance_status": acceptance_summary.get("status"),
        "accepted_for_shadow": accepted_for_shadow,
        "accepted_for_runtime_promotion": accepted_for_runtime_promotion,
        "promotion_blockers": tuple(acceptance_summary.get("promotion_blockers", ())),
        "guarded_gpu_shadow_smoke_checked": bool(
            acceptance_summary.get("guarded_gpu_shadow_smoke_checked")
        ),
        "guarded_gpu_smoke_successes": acceptance_summary.get(
            "guarded_gpu_smoke_successes"
        ),
        "num_packaged_files": len(file_checksums),
        "all_file_checksums_present": all(
            item["exists"] and item["sha256"] for item in file_checksums
        ),
        "runtime_selector_changed": False,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
    }


def _submission_manifest(
    *,
    summary: dict[str, Any],
    artifact: dict[str, Any],
    model_artifact_path: Path,
    acceptance_summary_path: Path,
    acceptance_rows_path: Path,
    file_checksums: tuple[dict[str, Any], ...],
    training_statement: str,
) -> dict[str, Any]:
    return {
        "schema_version": CSR_POLICY_MODEL_SUBMISSION_SCHEMA_VERSION,
        "submission_kind": CSR_POLICY_MODEL_SUBMISSION_KIND,
        "submission_id": (
            f"{summary['contributor_id']}:{summary['contribution_name']}:"
            f"{summary['contribution_version']}"
        ),
        "contributor": {
            "id": summary["contributor_id"],
            "name": summary["contributor_name"],
            "terms_acknowledged": summary["terms_acknowledged"],
            "required_terms_documents": summary["required_terms_documents"],
        },
        "model": {
            "model_id": summary["model_id"],
            "model_family": summary["model_family"],
            "adapter": summary["adapter"],
            "policy_model_artifact_path": str(model_artifact_path),
            "source_model_schema_version": artifact["model"][
                "source_model_schema_version"
            ],
            "training_statement": training_statement,
        },
        "acceptance": {
            "summary_path": str(acceptance_summary_path),
            "rows_path": str(acceptance_rows_path),
            "status": summary["acceptance_status"],
            "accepted_for_shadow": summary["accepted_for_shadow"],
            "accepted_for_runtime_promotion": summary[
                "accepted_for_runtime_promotion"
            ],
            "promotion_blockers": summary["promotion_blockers"],
            "guarded_gpu_shadow_smoke_checked": summary[
                "guarded_gpu_shadow_smoke_checked"
            ],
            "guarded_gpu_smoke_successes": summary["guarded_gpu_smoke_successes"],
        },
        "runtime_boundary": {
            "default_mode": summary["default_runtime_mode"],
            "guard_required": True,
            "promotion_requires_acceptance_gate": True,
            "promotion_requires_quality_gate": True,
            "runtime_selector_changed": False,
        },
        "files": file_checksums,
        "status": summary["status"],
        "validation_errors": summary["validation_errors"],
    }


def _rows(
    *,
    summary: dict[str, Any],
    submission: dict[str, Any],
    acceptance_summary: dict[str, Any],
    file_checksums: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    schema = summary["schema_version"]
    base_rows = (
        {
            "schema_version": schema,
            "row_kind": "submission_identity",
            "submission_id": submission["submission_id"],
            "contributor_id": summary["contributor_id"],
            "contribution_name": summary["contribution_name"],
            "contribution_version": summary["contribution_version"],
        },
        {
            "schema_version": schema,
            "row_kind": "terms",
            "terms_acknowledged": summary["terms_acknowledged"],
            "required_terms_documents": summary["required_terms_documents"],
        },
        {
            "schema_version": schema,
            "row_kind": "model_artifact",
            "model_id": summary["model_id"],
            "adapter": summary["adapter"],
            "model_artifact_ready": summary["model_artifact_ready"],
        },
        {
            "schema_version": schema,
            "row_kind": "acceptance_gate",
            "status": acceptance_summary.get("status"),
            "accepted_for_shadow": summary["accepted_for_shadow"],
            "accepted_for_runtime_promotion": summary[
                "accepted_for_runtime_promotion"
            ],
            "guarded_gpu_shadow_smoke_checked": summary[
                "guarded_gpu_shadow_smoke_checked"
            ],
        },
        {
            "schema_version": schema,
            "row_kind": "submission_decision",
            "status": summary["status"],
            "shadow_submission_ready": summary["shadow_submission_ready"],
            "runtime_promotion_ready": summary["runtime_promotion_ready"],
            "runtime_selector_changed": summary["runtime_selector_changed"],
            "validation_errors": summary["validation_errors"],
        },
    )
    checksum_rows = tuple(
        {
            "schema_version": schema,
            "row_kind": "file_checksum",
            **item,
        }
        for item in file_checksums
    )
    return (*base_rows, *checksum_rows)


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_POLICY_MODEL_SUBMISSION_SCHEMA_VERSION,
        "submission_kind": CSR_POLICY_MODEL_SUBMISSION_KIND,
        "required_terms_documents": REQUIRED_TERMS_DOCUMENTS,
        "required_file_labels": (
            "policy_model_artifact",
            "saved_model",
            "transformer_tensors",
            "request_index",
            "quality_gate_summary",
            "replay_summary",
            "acceptance_summary",
            "acceptance_rows",
        ),
        "runtime_boundary": {
            "default_mode": summary["default_runtime_mode"],
            "guard_required": True,
            "shadow_submission_ready": summary["shadow_submission_ready"],
            "runtime_promotion_ready": summary["runtime_promotion_ready"],
            "runtime_selector_changed": False,
        },
    }


def _model_card(
    *,
    summary: dict[str, Any],
    submission: dict[str, Any],
    training_statement: str,
) -> str:
    file_lines = [
        f"- `{item['label']}`: `{item['path']}` sha256 `{item['sha256']}`"
        for item in submission["files"]
    ]
    blocker_lines = [
        f"- `{blocker}`" for blocker in summary["promotion_blockers"]
    ] or ["- none"]
    return "\n".join(
        [
            "# CSR Policy Model Submission",
            "",
            "## Identity",
            "",
            f"- submission_id: `{submission['submission_id']}`",
            f"- contributor: `{summary['contributor_name']}`",
            f"- contribution_name: `{summary['contribution_name']}`",
            f"- contribution_version: `{summary['contribution_version']}`",
            "",
            "## Model",
            "",
            f"- model_id: `{summary['model_id']}`",
            f"- model_family: `{summary['model_family']}`",
            f"- adapter: `{summary['adapter']}`",
            f"- training_statement: {training_statement}",
            "",
            "## Acceptance",
            "",
            f"- accepted_for_shadow: `{summary['accepted_for_shadow']}`",
            f"- accepted_for_runtime_promotion: `{summary['accepted_for_runtime_promotion']}`",
            f"- guarded_gpu_shadow_smoke_checked: `{summary['guarded_gpu_shadow_smoke_checked']}`",
            f"- guarded_gpu_smoke_successes: `{summary['guarded_gpu_smoke_successes']}`",
            "",
            "## Promotion Blockers",
            "",
            *blocker_lines,
            "",
            "## Runtime Boundary",
            "",
            f"- default_runtime_mode: `{summary['default_runtime_mode']}`",
            "- guard_required: `True`",
            "- promotion_requires_acceptance_gate: `True`",
            "- promotion_requires_quality_gate: `True`",
            "- runtime_selector_changed: `False`",
            "",
            "## Files",
            "",
            *file_lines,
            "",
            "## Terms",
            "",
            f"- terms_acknowledged: `{summary['terms_acknowledged']}`",
            "- required_terms_documents: `MODEL_CONTRIBUTION_TERMS.md`, `CONTRIBUTOR_LICENSE_AGREEMENT.md`",
            "",
        ]
    )


def _report_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CSR Policy Model Submission",
            "",
            f"- status: `{summary['status']}`",
            f"- schema_version: `{summary['schema_version']}`",
            f"- submission_ready: `{summary['submission_ready']}`",
            f"- shadow_submission_ready: `{summary['shadow_submission_ready']}`",
            f"- runtime_promotion_ready: `{summary['runtime_promotion_ready']}`",
            f"- default_runtime_mode: `{summary['default_runtime_mode']}`",
            f"- model_id: `{summary['model_id']}`",
            f"- adapter: `{summary['adapter']}`",
            f"- num_packaged_files: `{summary['num_packaged_files']}`",
            f"- all_file_checksums_present: `{summary['all_file_checksums_present']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            f"- validation_error_count: `{summary['validation_error_count']}`",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
