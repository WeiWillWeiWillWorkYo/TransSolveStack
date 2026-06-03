"""Consumer-side dry run for external CSR Transformer training packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import write_jsonl


CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_SCHEMA_VERSION = (
    "phase1_csr_transformer_package_consumer_dry_run_v1"
)
DRY_RUN_KIND = "csr_transformer_package_consumer_dry_run"


REFERENCE_OUTPUT_MAP = {
    "saved_model": "adapted_csr_transformer_ranker_model.json",
    "predictions": "adapted_csr_transformer_ranker_predictions.jsonl",
    "replay_summary": "csr_external_model_replay_summary.json",
    "quality_gate_summary": "csr_external_model_quality_gate_summary.json",
    "policy_model_artifact": "csr_policy_model_artifact.json",
    "submission_manifest": "csr_policy_model_submission_manifest.json",
}


def build_csr_transformer_package_consumer_dry_run_from_files(
    *,
    package_dir: str | Path = "runs/phase1_csr_transformer_training_package",
    reference_intake_dir: str | Path = "runs/phase1_csr_external_model_intake",
    output_dir: str | Path = "runs/phase1_csr_transformer_package_consumer_dry_run",
) -> dict[str, Any]:
    """Validate that a D23 package can be consumed by the model intake chain.

    The dry run checks paths, checksums, expected output labels, and validation
    step routing. It intentionally does not train a model or modify runtime
    selector state.
    """

    package = Path(package_dir)
    reference = Path(reference_intake_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    package_manifest_path = package / "csr_transformer_training_package_manifest.json"
    package_summary_path = package / "csr_transformer_training_package_summary.json"
    package_rows_path = package / "csr_transformer_training_package_rows.jsonl"
    reference_summary_path = reference / "csr_external_model_intake_summary.json"
    reference_submission_summary_path = (
        reference / "csr_policy_model_submission_summary.json"
    )
    reference_acceptance_summary_path = (
        reference / "csr_policy_model_acceptance_summary.json"
    )

    package_manifest = _read_json(package_manifest_path)
    package_summary = _read_json(package_summary_path)
    reference_summary = _read_json(reference_summary_path)
    reference_submission_summary = _read_json(reference_submission_summary_path)
    reference_acceptance_summary = _read_json(reference_acceptance_summary_path)

    input_checks = _input_checks(package_manifest["inputs"])
    expected_output_checks = _expected_output_checks(
        package_manifest["expected_outputs"],
        reference_dir=reference,
    )
    validation_step_checks = _validation_step_checks(
        package_manifest["post_training_validation_steps"],
    )
    validation_errors = _validation_errors(
        package_manifest=package_manifest,
        package_summary=package_summary,
        reference_summary=reference_summary,
        reference_submission_summary=reference_submission_summary,
        reference_acceptance_summary=reference_acceptance_summary,
        input_checks=input_checks,
        expected_output_checks=expected_output_checks,
        validation_step_checks=validation_step_checks,
    )
    summary = _summary(
        package_summary=package_summary,
        reference_summary=reference_summary,
        reference_submission_summary=reference_submission_summary,
        reference_acceptance_summary=reference_acceptance_summary,
        input_checks=input_checks,
        expected_output_checks=expected_output_checks,
        validation_step_checks=validation_step_checks,
        validation_errors=validation_errors,
    )
    consumer_manifest = _consumer_manifest(
        summary=summary,
        package_manifest_path=package_manifest_path,
        package_summary_path=package_summary_path,
        package_rows_path=package_rows_path,
        reference_summary_path=reference_summary_path,
        reference_submission_summary_path=reference_submission_summary_path,
        reference_acceptance_summary_path=reference_acceptance_summary_path,
        expected_output_checks=expected_output_checks,
        validation_step_checks=validation_step_checks,
    )
    rows = _rows(
        summary=summary,
        input_checks=input_checks,
        expected_output_checks=expected_output_checks,
        validation_step_checks=validation_step_checks,
    )
    schema = _schema(summary)
    report = _report(summary)

    paths = {
        "consumer_manifest": output
        / "csr_transformer_package_consumer_dry_run_manifest.json",
        "rows": output / "csr_transformer_package_consumer_dry_run_rows.jsonl",
        "summary": output / "csr_transformer_package_consumer_dry_run_summary.json",
        "schema": output / "csr_transformer_package_consumer_dry_run_schema.json",
        "report": output / "csr_transformer_package_consumer_dry_run_report.md",
    }
    _write_json(consumer_manifest, paths["consumer_manifest"])
    write_jsonl(rows, paths["rows"])
    _write_json(summary, paths["summary"])
    _write_json(schema, paths["schema"])
    paths["report"].write_text(report, encoding="utf-8")
    return {
        "summary": summary,
        "schema": schema,
        "consumer_manifest": consumer_manifest,
        "rows": rows,
        "report": report,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _input_checks(inputs: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    forbidden = ("docs/toms_paper/", "MILESTONE_LOG", "SCOPE_PLATFORM", "ADVICE")
    for item in inputs:
        path = Path(str(item["path"]))
        exists = path.exists()
        actual_sha = _sha(path) if exists else None
        rows.append(
            {
                "schema_version": CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_SCHEMA_VERSION,
                "label": str(item["label"]),
                "path": str(path),
                "source_kind": str(item["source_kind"]),
                "exists": exists,
                "expected_sha256": item.get("sha256"),
                "actual_sha256": actual_sha,
                "checksum_match": exists and actual_sha == item.get("sha256"),
                "internal_development_doc": str(path).startswith(forbidden),
            }
        )
    return tuple(rows)


def _expected_output_checks(
    expected_outputs: list[dict[str, Any]],
    *,
    reference_dir: Path,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in expected_outputs:
        label = str(item["label"])
        reference_name = REFERENCE_OUTPUT_MAP.get(label)
        reference_path = reference_dir / reference_name if reference_name else None
        exists = reference_path.exists() if reference_path is not None else False
        rows.append(
            {
                "schema_version": CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_SCHEMA_VERSION,
                "label": label,
                "path_template": str(item["path_template"]),
                "reference_path": None if reference_path is None else str(reference_path),
                "exists_in_reference_fixture": exists,
                "sha256": _sha(reference_path) if exists and reference_path else None,
                "required": bool(item["required"]),
            }
        )
    return tuple(rows)


def _validation_step_checks(
    validation_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in validation_steps:
        script = Path(str(item["script"]))
        rows.append(
            {
                "schema_version": CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_SCHEMA_VERSION,
                "step_index": int(item["step_index"]),
                "step_id": str(item["step_id"]),
                "script": str(script),
                "script_exists": script.exists(),
                "command_template": str(item["command_template"]),
                "required": bool(item["required"]),
                "guarded_or_submission_step": _is_guarded_or_submission_step(
                    str(item["step_id"])
                ),
            }
        )
    return tuple(rows)


def _validation_errors(
    *,
    package_manifest: dict[str, Any],
    package_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    reference_submission_summary: dict[str, Any],
    reference_acceptance_summary: dict[str, Any],
    input_checks: tuple[dict[str, Any], ...],
    expected_output_checks: tuple[dict[str, Any], ...],
    validation_step_checks: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    if package_summary.get("status") != "passed":
        errors.append("source_package_not_passed")
    if package_summary.get("package_ready") is not True:
        errors.append("source_package_not_ready")
    if package_manifest.get("runtime_boundary", {}).get("guard_required") is not True:
        errors.append("source_package_guard_not_required")
    if package_manifest.get("runtime_boundary", {}).get("runtime_selector_changed") is not False:
        errors.append("source_package_changed_runtime_selector")
    if package_manifest.get("runtime_boundary", {}).get("model_trained") is not False:
        errors.append("source_package_trained_model")
    if package_manifest.get("runtime_boundary", {}).get("executes_gpu") is not False:
        errors.append("source_package_executes_gpu")
    if any(row["exists"] is not True for row in input_checks):
        errors.append("missing_package_input")
    if any(row["checksum_match"] is not True for row in input_checks):
        errors.append("package_input_checksum_mismatch")
    if any(row["internal_development_doc"] is True for row in input_checks):
        errors.append("package_references_internal_development_docs")
    if len(input_checks) != 16:
        errors.append("package_input_count_mismatch")
    if len(expected_output_checks) != 6:
        errors.append("expected_output_count_mismatch")
    if any(row["exists_in_reference_fixture"] is not True for row in expected_output_checks):
        errors.append("missing_reference_expected_output")
    if len(validation_step_checks) != 6:
        errors.append("validation_step_count_mismatch")
    if any(row["script_exists"] is not True for row in validation_step_checks):
        errors.append("missing_validation_script")
    if reference_summary.get("status") != "passed":
        errors.append("reference_intake_not_passed")
    if reference_summary.get("intake_ready") is not True:
        errors.append("reference_intake_not_ready")
    if reference_summary.get("shadow_submission_ready") is not True:
        errors.append("reference_shadow_submission_not_ready")
    if reference_summary.get("runtime_promotion_ready") is not False:
        errors.append("reference_runtime_promotion_unexpected")
    if reference_summary.get("runtime_selector_changed") is not False:
        errors.append("reference_changed_runtime_selector")
    if reference_summary.get("guarded_gpu_shadow_smoke_checked") is not True:
        errors.append("reference_missing_guarded_gpu_shadow_smoke")
    if reference_submission_summary.get("shadow_submission_ready") is not True:
        errors.append("submission_not_shadow_ready")
    if reference_submission_summary.get("runtime_promotion_ready") is not False:
        errors.append("submission_runtime_promotion_unexpected")
    if reference_acceptance_summary.get("accepted_for_shadow") is not True:
        errors.append("acceptance_not_shadow_ready")
    if reference_acceptance_summary.get("accepted_for_runtime_promotion") is not False:
        errors.append("acceptance_runtime_promotion_unexpected")
    return tuple(errors)


def _summary(
    *,
    package_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    reference_submission_summary: dict[str, Any],
    reference_acceptance_summary: dict[str, Any],
    input_checks: tuple[dict[str, Any], ...],
    expected_output_checks: tuple[dict[str, Any], ...],
    validation_step_checks: tuple[dict[str, Any], ...],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    consumer_ready = not validation_errors
    return {
        "status": "passed" if consumer_ready else "failed",
        "schema_version": CSR_TRANSFORMER_PACKAGE_CONSUMER_DRY_RUN_SCHEMA_VERSION,
        "dry_run_kind": DRY_RUN_KIND,
        "consumer_ready": consumer_ready,
        "dry_run_only": True,
        "model_trained": False,
        "executes_gpu": False,
        "runtime_selector_changed": False,
        "source_package_ready": bool(package_summary.get("package_ready")),
        "source_package_id": package_summary.get("package_id"),
        "input_checksum_verified": all(
            row["checksum_match"] is True for row in input_checks
        ),
        "num_inputs_verified": len(input_checks),
        "num_expected_outputs_resolved": sum(
            1 for row in expected_output_checks if row["exists_in_reference_fixture"]
        ),
        "num_validation_steps_resolved": sum(
            1 for row in validation_step_checks if row["script_exists"]
        ),
        "reference_intake_ready": bool(reference_summary.get("intake_ready")),
        "reference_shadow_submission_ready": bool(
            reference_summary.get("shadow_submission_ready")
        ),
        "reference_runtime_promotion_ready": bool(
            reference_summary.get("runtime_promotion_ready")
        ),
        "reference_guarded_gpu_shadow_smoke_checked": bool(
            reference_summary.get("guarded_gpu_shadow_smoke_checked")
        ),
        "reference_guarded_gpu_smoke_successes": reference_summary.get(
            "guarded_gpu_smoke_successes"
        ),
        "submission_shadow_ready": bool(
            reference_submission_summary.get("shadow_submission_ready")
        ),
        "submission_runtime_promotion_ready": bool(
            reference_submission_summary.get("runtime_promotion_ready")
        ),
        "acceptance_shadow_ready": bool(
            reference_acceptance_summary.get("accepted_for_shadow")
        ),
        "acceptance_runtime_promotion_ready": bool(
            reference_acceptance_summary.get("accepted_for_runtime_promotion")
        ),
        "promotion_blockers": tuple(reference_summary.get("promotion_blockers", ())),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "next_step": (
            "run this package against a real externally trained Transformer checkpoint, "
            "then require the existing quality gate and runtime guard before promotion"
        ),
    }


def _consumer_manifest(
    *,
    summary: dict[str, Any],
    package_manifest_path: Path,
    package_summary_path: Path,
    package_rows_path: Path,
    reference_summary_path: Path,
    reference_submission_summary_path: Path,
    reference_acceptance_summary_path: Path,
    expected_output_checks: tuple[dict[str, Any], ...],
    validation_step_checks: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "dry_run_kind": summary["dry_run_kind"],
        "consumer_ready": summary["consumer_ready"],
        "source_package": {
            "manifest": str(package_manifest_path),
            "summary": str(package_summary_path),
            "rows": str(package_rows_path),
        },
        "reference_consumer_fixture": {
            "intake_summary": str(reference_summary_path),
            "submission_summary": str(reference_submission_summary_path),
            "acceptance_summary": str(reference_acceptance_summary_path),
            "shadow_submission_ready": summary["reference_shadow_submission_ready"],
            "runtime_promotion_ready": summary["reference_runtime_promotion_ready"],
        },
        "resolved_expected_outputs": expected_output_checks,
        "validation_step_routes": validation_step_checks,
        "runtime_boundary": {
            "dry_run_only": True,
            "model_trained": False,
            "executes_gpu": False,
            "guard_required": True,
            "runtime_selector_changed": False,
            "default_mode_after_submission": "shadow",
        },
        "status": summary["status"],
        "validation_errors": summary["validation_errors"],
    }


def _rows(
    *,
    summary: dict[str, Any],
    input_checks: tuple[dict[str, Any], ...],
    expected_output_checks: tuple[dict[str, Any], ...],
    validation_step_checks: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "schema_version": summary["schema_version"],
            "row_kind": "consumer_decision",
            "status": summary["status"],
            "consumer_ready": summary["consumer_ready"],
            "shadow_submission_ready": summary["reference_shadow_submission_ready"],
            "runtime_promotion_ready": summary["reference_runtime_promotion_ready"],
            "runtime_selector_changed": summary["runtime_selector_changed"],
        },
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "package_input_check",
                **row,
            }
            for row in input_checks
        ),
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "expected_output_reference",
                **row,
            }
            for row in expected_output_checks
        ),
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "validation_step_route",
                **row,
            }
            for row in validation_step_checks
        ),
    )


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "dry_run_kind": DRY_RUN_KIND,
        "consumer_ready": summary["consumer_ready"],
        "checks": {
            "package_checksums": True,
            "expected_output_mapping": True,
            "validation_script_routes": True,
            "reference_submission_shadow_ready": True,
            "runtime_promotion_blocked_without_quality_gate": True,
        },
        "runtime_boundary": {
            "dry_run_only": True,
            "model_trained": False,
            "executes_gpu": False,
            "guard_required": True,
            "runtime_selector_changed": False,
        },
    }


def _report(summary: dict[str, Any]) -> str:
    blockers = [f"- `{item}`" for item in summary["promotion_blockers"]] or ["- none"]
    return "\n".join(
        [
            "# CSR Transformer Package Consumer Dry Run",
            "",
            f"- status: `{summary['status']}`",
            f"- consumer_ready: `{summary['consumer_ready']}`",
            f"- dry_run_only: `{summary['dry_run_only']}`",
            f"- source_package_ready: `{summary['source_package_ready']}`",
            f"- input_checksum_verified: `{summary['input_checksum_verified']}`",
            f"- inputs_verified: `{summary['num_inputs_verified']}`",
            f"- expected_outputs_resolved: `{summary['num_expected_outputs_resolved']}`",
            f"- validation_steps_resolved: `{summary['num_validation_steps_resolved']}`",
            f"- reference_shadow_submission_ready: `{summary['reference_shadow_submission_ready']}`",
            f"- reference_runtime_promotion_ready: `{summary['reference_runtime_promotion_ready']}`",
            f"- guarded_gpu_shadow_smoke_checked: `{summary['reference_guarded_gpu_shadow_smoke_checked']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            f"- validation_error_count: `{summary['validation_error_count']}`",
            "",
            "## Promotion Blockers",
            "",
            *blockers,
            "",
        ]
    )


def _is_guarded_or_submission_step(step_id: str) -> bool:
    return step_id in {
        "policy_model_acceptance",
        "policy_model_submission",
        "guarded_shadow_smoke",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
