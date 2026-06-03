"""Minimal external-training package manifest for CSR Transformer policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION = (
    "phase1_csr_transformer_training_package_v1"
)
PACKAGE_KIND = "csr_transformer_external_training_package"
PACKAGE_ID = "csr_transformer_policy_v2_augmented_handoff_package"


def build_csr_transformer_training_package_from_files(
    *,
    handoff_dir: str | Path = "runs/phase1_csr_transformer_handoff_bundle",
    output_dir: str | Path = "runs/phase1_csr_transformer_training_package",
) -> dict[str, Any]:
    """Build a small manifest that external Transformer trainers can follow."""

    handoff = Path(handoff_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    handoff_summary_path = handoff / "csr_transformer_handoff_bundle_summary.json"
    handoff_rows_path = handoff / "csr_transformer_handoff_bundle_rows.jsonl"
    handoff_training_spec_path = handoff / "csr_transformer_handoff_training_spec.json"
    handoff_guard_contract_path = handoff / "csr_transformer_handoff_guard_contract.json"
    handoff_schema_path = handoff / "csr_transformer_handoff_bundle_schema.json"
    handoff_manifest_path = handoff / "artifact_manifest.json"

    handoff_summary = _read_json(handoff_summary_path)
    handoff_rows = tuple(read_jsonl(handoff_rows_path))
    training_spec = _read_json(handoff_training_spec_path)
    guard_contract = _read_json(handoff_guard_contract_path)
    source_by_label = {
        str(row["label"]): row
        for row in handoff_rows
        if row.get("row_kind") == "source_file"
    }
    source_by_path = {
        str(row["path"]): row
        for row in handoff_rows
        if row.get("row_kind") == "source_file"
    }
    input_files = _input_files(
        training_spec=training_spec,
        source_by_label=source_by_label,
        source_by_path=source_by_path,
        handoff_summary_path=handoff_summary_path,
        handoff_training_spec_path=handoff_training_spec_path,
        handoff_guard_contract_path=handoff_guard_contract_path,
        handoff_schema_path=handoff_schema_path,
        handoff_manifest_path=handoff_manifest_path,
    )
    expected_outputs = _expected_outputs(training_spec)
    validation_steps = _validation_steps(training_spec)
    validation_errors = _validation_errors(
        handoff_summary=handoff_summary,
        training_spec=training_spec,
        guard_contract=guard_contract,
        input_files=input_files,
        expected_outputs=expected_outputs,
        validation_steps=validation_steps,
    )
    summary = _summary(
        handoff_summary=handoff_summary,
        input_files=input_files,
        expected_outputs=expected_outputs,
        validation_steps=validation_steps,
        validation_errors=validation_errors,
    )
    package_manifest = _package_manifest(
        summary=summary,
        input_files=input_files,
        expected_outputs=expected_outputs,
        validation_steps=validation_steps,
        guard_contract=guard_contract,
    )
    rows = _rows(
        summary=summary,
        input_files=input_files,
        expected_outputs=expected_outputs,
        validation_steps=validation_steps,
    )
    schema = _schema(summary)
    readme = _readme(summary=summary, package_manifest=package_manifest)
    output_paths = {
        "package_manifest": output / "csr_transformer_training_package_manifest.json",
        "rows": output / "csr_transformer_training_package_rows.jsonl",
        "summary": output / "csr_transformer_training_package_summary.json",
        "schema": output / "csr_transformer_training_package_schema.json",
        "readme": output / "TRAINING_PACKAGE.md",
        "report": output / "csr_transformer_training_package_report.md",
    }
    _write_json(package_manifest, output_paths["package_manifest"])
    write_jsonl(rows, output_paths["rows"])
    _write_json(summary, output_paths["summary"])
    _write_json(schema, output_paths["schema"])
    output_paths["readme"].write_text(readme, encoding="utf-8")
    output_paths["report"].write_text(_report(summary), encoding="utf-8")
    return {
        "summary": summary,
        "schema": schema,
        "package_manifest": package_manifest,
        "rows": rows,
        "readme": readme,
        "paths": {key: str(value) for key, value in output_paths.items()},
    }


def _input_files(
    *,
    training_spec: dict[str, Any],
    source_by_label: dict[str, dict[str, Any]],
    source_by_path: dict[str, dict[str, Any]],
    handoff_summary_path: Path,
    handoff_training_spec_path: Path,
    handoff_guard_contract_path: Path,
    handoff_schema_path: Path,
    handoff_manifest_path: Path,
) -> tuple[dict[str, Any], ...]:
    paths = training_spec["inputs"]
    baseline = training_spec["baseline_context"]
    labels_and_paths = (
        ("tensor_arrays", paths["tensor_arrays"], "training_input"),
        ("request_index", paths["request_index"], "training_input"),
        ("label_free_model_requests", paths["label_free_model_requests"], "training_input"),
        ("offline_targets", paths["offline_targets"], "training_input"),
        ("selector_rows", paths["selector_rows"], "training_input"),
        ("handoff_summary", str(handoff_summary_path), "handoff_metadata"),
        ("training_spec", str(handoff_training_spec_path), "handoff_metadata"),
        ("guard_contract", str(handoff_guard_contract_path), "runtime_safety"),
        ("handoff_schema", str(handoff_schema_path), "handoff_metadata"),
        ("handoff_manifest", str(handoff_manifest_path), "handoff_metadata"),
        ("shadow_ranker_summary", baseline["shadow_ranker_summary"], "shadow_baseline"),
        ("shadow_ranker_predictions", baseline["shadow_ranker_predictions"], "shadow_baseline"),
        ("guarded_replay_summary", baseline["guarded_replay_summary"], "runtime_safety"),
        ("model_submission_contract", baseline["model_submission_contract"], "contribution_contract"),
        ("model_contribution_terms", "MODEL_CONTRIBUTION_TERMS.md", "contribution_contract"),
        ("contributor_license_agreement", "CONTRIBUTOR_LICENSE_AGREEMENT.md", "contribution_contract"),
    )
    return tuple(
        _descriptor(
            label=label,
            path=path,
            source_kind=source_kind,
            handoff_row=source_by_label.get(label) or source_by_path.get(path),
        )
        for label, path, source_kind in labels_and_paths
    )


def _descriptor(
    *,
    label: str,
    path: str,
    source_kind: str,
    handoff_row: dict[str, Any] | None,
) -> dict[str, Any]:
    file_path = Path(path)
    exists = file_path.exists()
    if handoff_row is not None:
        return {
            "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
            "label": label,
            "path": str(path),
            "source_kind": source_kind,
            "exists": bool(handoff_row["exists"]),
            "size_bytes": int(handoff_row["size_bytes"]),
            "sha256": handoff_row["sha256"],
            "checksum_source": "handoff_bundle",
        }
    data = file_path.read_bytes() if exists else b""
    return {
        "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
        "label": label,
        "path": str(path),
        "source_kind": source_kind,
        "exists": exists,
        "size_bytes": len(data),
        "sha256": _sha(data) if exists else None,
        "checksum_source": "direct",
    }


def _expected_outputs(training_spec: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
            "label": label,
            "path_template": path,
            "required": True,
        }
        for label, path in sorted(training_spec["expected_outputs"].items())
    )


def _validation_steps(training_spec: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    inputs = training_spec["inputs"]
    expected = training_spec["expected_outputs"]
    templates = (
        (
            "saved_model_replay",
            "scripts/tss_csr_transformer_model_replay.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_transformer_model_replay.py "
                f"--model {expected['saved_model']} "
                f"--tensors {inputs['tensor_arrays']} "
                f"--request-index {inputs['request_index']} "
                f"--reference-predictions {expected['predictions']} "
                "--out runs/future_csr_transformer_model_replay"
            ),
        ),
        (
            "quality_gate",
            "scripts/tss_csr_selector_model_eval.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_selector_model_eval.py "
                "--baseline-summary runs/phase1_csr_learning_readiness/csr_learning_summary.json "
                "--baseline-predictions runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl "
                f"--ranker-summary {expected['quality_gate_summary']} "
                f"--ranker-predictions {expected['predictions']} "
                "--out runs/future_csr_selector_model_eval"
            ),
        ),
        (
            "policy_model_artifact",
            "scripts/tss_csr_policy_model_artifact.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_policy_model_artifact.py "
                f"--model {expected['saved_model']} "
                f"--tensors {inputs['tensor_arrays']} "
                f"--request-index {inputs['request_index']} "
                f"--quality-gate-summary {expected['quality_gate_summary']} "
                f"--replay-summary {expected['replay_summary']} "
                "--out runs/future_csr_policy_model_artifact"
            ),
        ),
        (
            "policy_model_acceptance",
            "scripts/tss_csr_policy_model_acceptance.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_policy_model_acceptance.py "
                "--model-artifact future_csr_policy_model_artifact.json "
                "--out runs/future_csr_policy_model_acceptance"
            ),
        ),
        (
            "policy_model_submission",
            "scripts/tss_csr_policy_model_submission.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_policy_model_submission.py "
                "--model-artifact future_csr_policy_model_artifact.json "
                "--acceptance-summary runs/future_csr_policy_model_acceptance/csr_policy_model_acceptance_summary.json "
                "--acceptance-rows runs/future_csr_policy_model_acceptance/csr_policy_model_acceptance_rows.jsonl "
                "--contributor-name '<name>' --contribution-name '<model-name>' "
                "--out runs/future_csr_policy_model_submission"
            ),
        ),
        (
            "guarded_shadow_smoke",
            "scripts/tss_csr_learned_guard_smoke.py",
            (
                ".venv-tss/bin/python scripts/tss_csr_learned_guard_smoke.py "
                "--learned-model-artifact future_csr_policy_model_artifact.json "
                "--out runs/future_csr_learned_guard"
            ),
        ),
    )
    return tuple(
        {
            "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
            "step_index": index,
            "step_id": step_id,
            "script": script,
            "command_template": command,
            "required": True,
        }
        for index, (step_id, script, command) in enumerate(templates, start=1)
    )


def _validation_errors(
    *,
    handoff_summary: dict[str, Any],
    training_spec: dict[str, Any],
    guard_contract: dict[str, Any],
    input_files: tuple[dict[str, Any], ...],
    expected_outputs: tuple[dict[str, Any], ...],
    validation_steps: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    if handoff_summary.get("status") != "passed":
        errors.append("handoff_not_passed")
    if handoff_summary.get("handoff_ready") is not True:
        errors.append("handoff_not_ready")
    if handoff_summary.get("runtime_selector_changed") is not False:
        errors.append("handoff_changed_runtime_selector")
    if handoff_summary.get("model_trained") is not False:
        errors.append("handoff_trained_model")
    if training_spec.get("runtime_selector_changed") is not False:
        errors.append("training_spec_changed_runtime_selector")
    if training_spec.get("model_trained_by_this_artifact") is not False:
        errors.append("training_spec_trained_model")
    if guard_contract.get("guard_required") is not True:
        errors.append("guard_not_required")
    if guard_contract.get("runtime_selector_changed") is not False:
        errors.append("guard_contract_changed_runtime_selector")
    if any(row["exists"] is not True for row in input_files):
        errors.append("missing_training_package_input")
    if any(not row["sha256"] for row in input_files):
        errors.append("missing_training_package_checksum")
    if len({row["label"] for row in input_files}) != len(input_files):
        errors.append("duplicate_training_package_input_label")
    if len(expected_outputs) != 6:
        errors.append("expected_output_count_mismatch")
    if len(validation_steps) != 6:
        errors.append("validation_step_count_mismatch")
    forbidden = ("docs/toms_paper/", "MILESTONE_LOG", "SCOPE_PLATFORM", "ADVICE")
    if any(
        str(row["path"]).startswith(forbidden)
        for row in input_files
    ):
        errors.append("internal_development_doc_in_package")
    return tuple(errors)


def _summary(
    *,
    handoff_summary: dict[str, Any],
    input_files: tuple[dict[str, Any], ...],
    expected_outputs: tuple[dict[str, Any], ...],
    validation_steps: tuple[dict[str, Any], ...],
    validation_errors: tuple[str, ...],
) -> dict[str, Any]:
    package_ready = not validation_errors
    return {
        "status": "passed" if package_ready else "failed",
        "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
        "package_kind": PACKAGE_KIND,
        "package_id": PACKAGE_ID,
        "package_ready": package_ready,
        "source_handoff_ready": bool(handoff_summary.get("handoff_ready")),
        "model_training_required": True,
        "model_trained": False,
        "runtime_selector_changed": False,
        "executes_gpu": False,
        "copies_large_files": False,
        "input_reference_mode": "path_reference_with_sha256",
        "num_input_files": len(input_files),
        "input_total_size_bytes": sum(int(row["size_bytes"]) for row in input_files),
        "num_expected_outputs": len(expected_outputs),
        "num_validation_steps": len(validation_steps),
        "training_model_requests": int(handoff_summary["training_model_requests"]),
        "training_tensor_requests": int(handoff_summary["training_tensor_requests"]),
        "training_train_requests": int(handoff_summary["training_train_requests"]),
        "training_eval_requests": int(handoff_summary["training_eval_requests"]),
        "training_global_candidates": int(handoff_summary["training_global_candidates"]),
        "matrix_feature_dim": int(handoff_summary["matrix_feature_dim"]),
        "candidate_feature_dim": int(handoff_summary["candidate_feature_dim"]),
        "current_shadow_baseline_model_id": handoff_summary["ranker_model_id"],
        "current_shadow_baseline_runtime_eligible": False,
        "guard_actual_quality_gate_blocks": int(
            handoff_summary["guard_actual_quality_gate_blocks"]
        ),
        "guard_actual_runtime_selector_changes": int(
            handoff_summary["guard_actual_runtime_selector_changes"]
        ),
        "guard_fallback_chain_enforced_rows": int(
            handoff_summary["guard_fallback_chain_enforced_rows"]
        ),
        "terms_documents_required": tuple(handoff_summary["terms_documents_required"]),
        "internal_development_docs_included": False,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "next_step": (
            "train an external Transformer model using the package manifest, "
            "then run every validation step before any runtime consideration"
        ),
    }


def _package_manifest(
    *,
    summary: dict[str, Any],
    input_files: tuple[dict[str, Any], ...],
    expected_outputs: tuple[dict[str, Any], ...],
    validation_steps: tuple[dict[str, Any], ...],
    guard_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "package_kind": summary["package_kind"],
        "package_id": summary["package_id"],
        "package_ready": summary["package_ready"],
        "runtime_boundary": {
            "model_trained": False,
            "runtime_selector_changed": False,
            "executes_gpu": False,
            "guard_required": True,
            "default_mode": "shadow",
        },
        "input_reference_mode": summary["input_reference_mode"],
        "inputs": input_files,
        "expected_outputs": expected_outputs,
        "post_training_validation_steps": validation_steps,
        "guard_contract": {
            "min_confidence": guard_contract["min_confidence"],
            "promotion_requires": guard_contract["promotion_requires"],
            "counterfactual_rows_are_runtime_evidence": False,
        },
        "contribution_terms": {
            "required_documents": summary["terms_documents_required"],
            "commercial_use_requires_permission": True,
            "attribution": "Wei CUI",
        },
        "status": summary["status"],
        "validation_errors": summary["validation_errors"],
    }


def _rows(
    *,
    summary: dict[str, Any],
    input_files: tuple[dict[str, Any], ...],
    expected_outputs: tuple[dict[str, Any], ...],
    validation_steps: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "schema_version": summary["schema_version"],
            "row_kind": "package_summary",
            "package_ready": summary["package_ready"],
            "runtime_selector_changed": summary["runtime_selector_changed"],
            "model_trained": summary["model_trained"],
        },
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "input_file",
                **row,
            }
            for row in input_files
        ),
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "expected_output",
                **row,
            }
            for row in expected_outputs
        ),
        *(
            {
                "schema_version": summary["schema_version"],
                "row_kind": "validation_step",
                **row,
            }
            for row in validation_steps
        ),
    )


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CSR_TRANSFORMER_TRAINING_PACKAGE_SCHEMA_VERSION,
        "package_kind": PACKAGE_KIND,
        "package_ready": summary["package_ready"],
        "runtime_boundary": {
            "model_trained": False,
            "runtime_selector_changed": False,
            "executes_gpu": False,
            "guard_required": True,
        },
        "acceptance_gate": {
            "requires_handoff_ready": True,
            "requires_input_checksums": True,
            "requires_no_internal_development_docs": True,
            "requires_validation_pipeline": True,
            "requires_model_submission_terms": True,
        },
    }


def _readme(*, summary: dict[str, Any], package_manifest: dict[str, Any]) -> str:
    input_lines = [
        f"- `{row['label']}`: `{row['path']}`"
        for row in package_manifest["inputs"]
    ]
    step_lines = [
        f"{row['step_index']}. `{row['step_id']}`: `{row['command_template']}`"
        for row in package_manifest["post_training_validation_steps"]
    ]
    return "\n".join(
        [
            "# CSR Transformer Training Package",
            "",
            f"- package_id: `{summary['package_id']}`",
            f"- status: `{summary['status']}`",
            f"- package_ready: `{summary['package_ready']}`",
            f"- model_trained: `{summary['model_trained']}`",
            f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
            "",
            "## Inputs",
            "",
            *input_lines,
            "",
            "## Validation Steps",
            "",
            *step_lines,
            "",
            "## Boundary",
            "",
            "- This package does not train a model.",
            "- This package does not change runtime selection.",
            "- A trained model must pass every validation step before runtime use.",
            "",
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# CSR Transformer Training Package",
        "",
        f"- status: `{summary['status']}`",
        f"- package_ready: `{summary['package_ready']}`",
        f"- input_reference_mode: `{summary['input_reference_mode']}`",
        f"- input_files: `{summary['num_input_files']}`",
        f"- expected_outputs: `{summary['num_expected_outputs']}`",
        f"- validation_steps: `{summary['num_validation_steps']}`",
        f"- training_model_requests: `{summary['training_model_requests']}`",
        f"- training_global_candidates: `{summary['training_global_candidates']}`",
        f"- current_shadow_baseline_model_id: `{summary['current_shadow_baseline_model_id']}`",
        f"- guard_actual_runtime_selector_changes: `{summary['guard_actual_runtime_selector_changes']}`",
        f"- internal_development_docs_included: `{summary['internal_development_docs_included']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        f"- next_step: `{summary['next_step']}`",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
