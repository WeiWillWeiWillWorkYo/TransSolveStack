"""Public-release hygiene checks for the repository workspace."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import write_jsonl


PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION = "phase1_public_release_hygiene_v1"

REQUIRED_GITIGNORE_PATTERNS = (
    ".venv*/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "/datasets/",
    "/external/",
    "/downloads/",
    "/data/raw/",
    "/data/external/",
    "/data/suitesparse/",
    "runs/**/*.npy",
    "runs/**/*.npz",
    "runs/**/*.pkl",
    "runs/**/*.bin",
    "runs/**/*.mp4",
    "runs/**/*.zip",
    "runs/**/csr_matrices.jsonl",
    "ADVICE.md",
    "MILESTONE_LOG.md",
    "FOOTAGE_SPEC.md",
    "PUBLIC_RELEASE.md",
    "GITHUB_RELEASE_GATE.md",
    "PHASE1_ROADMAP.md",
    "advice.md",
    "scope.md",
    "footage/",
    "docs/toms_paper/",
    "reports/",
)

PUBLIC_SOURCE_PATHS = (
    "transsolvestack",
    "scripts",
    "tests",
    "configs",
    "runs",
    "LICENSE",
    "README.md",
    "docs/PROJECT_OVERVIEW.md",
    "LICENSE_POLICY.md",
    "COMMERCIAL_USE.md",
    "MODEL_CONTRIBUTION_TERMS.md",
    "CONTRIBUTOR_LICENSE_AGREEMENT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    ".gitignore",
    "pyproject.toml",
)

REQUIRED_RELEASE_DOCS = (
    "README.md",
    "docs/PROJECT_OVERVIEW.md",
    "LICENSE",
    "LICENSE_POLICY.md",
    "COMMERCIAL_USE.md",
    "MODEL_CONTRIBUTION_TERMS.md",
    "CONTRIBUTOR_LICENSE_AGREEMENT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
)

LOCAL_ONLY_DIRS = (
    ".venv-tss",
    ".pytest_cache",
    "__pycache__",
    ".git",
)

SECRET_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_rsa.pub",
)

HEAVY_FILE_PATTERNS = (
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.pickle",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "runs/**/csr_matrices.jsonl",
)


def build_public_release_hygiene_artifact(
    root: str | Path = ".",
    output_dir: str | Path = "runs/phase1_public_release_hygiene",
    *,
    max_public_file_size_bytes: int = 10_000_000,
) -> dict[str, Any]:
    """Scan public-release hygiene without requiring a git repository."""

    repo = Path(root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=True)

    gitignore_path = repo / ".gitignore"
    gitignore_lines = _gitignore_lines(gitignore_path)
    missing_patterns = tuple(
        pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in gitignore_lines
    )
    scan = _scan_workspace(
        repo,
        max_public_file_size_bytes=max_public_file_size_bytes,
        output_dir=output,
    )
    source_status = _source_path_status(repo)
    release_doc_status = _release_doc_status(repo)
    git_initialized = (repo / ".git").exists()
    validation_errors = []
    if not gitignore_path.exists():
        validation_errors.append("missing_gitignore")
    validation_errors.extend(f"missing_gitignore_pattern:{item}" for item in missing_patterns)
    validation_errors.extend(
        f"large_unignored_file:{item['path']}" for item in scan["large_files"]
    )
    validation_errors.extend(
        f"secret_like_file:{item['path']}" for item in scan["secret_like_files"]
    )
    validation_errors.extend(
        f"missing_public_source_path:{item['path']}"
        for item in source_status
        if item["exists"] is not True
    )
    validation_errors.extend(
        f"missing_release_doc:{item['path']}"
        for item in release_doc_status
        if item["exists"] is not True
    )

    hygiene_ready = not validation_errors
    rows = _rows(
        gitignore_lines=gitignore_lines,
        missing_patterns=missing_patterns,
        scan=scan,
        source_status=source_status,
        release_doc_status=release_doc_status,
        git_initialized=git_initialized,
    )
    release_decision_blockers = () if git_initialized else ("git_init_pending",)
    github_upload_ready = hygiene_ready and git_initialized and not release_decision_blockers
    summary = {
        "status": "passed" if hygiene_ready else "failed",
        "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
        "public_release_hygiene_ready": hygiene_ready,
        "github_upload_ready": github_upload_ready,
        "project_owner": "Wei CUI",
        "citation_author": "Wei CUI",
        "license_route_selected": True,
        "formal_license_finalized": True,
        "source_code_license": "PolyForm Noncommercial License 1.0.0",
        "documentation_license_direction": "CC BY-NC 4.0",
        "model_terms_documented": all(
            item["exists"]
            for item in release_doc_status
            if item["path"] == "MODEL_CONTRIBUTION_TERMS.md"
        ),
        "contributor_terms_documented": all(
            item["exists"]
            for item in release_doc_status
            if item["path"] == "CONTRIBUTOR_LICENSE_AGREEMENT.md"
        ),
        "commercial_use_requires_permission": True,
        "license_policy_documented": all(
            item["exists"] for item in release_doc_status if item["path"] == "LICENSE_POLICY.md"
        ),
        "permissive_license_family_rejected": True,
        "license_upload_blockers": release_decision_blockers,
        "git_repository_initialized": git_initialized,
        "git_init_required": not git_initialized,
        "runtime_selector_changed": False,
        "max_public_file_size_bytes": max_public_file_size_bytes,
        "required_gitignore_patterns": REQUIRED_GITIGNORE_PATTERNS,
        "missing_gitignore_patterns": missing_patterns,
        "public_source_paths": PUBLIC_SOURCE_PATHS,
        "public_source_path_count": len(PUBLIC_SOURCE_PATHS),
        "required_release_docs": REQUIRED_RELEASE_DOCS,
        "required_release_doc_count": len(REQUIRED_RELEASE_DOCS),
        "workspace_file_count_scanned": scan["file_count"],
        "workspace_bytes_scanned": scan["total_bytes"],
        "large_unignored_file_count": len(scan["large_files"]),
        "secret_like_file_count": len(scan["secret_like_files"]),
        "local_only_dir_count": len(scan["local_only_dirs"]),
        "ignored_heavy_file_count": len(scan["ignored_heavy_files"]),
        "generated_artifact_policy": (
            "commit small JSON/Markdown runs artifacts for tests; exclude binary/heavy payloads"
        ),
        "external_dataset_policy": (
            "keep full SuiteSparse and large matrix collections under /mnt/tss_external"
        ),
        "validation_error_count": len(validation_errors),
        "validation_errors": tuple(validation_errors),
        "next_step": (
            "configure_github_remote_and_push"
            if github_upload_ready
            else "git_init_review_status_then_commit_public_source_snapshot"
        ),
    }
    schema = _schema(summary)
    paths = {
        "rows": output / "public_release_hygiene_rows.jsonl",
        "summary": output / "public_release_hygiene_summary.json",
        "schema": output / "public_release_hygiene_schema.json",
        "checklist": output / "public_release_checklist.md",
        "report": output / "public_release_hygiene_report.md",
    }
    write_jsonl(rows, paths["rows"])
    _write_json(summary, paths["summary"])
    _write_json(schema, paths["schema"])
    _write_checklist(summary, paths["checklist"])
    _write_report(summary, rows, paths["report"])
    return {
        "rows": rows,
        "summary": summary,
        "schema": schema,
        "paths": paths,
    }


def _gitignore_lines(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return tuple(lines)


def _scan_workspace(
    repo: Path,
    *,
    max_public_file_size_bytes: int,
    output_dir: Path,
) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    large_files: list[dict[str, Any]] = []
    secret_like_files: list[dict[str, Any]] = []
    ignored_heavy_files: list[dict[str, Any]] = []
    local_only_dirs: set[str] = set()
    output_rel = _relpath(output_dir, repo)
    for current, dirnames, filenames in os.walk(repo):
        current_path = Path(current)
        rel_current = _relpath(current_path, repo)
        kept_dirs = []
        for dirname in dirnames:
            rel_dir = _join_rel(rel_current, dirname)
            if dirname in LOCAL_ONLY_DIRS or rel_dir.startswith(".venv"):
                local_only_dirs.add(rel_dir)
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            path = current_path / filename
            rel = _relpath(path, repo)
            if rel.startswith(output_rel + "/"):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            file_count += 1
            total_bytes += size
            if _matches_any(filename, SECRET_NAME_PATTERNS) or _matches_any(rel, SECRET_NAME_PATTERNS):
                secret_like_files.append({"path": rel, "size_bytes": size})
            if _matches_any(filename, HEAVY_FILE_PATTERNS) or _matches_any(rel, HEAVY_FILE_PATTERNS):
                ignored_heavy_files.append({"path": rel, "size_bytes": size})
                continue
            if size > max_public_file_size_bytes:
                large_files.append({"path": rel, "size_bytes": size})
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "large_files": tuple(large_files),
        "secret_like_files": tuple(secret_like_files),
        "ignored_heavy_files": tuple(ignored_heavy_files),
        "local_only_dirs": tuple(sorted(local_only_dirs)),
    }


def _source_path_status(repo: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for rel in PUBLIC_SOURCE_PATHS:
        path = repo / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
            }
        )
    return tuple(rows)


def _release_doc_status(repo: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for rel in REQUIRED_RELEASE_DOCS:
        path = repo / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "kind": "file",
            }
        )
    return tuple(rows)


def _rows(
    *,
    gitignore_lines: tuple[str, ...],
    missing_patterns: tuple[str, ...],
    scan: dict[str, Any],
    source_status: tuple[dict[str, Any], ...],
    release_doc_status: tuple[dict[str, Any], ...],
    git_initialized: bool,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = [
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "git_repository",
            "path": ".git",
            "status": "initialized" if git_initialized else "not_initialized",
            "blocks_hygiene": False,
            "detail": (
                "git repository is initialized"
                if git_initialized
                else "git init is still required before upload"
            ),
        }
    ]
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "gitignore_required_pattern",
            "path": pattern,
            "status": "missing" if pattern in missing_patterns else "present",
            "blocks_hygiene": pattern in missing_patterns,
            "detail": "required public-release ignore rule",
        }
        for pattern in REQUIRED_GITIGNORE_PATTERNS
    )
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "public_source_path",
            "path": item["path"],
            "status": "present" if item["exists"] else "missing",
            "blocks_hygiene": item["exists"] is not True,
            "detail": item["kind"],
        }
        for item in source_status
    )
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "release_required_document",
            "path": item["path"],
            "status": "present" if item["exists"] else "missing",
            "blocks_hygiene": item["exists"] is not True,
            "detail": item["kind"],
        }
        for item in release_doc_status
    )
    rows.extend(
        (
            {
                "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
                "row_kind": "license_decision",
                "path": "LICENSE",
                "status": "polyform_noncommercial_selected",
                "blocks_hygiene": False,
                "detail": "commercial use requires separate permission from Wei CUI",
            },
            {
                "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
                "row_kind": "license_decision",
                "path": "LICENSE_POLICY.md",
                "status": "permissive_family_rejected",
                "blocks_hygiene": False,
                "detail": "Apache-2.0/MIT/BSD are not the intended route",
            },
            {
                "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
                "row_kind": "license_decision",
                "path": "MODEL_CONTRIBUTION_TERMS.md",
                "status": "noncommercial_model_terms_documented",
                "blocks_hygiene": False,
                "detail": "model cards, provenance, and quality gates required",
            },
            {
                "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
                "row_kind": "license_decision",
                "path": "CONTRIBUTOR_LICENSE_AGREEMENT.md",
                "status": "cla_light_documented",
                "blocks_hygiene": False,
                "detail": "contributions grant project reuse and relicensing rights",
            },
            {
                "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
                "row_kind": "citation",
                "path": "CITATION.cff",
                "status": "present",
                "blocks_hygiene": False,
                "detail": "author: Wei CUI",
            },
        )
    )
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "local_only_dir_present",
            "path": path,
            "status": "ignored_by_policy",
            "blocks_hygiene": False,
            "detail": "must not be committed",
        }
        for path in scan["local_only_dirs"]
    )
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "large_unignored_file",
            "path": item["path"],
            "status": "blocked",
            "blocks_hygiene": True,
            "detail": f"{item['size_bytes']} bytes",
        }
        for item in scan["large_files"]
    )
    rows.extend(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "secret_like_file",
            "path": item["path"],
            "status": "blocked",
            "blocks_hygiene": True,
            "detail": f"{item['size_bytes']} bytes",
        }
        for item in scan["secret_like_files"]
    )
    rows.append(
        {
            "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
            "row_kind": "gitignore_line_count",
            "path": ".gitignore",
            "status": "present",
            "blocks_hygiene": False,
            "detail": str(len(gitignore_lines)),
        }
    )
    return tuple(rows)


def _schema(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PUBLIC_RELEASE_HYGIENE_SCHEMA_VERSION,
        "task": "public_repository_hygiene_gate",
        "public_release_hygiene_ready": summary["public_release_hygiene_ready"],
        "runtime_selector_changed": False,
        "checks": {
            "gitignore_required": True,
            "large_file_scan": True,
            "secret_name_scan": True,
            "external_dataset_policy": True,
            "release_docs_required": True,
            "license_policy_required": True,
            "git_repository_required_for_upload": True,
        },
        "license_policy": {
            "project_owner": summary["project_owner"],
            "license_route_selected": summary["license_route_selected"],
            "formal_license_finalized": summary["formal_license_finalized"],
            "source_code_license": summary["source_code_license"],
            "commercial_use_requires_permission": summary[
                "commercial_use_requires_permission"
            ],
            "permissive_license_family_rejected": summary[
                "permissive_license_family_rejected"
            ],
            "github_upload_ready": summary["github_upload_ready"],
        },
        "non_goals": (
            "does_not_modify_git_state",
            "does_not_delete_local_generated_files",
            "does_not_upload_to_github",
        ),
    }


def _write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_checklist(summary: dict[str, Any], path: Path) -> Path:
    lines = [
        "# Public Release Checklist",
        "",
        f"- hygiene_ready: `{summary['public_release_hygiene_ready']}`",
        f"- github_upload_ready: `{summary['github_upload_ready']}`",
        f"- project_owner: `{summary['project_owner']}`",
        f"- license_route_selected: `{summary['license_route_selected']}`",
        f"- formal_license_finalized: `{summary['formal_license_finalized']}`",
        f"- source_code_license: `{summary['source_code_license']}`",
        f"- commercial_use_requires_permission: `{summary['commercial_use_requires_permission']}`",
        f"- permissive_license_family_rejected: `{summary['permissive_license_family_rejected']}`",
        f"- git_repository_initialized: `{summary['git_repository_initialized']}`",
        f"- git_init_required: `{summary['git_init_required']}`",
        f"- runtime_selector_changed: `{summary['runtime_selector_changed']}`",
        f"- large_unignored_file_count: `{summary['large_unignored_file_count']}`",
        f"- secret_like_file_count: `{summary['secret_like_file_count']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        "",
        "Before public upload:",
        "",
        "1. Final-review the selected license and contribution documents.",
        "2. Run the artifact verifier and full test suite.",
        "3. Initialize git only after reviewing ignored and generated files.",
        "4. Commit source, tests, professional docs, configs, and small JSON/Markdown runs artifacts.",
        "5. Keep virtual environments, caches, internal development docs, paper-prep notes, footage, full SuiteSparse data, raw matrix archives, and heavy binary artifacts out of git.",
        "6. Push to a fresh GitHub repository after `git status` matches the release manifest.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_report(summary: dict[str, Any], rows: tuple[dict[str, Any], ...], path: Path) -> Path:
    lines = [
        "# Public Release Hygiene",
        "",
        f"- status: `{summary['status']}`",
        f"- schema_version: `{summary['schema_version']}`",
        f"- public_release_hygiene_ready: `{summary['public_release_hygiene_ready']}`",
        f"- github_upload_ready: `{summary['github_upload_ready']}`",
        f"- project_owner: `{summary['project_owner']}`",
        f"- license_route_selected: `{summary['license_route_selected']}`",
        f"- formal_license_finalized: `{summary['formal_license_finalized']}`",
        f"- source_code_license: `{summary['source_code_license']}`",
        f"- commercial_use_requires_permission: `{summary['commercial_use_requires_permission']}`",
        f"- permissive_license_family_rejected: `{summary['permissive_license_family_rejected']}`",
        f"- git_repository_initialized: `{summary['git_repository_initialized']}`",
        f"- git_init_required: `{summary['git_init_required']}`",
        f"- workspace_file_count_scanned: `{summary['workspace_file_count_scanned']}`",
        f"- workspace_bytes_scanned: `{summary['workspace_bytes_scanned']}`",
        f"- local_only_dir_count: `{summary['local_only_dir_count']}`",
        f"- ignored_heavy_file_count: `{summary['ignored_heavy_file_count']}`",
        f"- large_unignored_file_count: `{summary['large_unignored_file_count']}`",
        f"- secret_like_file_count: `{summary['secret_like_file_count']}`",
        f"- validation_error_count: `{summary['validation_error_count']}`",
        "",
        "| row_kind | status | path | detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['row_kind']} | {row['status']} | {row['path']} | {row['detail']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _join_rel(prefix: str, name: str) -> str:
    return name if prefix == "." else f"{prefix}/{name}"
