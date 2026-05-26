"""Build the public-release hygiene artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.provenance import (
    CORE_PUBLIC_RELEASE_HYGIENE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.public_release_hygiene import (
    build_public_release_hygiene_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="runs/phase1_public_release_hygiene")
    parser.add_argument("--max-file-size-bytes", type=int, default=10_000_000)
    args = parser.parse_args()

    export = build_public_release_hygiene_artifact(
        args.root,
        args.out,
        max_public_file_size_bytes=args.max_file_size_bytes,
    )
    summary = export["summary"]
    paths = dict(export["paths"])
    manifest_path = Path(args.out) / "artifact_manifest.json"
    paths["manifest"] = manifest_path
    write_manifest(
        build_artifact_manifest(
            artifact_kind="public_release_hygiene",
            command="scripts/tss_public_release_hygiene.py",
            tracked_files=CORE_PUBLIC_RELEASE_HYGIENE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": summary["status"],
                "schema_version": summary["schema_version"],
                "public_release_hygiene_ready": summary["public_release_hygiene_ready"],
                "git_repository_initialized": summary["git_repository_initialized"],
                "git_init_required": summary["git_init_required"],
                "runtime_selector_changed": summary["runtime_selector_changed"],
                "large_unignored_file_count": summary["large_unignored_file_count"],
                "secret_like_file_count": summary["secret_like_file_count"],
                "validation_error_count": summary["validation_error_count"],
            },
        ),
        manifest_path,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {summary['status']}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
