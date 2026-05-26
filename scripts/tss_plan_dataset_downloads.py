"""Build a dataset download dry-run plan without fetching matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.datasets.catalog import (
    build_download_plan,
    load_matrix_catalog,
    write_download_plan,
    write_download_plan_report,
)
from transsolvestack.profiling.provenance import (
    CORE_DATASET_PLAN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.runtime.config import load_resource_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="configs/datasets/phase1_external_small.yaml")
    parser.add_argument("--out", default="runs/phase1_dataset_plan")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    catalog = load_matrix_catalog(args.catalog)
    if catalog.resource_limits_path is None:
        raise SystemExit("catalog does not define resource_limits")
    budget = load_resource_budget(catalog.resource_limits_path)
    plan = build_download_plan(catalog, budget, data_root=args.data_root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "download_plan": output / "download_plan.jsonl",
        "report": output / "download_plan.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_download_plan(plan, paths["download_plan"])
    write_download_plan_report(plan, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="dataset_download_plan",
            command="scripts/tss_plan_dataset_downloads.py",
            tracked_files=CORE_DATASET_PLAN_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "catalog_id": catalog.catalog_id,
                "num_matrices": len(plan.entries),
                "total_estimated_download_gb": plan.total_estimated_download_gb,
                "dry_run": True,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
