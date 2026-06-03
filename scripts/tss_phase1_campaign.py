"""Build the Phase 1 readiness campaign report from existing artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.campaign import (
    build_phase1_campaign_report,
    write_campaign_artifacts,
)
from transsolvestack.profiling.provenance import (
    CORE_CAMPAIGN_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/phase1_campaign")
    args = parser.parse_args()

    report = build_phase1_campaign_report(args.runs_root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = write_campaign_artifacts(report, output)
    paths["manifest"] = output / "artifact_manifest.json"
    stage_manifest_files = tuple(
        str(Path(stage.artifact_dir) / "artifact_manifest.json")
        for stage in report.stages
        if (Path(stage.artifact_dir) / "artifact_manifest.json").exists()
    )
    tracked_files = tuple(
        dict.fromkeys((*CORE_CAMPAIGN_PROVENANCE_FILES, *stage_manifest_files))
    )
    write_manifest(
        build_artifact_manifest(
            artifact_kind="phase1_campaign_report",
            command="scripts/tss_phase1_campaign.py",
            tracked_files=tracked_files,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "campaign_id": report.campaign_id,
                "status": report.status,
                "num_stages": report.num_stages,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    if report.status != "passed":
        raise SystemExit("phase1 campaign status is not passed")


if __name__ == "__main__":
    main()
