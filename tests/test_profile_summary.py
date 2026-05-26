import json

from transsolvestack.benchmarks.oracle import (
    build_oracles_for_contexts,
    write_oracle_plans,
)
from transsolvestack.benchmarks.summary import (
    summarize_candidate_performance,
    write_performance_summary_report,
)
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    read_candidate_performance,
    write_candidate_performance,
)


def test_read_candidate_performance_roundtrip(tmp_path):
    records = (
        CandidatePerformanceRecord(
            run_id="r",
            candidate_id="c",
            context_id="ctx",
            backend="taichi_gpu",
            status="dry_run",
        ),
    )
    path = write_candidate_performance(records, tmp_path / "perf.jsonl")
    loaded = read_candidate_performance(path)
    assert loaded == records


def test_build_oracles_for_contexts_selects_each_context_best():
    records = (
        CandidatePerformanceRecord(
            run_id="r1",
            system_id="sys1",
            candidate_id="slow",
            context_id="ctx1",
            backend="taichi_gpu",
            status="success",
            total_time_ms=10.0,
        ),
        CandidatePerformanceRecord(
            run_id="r2",
            system_id="sys1",
            candidate_id="fast",
            context_id="ctx1",
            backend="taichi_gpu",
            status="success",
            total_time_ms=3.0,
        ),
        CandidatePerformanceRecord(
            run_id="r3",
            system_id="sys2",
            candidate_id="only",
            context_id="ctx2",
            backend="taichi_gpu",
            status="failed",
        ),
    )
    oracles = build_oracles_for_contexts(records)
    assert len(oracles) == 2
    assert oracles[0].system_id == "sys1"
    assert oracles[0].context_id == "ctx1"
    assert oracles[0].best_candidate_id == "fast"
    assert oracles[1].system_id == "sys2"
    assert oracles[1].context_id == "ctx2"
    assert oracles[1].best_candidate_id is None


def test_write_oracle_plans(tmp_path):
    records = (
        CandidatePerformanceRecord(
            run_id="r",
            candidate_id="c",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=1.0,
        ),
    )
    oracles = build_oracles_for_contexts(records)
    path = write_oracle_plans(oracles, tmp_path / "oracle.jsonl")
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["best_candidate_id"] == "c"


def test_candidate_performance_summary_counts_dry_run():
    summary = summarize_candidate_performance(
        [
            CandidatePerformanceRecord(
                run_id="r1",
                candidate_id="c1",
                context_id="ctx",
                backend="taichi_gpu",
                status="dry_run",
            ),
            CandidatePerformanceRecord(
                run_id="r2",
                candidate_id="c2",
                context_id="ctx",
                backend="taichi_gpu",
                status="failed",
            ),
        ]
    )
    assert summary.num_records == 2
    assert summary.system_counts == {"unknown": 2}
    assert summary.num_dry_run == 1
    assert summary.num_failed == 1
    assert summary.best_candidate_id is None


def test_performance_summary_report_writer(tmp_path):
    records = (
        CandidatePerformanceRecord(
            run_id="r",
            candidate_id="c",
            context_id="ctx",
            backend="taichi_gpu",
            status="dry_run",
        ),
    )
    summary = summarize_candidate_performance(records)
    oracles = build_oracles_for_contexts(records)
    path = write_performance_summary_report(
        summary,
        oracles,
        tmp_path / "summary.md",
    )
    text = path.read_text(encoding="utf-8")
    assert "Candidate Performance Summary" in text
    assert "num_dry_run" in text
