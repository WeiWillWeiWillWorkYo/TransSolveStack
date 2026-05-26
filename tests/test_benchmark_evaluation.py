from transsolvestack.benchmarks.evaluation import (
    build_candidate_evaluations,
    write_benchmark_report,
    write_candidate_evaluations,
)
from transsolvestack.benchmarks.oracle import build_oracles_for_contexts
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    read_jsonl,
)


def test_candidate_evaluation_computes_regret_per_system():
    records = (
        CandidatePerformanceRecord(
            run_id="r1",
            system_id="s1",
            candidate_id="fast",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=2.0,
        ),
        CandidatePerformanceRecord(
            run_id="r2",
            system_id="s1",
            candidate_id="slow",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=3.0,
        ),
        CandidatePerformanceRecord(
            run_id="r3",
            system_id="s2",
            candidate_id="other",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=5.0,
        ),
    )
    evaluations = build_candidate_evaluations(records, build_oracles_for_contexts(records))
    by_key = {(row.system_id, row.candidate_id): row for row in evaluations}
    assert by_key[("s1", "fast")].is_oracle is True
    assert by_key[("s1", "fast")].regret_vs_oracle == 0.0
    assert by_key[("s1", "slow")].regret_vs_oracle == 0.5
    assert by_key[("s2", "other")].is_oracle is True


def test_candidate_evaluation_writers(tmp_path):
    records = (
        CandidatePerformanceRecord(
            run_id="r",
            system_id="s",
            candidate_id="c",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=1.0,
            metadata={"relative_error_to_true": 1.0e-7},
        ),
    )
    evaluations = build_candidate_evaluations(records, build_oracles_for_contexts(records))
    table = write_candidate_evaluations(evaluations, tmp_path / "eval.jsonl")
    report = write_benchmark_report(evaluations, tmp_path / "report.md")
    assert read_jsonl(table)[0]["is_oracle"] is True
    assert "Benchmark Evaluation Report" in report.read_text(encoding="utf-8")

