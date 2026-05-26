import pytest
import io
import tarfile

import transsolvestack as tss
from transsolvestack.core.types import LinearOperatorSpec, LinearSystem, SolveContext
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.operators.synthetic import build_synthetic_system
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


def test_public_plan_entrypoint_builds_taichi_policy_plan():
    system = LinearSystem(
        system_id="sys",
        operator=LinearOperatorSpec(
            operator_id="op",
            kind="structured_stencil",
            shape=(1024, 1024),
            symmetry="spd",
        ),
    )
    context = SolveContext(context_id="ctx")
    plan = tss.plan(system, context)
    assert plan.backend == "taichi_gpu"
    assert plan.solver["name"] == "pcg"
    assert plan.audit["policy"] == "TaichiFirstHeuristicPlanner"


def test_public_plan_rejects_unknown_policy_mode():
    system = LinearSystem(
        system_id="sys",
        operator=LinearOperatorSpec(
            operator_id="op",
            kind="matrix_free",
            shape=(4, 4),
        ),
    )
    with pytest.raises(ValueError):
        tss.plan(system, SolveContext(context_id="ctx"), policy_mode="learned")


def test_public_artifact_plan_uses_profiled_policy():
    system = build_synthetic_system(
        "poisson_2d_stencil",
        (32, 32),
        dtype="float32",
        expected_operator_kind="structured_stencil",
    ).system
    context = SolveContext(context_id="default_f32")
    plan = tss.plan(
        system,
        context,
        policy_mode="benchmark_artifact",
        candidate_set="configs/candidates/phase1_taichi_gpu.yaml",
        evaluation_path="runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    assert plan.backend == "taichi_gpu"
    assert plan.audit["selector"] == "benchmark_artifact"
    assert plan.audit["is_oracle"] is True
    assert len(plan.fallback_chain) == 2


def test_public_auto_solve_runs_supported_taichi_system():
    pytest.importorskip("taichi")
    system = build_synthetic_system(
        "poisson_2d_stencil",
        (32, 32),
        dtype="float32",
        expected_operator_kind="structured_stencil",
    ).system
    context = SolveContext(context_id="default_f32", max_iter=300)
    result = tss.auto_solve(
        system,
        rhs=None,
        context=context,
        policy_mode="benchmark_artifact",
        candidate_set="configs/candidates/phase1_taichi_gpu.yaml",
        evaluation_path="runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
        device_memory_gb=0.5,
    )
    assert result.status == "success"
    assert result.trace.backend == "taichi_gpu"
    assert result.trace.final_residual_norm <= 1.0e-6
    assert result.trace.metadata["relative_error_to_true"] < 5.0e-3


def test_public_solve_rejects_general_rhs_until_sparse_path_exists():
    system = build_synthetic_system(
        "poisson_2d_stencil",
        (32, 32),
        dtype="float32",
        expected_operator_kind="structured_stencil",
    ).system
    with pytest.raises(NotImplementedError):
        tss.solve(system, rhs=[1.0], context=SolveContext(context_id="default_f32"))


def test_public_csr_diagnostic_api_reports_actual_symmetry():
    csr = CsrMatrix(
        matrix_id="fixture:spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://spd",
    )

    diagnostic = tss.diagnose_csr_matrix(csr)

    assert diagnostic.actual_symmetric is True
    assert diagnostic.cg_candidate is True


def test_public_solve_csr_runs_tiny_spd_fixture():
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://spd",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.solve_csr(
        csr,
        rhs,
        context=SolveContext(
            context_id="public_csr",
            tolerance_rel=1.0e-6,
            max_iter=32,
            precision="float64",
        ),
        solver="cg",
        device_memory_gb=0.25,
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-6
    assert result.trace.metadata["operator_backend"] == "taichi_csr"


def test_public_solve_csr_runs_tiny_nonsymmetric_bicgstab_fixture():
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:nonsym",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, 1.0, 2.0, 3.0, 1.0, -1.0, 2.0),
        field="real",
        symmetry="general",
        source_path="memory://nonsym",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.solve_csr(
        csr,
        rhs,
        context=SolveContext(
            context_id="public_csr_bicgstab",
            tolerance_rel=1.0e-8,
            max_iter=64,
            precision="float64",
        ),
        solver="bicgstab",
        preconditioner="jacobi",
        device_memory_gb=0.25,
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"


def test_public_solve_csr_runs_tiny_nonsymmetric_gmres_fixture():
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:nonsym_gmres",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, 1.0, 2.0, 3.0, 1.0, -1.0, 2.0),
        field="real",
        symmetry="general",
        source_path="memory://nonsym_gmres",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.solve_csr(
        csr,
        rhs,
        context=SolveContext(
            context_id="public_csr_gmres",
            tolerance_rel=1.0e-8,
            max_iter=64,
            precision="float64",
        ),
        solver="gmres",
        solver_parameters={"restart": 3},
        device_memory_gb=0.25,
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"
    assert result.trace.metadata["restart"] == 3


def test_public_csr_linear_ranker_api_trains_offline_model():
    export = tss.train_csr_linear_ranker(
        "runs/phase1_csr_training_tensors/csr_training_tensors.json",
        "runs/phase1_csr_training_tensors/csr_training_request_index.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.model_trained is True
    assert export.summary.runtime_selector_changed is False
    assert export.summary.num_features == 374
    assert export.summary.num_pairwise_constraints == 110
    assert export.summary.num_pairwise_updates == 148
    assert export.summary.train_oracle_top1_accuracy == 0.5
    assert export.summary.eval_oracle_top1_accuracy == 0.0


def test_public_csr_selector_model_eval_api_blocks_runtime_integration():
    export = tss.evaluate_csr_selector_models(
        "runs/phase1_csr_learning_readiness/csr_learning_summary.json",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_summary.json",
        "runs/phase1_csr_linear_ranker/csr_linear_ranker_predictions.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.runtime_selector_changed is False
    assert export.summary.challenger_model_id == "csr_pairwise_linear_ranker_v1"
    assert export.summary.challenger_beats_baseline is False
    assert export.summary.challenger_runtime_eligible is False
    assert export.summary.runtime_selected_model_id is None
    assert "below_baseline_profiled_success_rate" in export.summary.challenger_gate_failures


def test_public_csr_benchmark_expansion_plan_api_builds_safe_queue():
    plan = tss.plan_csr_benchmark_expansion()

    assert plan.summary.status == "passed"
    assert plan.summary.runtime_selector_changed is False
    assert plan.summary.executes_gpu is False
    assert plan.summary.planned_matrices == 8
    assert plan.summary.planned_candidate_jobs == 24
    assert plan.summary.planned_gpu_solve_attempts == 24
    assert len(plan.matrix_rows) == 8
    assert len(plan.candidate_rows) == 24
    assert plan.summary.by_candidate_profile == {"general": 4, "symmetric": 4}
    assert all(row.requires_cpu_screen for row in plan.candidate_rows)
    assert all(
        row.planned_status == "queued_cpu_screen_required"
        for row in plan.candidate_rows
    )


def test_public_csr_transformer_ready_bundle_api_exports_contract(tmp_path):
    bundle = tss.build_csr_transformer_ready_bundle(output_dir=tmp_path)

    assert bundle["summary"]["status"] == "passed"
    assert bundle["summary"]["transformer_connectable"] is True
    assert bundle["summary"]["runtime_selector_changed"] is False
    assert bundle["summary"]["num_selector_rows"] == 132
    assert bundle["summary"]["num_model_requests"] == 20
    assert bundle["summary"]["num_global_candidates"] == 9
    assert bundle["summary"]["matrix_feature_dim"] == 21
    assert bundle["summary"]["candidate_feature_dim"] == 16
    assert bundle["summary"]["validation_error_count"] == 0
    assert (tmp_path / "csr_transformer_training_tensors.json").exists()


def test_public_csr_transformer_ranker_api_trains_attention_model():
    export = tss.train_csr_transformer_ranker()

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_transformer_ranker_v1"
    assert export.summary.model_family == "masked_self_attention_ranker_v1"
    assert export.summary.model_trained is True
    assert export.summary.runtime_selector_changed is False
    assert export.summary.num_requests == 20
    assert export.summary.num_eval_requests == 5
    assert export.summary.num_eval_oracle_requests == 4
    assert export.summary.token_feature_dim == 37
    assert export.summary.scorer_feature_dim == 62
    assert export.summary.eval_oracle_top1_accuracy == 0.5
    assert export.summary.eval_profiled_success_selection_rate == 0.6
    assert export.summary.eval_non_success_selection_count == 2


def test_public_csr_transformer_training_entrypoint_api_exports_contract(tmp_path):
    export = tss.build_csr_transformer_training_entrypoint(output_dir=tmp_path)

    assert export["summary"]["status"] == "passed"
    assert export["summary"]["training_entrypoint_ready"] is True
    assert export["summary"]["model_training_required"] is True
    assert export["summary"]["model_trained"] is False
    assert export["summary"]["runtime_selector_changed"] is False
    assert export["summary"]["transformer_connectable"] is True
    assert export["summary"]["current_quality_gate_runtime_eligible"] is False
    assert export["summary"]["num_model_requests"] == 20
    assert export["summary"]["num_tensor_requests"] == 20
    assert (tmp_path / "csr_transformer_training_job_spec.json").exists()
    assert (tmp_path / "csr_transformer_training_quality_contract.json").exists()


def test_public_solve_csr_runs_tiny_richardson_fixture():
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:richardson_spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://richardson_spd",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.solve_csr(
        csr,
        rhs,
        context=SolveContext(
            context_id="public_csr_richardson",
            tolerance_rel=1.0e-8,
            max_iter=128,
            precision="float64",
        ),
        solver="richardson",
        device_memory_gb=0.25,
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"


def test_public_solve_csr_runs_tiny_chebyshev_fixture_with_explicit_bounds():
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:chebyshev_spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://chebyshev_spd",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.solve_csr(
        csr,
        rhs,
        context=SolveContext(
            context_id="public_csr_chebyshev",
            tolerance_rel=1.0e-8,
            max_iter=64,
            precision="float64",
        ),
        solver="chebyshev",
        solver_parameters={
            "lambda_min": 0.5,
            "lambda_max": 1.5,
        },
        device_memory_gb=0.25,
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"
    assert result.trace.metadata["lambda_min"] == pytest.approx(0.5)
    assert result.trace.metadata["lambda_max"] == pytest.approx(1.5)


def test_public_auto_solve_csr_selects_and_runs_tiny_fixture(tmp_path):
    pytest.importorskip("taichi")
    csr = CsrMatrix(
        matrix_id="fixture:auto_spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://auto_spd",
    )
    selector_rows = tmp_path / "csr_selector_rows.jsonl"
    write_jsonl(
        (
            {
                "schema_version": "phase1_csr_selector_features_v4",
                "matrix_id": csr.matrix_id,
                "context_id": "fixture_auto",
                "candidate_id": "fixture_pcg_jacobi_float64",
                "solver": "pcg",
                "preconditioner": "jacobi",
                "precision": "float64",
                "solver_parameters": {},
                "label_is_oracle": True,
                "target_status": "success",
                "target_success_rate": 1.0,
                "target_measurement_repeats": 3,
                "target_solve_time_ms": 1.0,
                "target_median_solve_time_ms": 1.0,
                "target_solve_time_iqr_ms": 0.0,
                "target_wall_time_ms": 1.0,
                "target_regret_vs_oracle_ms": 0.0,
                "target_num_iterations": 2,
                "target_final_relative_residual": 1.0e-12,
                "target_cpu_recomputed_relative_residual": 1.0e-12,
                "target_solution_relative_error": 1.0e-12,
                "features": {},
            },
        ),
        selector_rows,
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))

    result = tss.auto_solve_csr(
        csr,
        rhs,
        selector_path=selector_rows,
        context=SolveContext(
            context_id="fixture_auto",
            tolerance_rel=1.0e-6,
            max_iter=32,
            precision="float64",
        ),
        device_memory_gb=0.25,
    )

    auto_metadata = result.trace.metadata["auto_solve_csr"]
    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.backend == "taichi_gpu"
    assert result.trace.final_residual_norm <= 1.0e-6
    assert auto_metadata["candidate_id"] == "fixture_pcg_jacobi_float64"
    assert auto_metadata["plan"]["solver"]["name"] == "pcg"
    assert auto_metadata["plan"]["audit"]["is_oracle"] is True


def test_public_dataset_download_plan_api_is_dry_run():
    plan = tss.plan_dataset_downloads("configs/datasets/phase1_external_small.yaml")
    assert plan.dry_run is True
    assert len(plan.entries) == 3
    assert plan.total_estimated_download_gb < 0.01


def test_public_suitesparse_index_api_builds_metadata_index(tmp_path):
    stats_path = tmp_path / "ssstats.csv"
    stats_path.write_text(
        "\n".join(
            [
                "1",
                "31-Oct-2023 18:12:37",
                "HB,1138_bus,1138,1138,4054,1,0,0,1,1,1,power network problem,4054",
                "",
            ]
        ),
        encoding="utf-8",
    )
    urls_path = tmp_path / "mm_urls.txt"
    urls_path.write_text(
        "https://sparse.tamu.edu/MM/HB/1138_bus.tar.gz\n",
        encoding="utf-8",
    )
    mm_root = tmp_path / "MM"
    (mm_root / "HB").mkdir(parents=True)
    (mm_root / "HB" / "1138_bus.tar.gz").write_bytes(b"archive")

    index = tss.build_suitesparse_index(stats_path, urls_path, mm_root)

    assert index.summary.status == "complete"
    assert index.summary.indexed_matrices == 1
    assert index.entries[0].matrix_id == "suitesparse:HB/1138_bus"
    assert index.entries[0].download_present is True


def test_public_suitesparse_selection_api_reads_index(tmp_path):
    stats_path = tmp_path / "ssstats.csv"
    stats_path.write_text(
        "\n".join(
            [
                "1",
                "31-Oct-2023 18:12:37",
                "HB,1138_bus,1138,1138,4054,1,0,0,1,1,1,power network problem,4054",
                "",
            ]
        ),
        encoding="utf-8",
    )
    urls_path = tmp_path / "mm_urls.txt"
    urls_path.write_text(
        "https://sparse.tamu.edu/MM/HB/1138_bus.tar.gz\n",
        encoding="utf-8",
    )
    mm_root = tmp_path / "MM"
    (mm_root / "HB").mkdir(parents=True)
    (mm_root / "HB" / "1138_bus.tar.gz").write_bytes(b"archive")
    index = tss.build_suitesparse_index(stats_path, urls_path, mm_root)
    index_path = tmp_path / "matrix_manifest.jsonl"
    from transsolvestack.datasets import write_suitesparse_index

    write_suitesparse_index(index, index_path)

    selection = tss.select_suitesparse_subset(
        index_path,
        criteria=tss.MatrixSelectionCriteria(max_matrices=1),
    )

    assert selection.summary.status == "ready"
    assert selection.summary.selected_matrices == 1
    assert selection.records[0].matrix_id == "suitesparse:HB/1138_bus"


def test_public_suitesparse_header_probe_api_reads_selection_file(tmp_path):
    archive_path = tmp_path / "tiny.tar.gz"
    payload = (
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 5\n"
        "1 1 1.0\n"
    ).encode("utf-8")
    info = tarfile.TarInfo("tiny/tiny.mtx")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    selection_path = tmp_path / "selected_matrices.jsonl"
    write_jsonl(
        (
            {
                "selection_rank": 1,
                "matrix_id": "suitesparse:Test/tiny",
                "source": "suitesparse",
                "group": "Test",
                "name": "tiny",
                "local_path": str(archive_path),
                "n_rows": 3,
                "n_cols": 3,
                "nnz": 5,
                "is_binary": False,
                "is_real": True,
                "archive_size_bytes": archive_path.stat().st_size,
                "kind": "unit-test problem",
                "size_bucket": "tiny",
            },
        ),
        selection_path,
    )

    probe = tss.probe_suitesparse_headers(selection_path)

    assert probe.summary.status == "passed"
    assert probe.records[0].field == "real"
    assert probe.records[0].shape_matches is True


def test_public_policy_training_export_api():
    rows = tss.export_policy_training_rows(
        "configs/workloads/phase1_smoke.yaml",
        "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    assert len(rows) == 21
    assert sum(1 for row in rows if row.label_is_oracle) == 7


def test_public_csr_selector_export_api():
    export = tss.export_csr_selector_rows(
        "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
        (
            "runs/phase1_taichi_csr_solve/csr_solve_results.jsonl",
            "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_results.jsonl",
            "runs/phase1_taichi_csr_gmres/csr_gmres_results.jsonl",
            "runs/phase1_taichi_csr_richardson/csr_richardson_results.jsonl",
            "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_results.jsonl",
        ),
        screened_results_path=(
            "runs/phase1_taichi_csr_solve/csr_solve_summary.json",
            "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_summary.json",
            "runs/phase1_taichi_csr_gmres/csr_gmres_summary.json",
            "runs/phase1_taichi_csr_richardson/csr_richardson_summary.json",
            "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_summary.json",
        ),
    )
    assert export.summary.status == "passed"
    assert export.summary.num_diagnostic_rows == 12
    assert export.summary.num_selector_rows == 108
    assert export.summary.num_failed_rows == 15
    assert export.summary.num_applicability_rows == 78
    assert export.summary.num_oracle_rows == 4


def test_public_csr_learning_export_api():
    export = tss.export_csr_learning_rows(
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl"
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_learning_features_v1"
    assert export.summary.num_rows == 108
    assert export.summary.num_train_rows == 81
    assert export.summary.num_eval_rows == 27
    assert export.summary.num_eval_predictions == 3
    assert export.summary.eval_oracle_top1_accuracy == 1.0 / 3.0
    assert export.summary.eval_profiled_success_rate == 2.0 / 3.0


def test_public_csr_model_contract_export_api():
    export = tss.export_csr_model_contract(
        "runs/phase1_csr_learning_readiness/csr_learning_rows.jsonl",
        "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_model_contract_v1"
    assert export.summary.num_requests == 12
    assert export.summary.num_predictions == 3
    assert export.summary.model_required is False
    assert export.summary.runtime_selector_changed is False
    assert export.summary.validation_error_count == 0


def test_public_csr_training_tensor_export_api():
    export = tss.export_csr_training_tensors(
        "runs/phase1_csr_model_contract/csr_model_requests.jsonl",
        "runs/phase1_csr_model_contract/csr_model_targets.jsonl",
    )

    assert export.summary.status == "passed"
    assert export.summary.schema_version == "phase1_csr_training_tensors_v1"
    assert export.summary.num_requests == 12
    assert export.summary.num_global_candidates == 9
    assert export.summary.num_active_candidate_slots == 108
    assert export.summary.model_required is False
    assert export.summary.runtime_selector_changed is False


def test_public_plan_csr_selects_profiled_oracle_plan():
    rows = {
        str(row["matrix_id"]): row
        for row in read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
    }
    selector_rows = read_jsonl("runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl")
    oracle = next(
        row
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:HB/curtis54" and row["label_is_oracle"]
    )

    selection = tss.plan_csr(rows["suitesparse:HB/curtis54"])

    assert selection.reason == "profiled_success"
    assert selection.candidate_id == oracle["candidate_id"]
    assert selection.plan.backend == "taichi_gpu"
    assert selection.plan.solver["name"] == oracle["solver"]
    assert selection.plan.preconditioner["name"] == oracle["preconditioner"]
    assert selection.plan.audit["selector"] == "csr_artifact"
    assert selection.plan.audit["is_oracle"] is True
    assert set(selection.fallback_candidate_ids) == {
        row["candidate_id"]
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:HB/curtis54"
        and not row["label_is_oracle"]
        and row["target_status"] == "success"
    }
