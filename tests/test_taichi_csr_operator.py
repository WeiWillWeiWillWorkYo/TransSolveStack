import pytest


def test_taichi_csr_operator_matvec_matches_cpu_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:tiny",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://tiny",
    )
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)

    actual = operator.apply((1.0, 1.0, 1.0))

    assert actual == pytest.approx(csr.matvec((1.0, 1.0, 1.0)), abs=1.0e-6)
    assert operator.spec.kind == "assembled_sparse"
    assert operator.spec.device_resident is True


def test_taichi_csr_operator_residual_dot_and_norm_match_cpu_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:tiny",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://tiny",
    )
    operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)
    x = (1.0, 2.0, 3.0)
    b = csr.matvec(x)

    residual = operator.residual(x=x, b=b)

    assert residual == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-6)
    assert operator.norm("r") <= 1.0e-6
    assert operator.dot("y", "y") == pytest.approx(
        sum(value * value for value in b),
        rel=1.0e-6,
        abs=1.0e-6,
    )
    assert operator.norm("y") == pytest.approx(
        sum(value * value for value in b) ** 0.5,
        rel=1.0e-6,
        abs=1.0e-6,
    )
    with pytest.raises(ValueError):
        operator.dot("x", "y")


def test_taichi_csr_operator_cg_and_pcg_solve_tiny_spd_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:tiny",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://tiny",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))
    context = SolveContext(context_id="tiny_spd", tolerance_rel=1.0e-6, max_iter=32)
    engine = TaichiExecutionEngine()

    cg_operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)
    cg = engine.solve(
        cg_operator,
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_cg",
            backend="taichi_gpu",
            solver={"name": "cg"},
            preconditioner={"name": "none"},
        ),
    )

    pcg_operator = TaichiCsrMatrixOperator.from_csr_matrix(csr)
    pcg = engine.solve(
        pcg_operator,
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_pcg",
            backend="taichi_gpu",
            solver={"name": "pcg"},
            preconditioner={"name": "jacobi"},
        ),
    )

    assert cg.status == "success"
    assert pcg.status == "success"
    assert cg.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-5)
    assert pcg.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-5)
    assert cg.trace.final_residual_norm <= 1.0e-6
    assert pcg.trace.final_residual_norm <= 1.0e-6
    assert cg.trace.metadata["operator_backend"] == "taichi_csr"
    assert pcg.trace.metadata["preconditioner"] == "jacobi"


def test_taichi_csr_operator_pcg_symmetric_equilibration_solves_scaled_spd_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:scaled_spd",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(1000.0, -10.0, -10.0, 4.0, -0.1, -0.1, 0.1),
        field="real",
        symmetry="symmetric",
        source_path="memory://scaled_spd",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))
    context = SolveContext(
        context_id="tiny_scaled_spd",
        tolerance_rel=1.0e-8,
        max_iter=64,
        precision="float64",
    )

    result = TaichiExecutionEngine().solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_pcg_symmetric_equilibration",
            backend="taichi_gpu",
            solver={"name": "pcg"},
            preconditioner={"name": "symmetric_equilibration"},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "symmetric_equilibration"
    assert result.trace.metadata["equilibration"] == "symmetric_diagonal"
    assert result.trace.metadata["scaled_relative_residual_norm"] <= 1.0e-8


def test_taichi_csr_operator_bicgstab_solves_tiny_nonsymmetric_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
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
    context = SolveContext(
        context_id="tiny_nonsym",
        tolerance_rel=1.0e-8,
        max_iter=64,
        precision="float64",
    )
    engine = TaichiExecutionEngine()

    result = engine.solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_bicgstab",
            backend="taichi_gpu",
            solver={"name": "bicgstab"},
            preconditioner={"name": "jacobi"},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"


def test_taichi_csr_operator_bicgstab_ilu0_solves_tiny_nonsymmetric_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:nonsym_ilu0",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(4.0, 1.0, 2.0, 3.0, 1.0, -1.0, 2.0),
        field="real",
        symmetry="general",
        source_path="memory://nonsym_ilu0",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))
    context = SolveContext(
        context_id="tiny_nonsym_ilu0",
        tolerance_rel=1.0e-10,
        max_iter=32,
        precision="float64",
    )

    result = TaichiExecutionEngine().solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_bicgstab_ilu0",
            backend="taichi_gpu",
            solver={"name": "bicgstab"},
            preconditioner={"name": "ilu0", "pivot_tolerance": 1.0e-12},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-7)
    assert result.trace.final_residual_norm <= 1.0e-10
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "ilu0"
    assert result.trace.metadata["ilu0_valid"] is True
    assert result.trace.metadata["ilu0_factorization"] == (
        "doolittle_level_zero_csr_pattern"
    )


def test_taichi_csr_operator_bicgstab_row_column_equilibration_solves_scaled_nonsym_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:scaled_nonsym",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(1000.0, 2.0, 3.0, 0.01, 4.0, -5.0, 20.0),
        field="real",
        symmetry="general",
        source_path="memory://scaled_nonsym",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))
    context = SolveContext(
        context_id="tiny_scaled_nonsym",
        tolerance_rel=1.0e-8,
        max_iter=128,
        precision="float64",
    )

    result = TaichiExecutionEngine().solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_bicgstab_row_column_equilibration",
            backend="taichi_gpu",
            solver={"name": "bicgstab"},
            preconditioner={"name": "row_column_equilibration", "passes": 4},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "row_column_equilibration"
    assert result.trace.metadata["equilibration"] == "row_column_l1"
    assert result.trace.metadata["row_column_equilibration_passes"] == 4


def test_taichi_csr_operator_gmres_solves_tiny_nonsymmetric_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
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
    context = SolveContext(
        context_id="tiny_nonsym_gmres",
        tolerance_rel=1.0e-8,
        max_iter=64,
        precision="float64",
    )
    engine = TaichiExecutionEngine()

    result = engine.solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_gmres",
            backend="taichi_gpu",
            solver={"name": "gmres", "restart": 3},
            preconditioner={"name": "jacobi"},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"
    assert result.trace.metadata["restart"] == 3
    assert result.trace.metadata["orthogonalization_backend"] == (
        "taichi_gpu_modified_gram_schmidt"
    )


def test_taichi_csr_operator_gmres_row_column_equilibration_solves_scaled_nonsym_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
    csr = CsrMatrix(
        matrix_id="fixture:scaled_nonsym_gmres",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 2, 5, 7),
        col_ind=(0, 1, 0, 1, 2, 1, 2),
        values=(1000.0, 2.0, 3.0, 0.01, 4.0, -5.0, 20.0),
        field="real",
        symmetry="general",
        source_path="memory://scaled_nonsym_gmres",
    )
    rhs = csr.matvec((1.0, 1.0, 1.0))
    context = SolveContext(
        context_id="tiny_scaled_nonsym_gmres",
        tolerance_rel=1.0e-8,
        max_iter=128,
        precision="float64",
    )

    result = TaichiExecutionEngine().solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_gmres_row_column_equilibration",
            backend="taichi_gpu",
            solver={"name": "gmres", "restart": 3},
            preconditioner={"name": "row_column_equilibration", "passes": 4},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "row_column_equilibration"
    assert result.trace.metadata["equilibration"] == "row_column_l1"
    assert result.trace.metadata["restart"] == 3


def test_taichi_csr_operator_richardson_solves_tiny_spd_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
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
    context = SolveContext(
        context_id="tiny_richardson",
        tolerance_rel=1.0e-8,
        max_iter=128,
        precision="float64",
    )
    engine = TaichiExecutionEngine()

    result = engine.solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_richardson",
            backend="taichi_gpu",
            solver={"name": "richardson", "omega": 2.0 / 3.0},
            preconditioner={"name": "jacobi"},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"
    assert result.trace.metadata["omega"] == pytest.approx(2.0 / 3.0)


def test_taichi_csr_operator_chebyshev_solves_tiny_spd_fixture():
    pytest.importorskip("taichi")
    from transsolvestack.core.result import PolicyPlan
    from transsolvestack.core.types import SolveContext
    from transsolvestack.datasets.csr import CsrMatrix
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda
    from transsolvestack.runtime.engine import TaichiExecutionEngine

    ensure_taichi_cuda(device_memory_gb=0.25)
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
    context = SolveContext(
        context_id="tiny_chebyshev",
        tolerance_rel=1.0e-8,
        max_iter=64,
        precision="float64",
    )
    engine = TaichiExecutionEngine()

    result = engine.solve(
        TaichiCsrMatrixOperator.from_csr_matrix(csr, dtype="float64"),
        rhs=rhs,
        context=context,
        plan=PolicyPlan(
            plan_id="tiny_chebyshev",
            backend="taichi_gpu",
            solver={
                "name": "chebyshev",
                "lambda_min": 0.5,
                "lambda_max": 1.5,
            },
            preconditioner={"name": "jacobi"},
        ),
    )

    assert result.status == "success"
    assert result.solution == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)
    assert result.trace.final_residual_norm <= 1.0e-8
    assert result.trace.metadata["operator_backend"] == "taichi_csr"
    assert result.trace.metadata["preconditioner"] == "jacobi"
    assert result.trace.metadata["lambda_min"] == pytest.approx(0.5)
    assert result.trace.metadata["lambda_max"] == pytest.approx(1.5)
