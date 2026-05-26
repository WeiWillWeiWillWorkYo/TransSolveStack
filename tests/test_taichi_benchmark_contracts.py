from transsolvestack.core.result import SolveResult
from transsolvestack.solvers.taichi_cg import (
    TaichiBiCGSTABSolver,
    TaichiCGSolver,
    TaichiPCGSolver,
)


def test_taichi_solver_names_match_candidate_configs():
    assert TaichiCGSolver().name == "cg"
    assert TaichiPCGSolver().name == "pcg"
    assert TaichiBiCGSTABSolver().name == "bicgstab"


def test_taichi_solver_dispatch_uses_operator_method():
    class Operator:
        def solve_pcg(self, rhs, context, plan):
            return "called"

    assert TaichiPCGSolver().solve(Operator(), None, None, None) == "called"

