"""Taichi Jacobi preconditioner skeleton."""

from __future__ import annotations

from typing import Any

from transsolvestack.operators.base import LinearOperator
from transsolvestack.preconditioners.base import Preconditioner


class TaichiJacobiPreconditioner(Preconditioner):
    """GPU Jacobi preconditioner contract."""

    name = "taichi_jacobi"

    def setup(self, operator: LinearOperator) -> None:
        raise NotImplementedError(
            "Diagonal extraction/setup kernel is not implemented yet."
        )

    def apply(self, r: Any, z: Any | None = None) -> Any:
        raise NotImplementedError("Jacobi apply kernel is not implemented yet.")


class TaichiBlockJacobiPreconditioner(TaichiJacobiPreconditioner):
    """GPU block-Jacobi preconditioner contract."""

    name = "taichi_block_jacobi"

    def setup(self, operator: LinearOperator) -> None:
        raise NotImplementedError(
            "Block diagonal extraction/setup kernel is not implemented yet."
        )


class TaichiSymmetricEquilibrationPreconditioner(TaichiJacobiPreconditioner):
    """GPU symmetric diagonal equilibration contract."""

    name = "taichi_symmetric_equilibration"

    def setup(self, operator: LinearOperator) -> None:
        raise NotImplementedError(
            "Symmetric equilibration setup is implemented by Taichi CSR operators."
        )


class TaichiRowColumnEquilibrationPreconditioner(TaichiJacobiPreconditioner):
    """GPU row/column diagonal equilibration contract."""

    name = "taichi_row_column_equilibration"

    def setup(self, operator: LinearOperator) -> None:
        raise NotImplementedError(
            "Row/column equilibration setup is implemented by Taichi CSR operators."
        )


class TaichiIlu0Preconditioner(TaichiJacobiPreconditioner):
    """GPU ILU0 preconditioner contract."""

    name = "taichi_ilu0"

    def setup(self, operator: LinearOperator) -> None:
        raise NotImplementedError("ILU0 setup is implemented by Taichi CSR operators.")
