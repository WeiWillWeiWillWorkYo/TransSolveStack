"""Known external sparse matrix sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixSource:
    name: str
    base_url: str
    license_note: str
    notes: str


SUITESPARSE = MatrixSource(
    name="SuiteSparse Matrix Collection",
    base_url="https://sparse.tamu.edu",
    license_note="Matrices are published with CC-BY 4.0 terms; preserve metadata.",
    notes="Primary real sparse matrix benchmark source.",
)

NIST_MATRIX_MARKET = MatrixSource(
    name="NIST Matrix Market",
    base_url="https://math.nist.gov/MatrixMarket/",
    license_note="Check matrix-level metadata and NIST terms before redistribution.",
    notes="Classic sparse matrix exchange-format repository.",
)

KNOWN_SOURCES = {
    "suitesparse": SUITESPARSE,
    "nist_matrix_market": NIST_MATRIX_MARKET,
}

