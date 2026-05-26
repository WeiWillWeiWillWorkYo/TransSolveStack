"""Trace serialization helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transsolvestack.core.result import RunTrace


def trace_to_record(trace: RunTrace) -> dict[str, Any]:
    """Convert a RunTrace to a JSONL/parquet-friendly record."""

    return asdict(trace)

