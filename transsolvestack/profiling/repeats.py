"""Repeated-measurement aggregation for benchmark-like artifacts."""

from __future__ import annotations

import math
from typing import Any, Iterable


def aggregate_repeated_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Collapse repeated solve rows into one selector-compatible record.

    The returned row keeps the old top-level fields, with ``solve_time_ms`` and
    ``wall_time_ms`` set to medians. It also records all repeat samples so
    selector exports can audit timing stability.
    """

    repeat_rows = tuple(dict(row) for row in rows)
    if not repeat_rows:
        raise ValueError("cannot aggregate zero repeated rows")
    successes = tuple(row for row in repeat_rows if row.get("status") == "success")
    timing_rows = successes or repeat_rows
    representative = _median_row(timing_rows, key="solve_time_ms")
    aggregated = dict(representative)
    solve_samples = tuple(_float(row.get("solve_time_ms")) for row in timing_rows)
    wall_samples = tuple(_float(row.get("wall_time_ms")) for row in timing_rows)
    failure_reasons = tuple(
        sorted(
            {
                str(reason)
                for row in repeat_rows
                for reason in row.get("failure_reasons", ())
            }
        )
    )
    all_success = len(successes) == len(repeat_rows)
    aggregated.update(
        {
            "status": "success" if all_success else "failed",
            "failure_reasons": failure_reasons,
            "measurement_repeats": len(repeat_rows),
            "success_count": len(successes),
            "success_rate": len(successes) / len(repeat_rows),
            "repeat_records": repeat_rows,
            "solve_time_samples_ms": solve_samples,
            "wall_time_samples_ms": wall_samples,
            "median_solve_time_ms": _median(solve_samples),
            "median_wall_time_ms": _median(wall_samples),
            "min_solve_time_ms": min(solve_samples),
            "max_solve_time_ms": max(solve_samples),
            "solve_time_iqr_ms": _iqr(solve_samples),
            "wall_time_iqr_ms": _iqr(wall_samples),
        }
    )
    aggregated["solve_time_ms"] = aggregated["median_solve_time_ms"]
    aggregated["wall_time_ms"] = aggregated["median_wall_time_ms"]
    if successes:
        aggregated["final_relative_residual"] = max(
            _float(row.get("final_relative_residual")) for row in successes
        )
        aggregated["cpu_recomputed_relative_residual"] = max(
            _float(row.get("cpu_recomputed_relative_residual")) for row in successes
        )
        aggregated["solution_relative_error"] = max(
            _float(row.get("solution_relative_error")) for row in successes
        )
    return aggregated


def _median_row(rows: tuple[dict[str, Any], ...], *, key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: _float(row.get(key)))
    return ordered[len(ordered) // 2]


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        return math.inf
    return result


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _iqr(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    return _percentile(ordered, 0.75) - _percentile(ordered, 0.25)


def _percentile(ordered: tuple[float, ...] | list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return lower + (upper - lower) * (position - lower_index)
