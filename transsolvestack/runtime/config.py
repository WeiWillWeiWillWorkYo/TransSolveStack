"""Runtime configuration loading."""

from __future__ import annotations

from pathlib import Path

from transsolvestack.config import load_yaml
from transsolvestack.runtime.resource_budget import ResourceBudget


def load_resource_budget(path: str | Path) -> ResourceBudget:
    data = load_yaml(path)
    limits = dict(data.get("limits", {}))
    if "mode" in data:
        limits.setdefault("mode", data["mode"])
    return ResourceBudget(**limits)

