"""Print lightweight TransSolveStack environment information."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.backends.registry import default_backend_registry
from transsolvestack.backends.taichi_backend import probe_taichi_backend


def main() -> None:
    registry = default_backend_registry()
    print("registered_backends:", ",".join(registry.names()))
    info = probe_taichi_backend()
    print("taichi_available:", info.available)
    if info.reason:
        print("taichi_reason:", info.reason)


if __name__ == "__main__":
    main()
