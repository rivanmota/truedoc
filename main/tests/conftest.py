"""
Pytest bootstrap for the standalone analyzer tests.

Puts the project root (the directory containing ``app/``) on ``sys.path``
so ``from app.analyzers import metadata`` resolves no matter how pytest is
invoked (``pytest``, ``python -m pytest``, from the repo root or from
inside ``tests/``). Nothing here touches the network, the filesystem
beyond this path tweak, or the FastAPI app.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
