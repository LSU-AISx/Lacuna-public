from __future__ import annotations

from pathlib import Path

import pytest

from lacuna import CoreEvaluator


@pytest.fixture(scope="session")
def core() -> CoreEvaluator:
    candidates = sorted(Path("build").glob("liblacuna_core.*"))
    if not candidates:
        pytest.fail("C core is not built; run 'cmake -S . -B build && cmake --build build'")
    return CoreEvaluator(candidates[0])
