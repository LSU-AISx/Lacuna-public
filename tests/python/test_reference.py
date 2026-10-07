from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lacuna.ffi import CoreEvaluator
from lacuna.reference import (
    REFERENCE_FIXTURE_SCHEMA,
    ReferenceFixtureError,
    build_reference_document,
    check_reference_document,
    load_reference_document,
)


FIXTURES = Path(__file__).parents[2] / "validation" / "reference_fixtures.json"


def test_checked_in_reference_corpus_matches_one_shot_and_incremental_core(
    core: CoreEvaluator,
) -> None:
    document = load_reference_document(FIXTURES)
    result = check_reference_document(core, document)
    assert result == {
        "schema": REFERENCE_FIXTURE_SCHEMA,
        "fixture_set": "lacuna-core-slice-1",
        "cases_checked": 6,
        "incremental_equivalence_checked": True,
        "passed": True,
    }


def test_reference_generation_is_deterministic(core: CoreEvaluator) -> None:
    first = build_reference_document(core)
    second = build_reference_document(core)
    assert first == second


def test_reference_identity_and_numeric_drift_are_distinguished(
    core: CoreEvaluator,
) -> None:
    document = copy.deepcopy(load_reference_document(FIXTURES))
    document["cases"][0]["graph_sha256"] = "0" * 64
    with pytest.raises(ReferenceFixtureError, match="graph identity changed"):
        check_reference_document(core, document, verify_incremental=False)

    document = copy.deepcopy(load_reference_document(FIXTURES))
    document["cases"][0]["expected"]["spikes"][0]["t"] = "0x1.8p+1"
    with pytest.raises(ReferenceFixtureError, match=r"spikes\[0\]\.t"):
        check_reference_document(core, document, verify_incremental=False)
