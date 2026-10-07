"""Checks for the prescribed numerical transfer validation protocol."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lava.lib.dl.slayer")

from lacuna.importers.slayer import (
    import_slayer_feedforward,
    validate_slayer_feedforward,
)
from validation.official_slayer_numerical_matrix import (
    build_model,
    case_grid,
    check_analytical_control,
    make_inputs,
    summarize,
)


def test_matrix_covers_prescribed_regimes_without_duplicate_cases():
    cases = case_grid()
    assert len(cases) == len({case["id"] for case in cases}) == 48
    assert {case["source_scale_argument"] for case in cases} == {64, 4096}
    assert {case["bins"] for case in cases} == {256, 1024}
    assert {case["timestep"] for case in cases} == {0.1, 0.25, 1.0}
    assert all(
        case["source_state_scale"] == 64 * case["source_scale_argument"]
        for case in cases
    )
    assert any(len(case["decay_integers"]) == 6 for case in cases)
    assert sum(case["analytical_control"] is not None for case in cases) == 22


@pytest.mark.parametrize("case", case_grid(), ids=lambda case: case["id"])
def test_matrix_inputs_and_source_parameters_are_reproducible(case):
    first, second = make_inputs(case), make_inputs(case)
    assert torch.equal(first, second)
    assert bool(torch.all((first == 0) | (first == 1)))
    assert first.shape == (4, *case["input_shape"], case["bins"])
    model = build_model(case)
    imported = import_slayer_feedforward(
        model,
        input_shape=tuple(case["input_shape"]),
        timestep=case["timestep"],
        acknowledge_quantization=True,
    )
    assert [layer["decay_integer"] for layer in imported.metadata["layers"]] == case[
        "decay_integers"
    ]
    assert all(
        layer["state_scale"] == case["source_state_scale"]
        for layer in imported.metadata["layers"]
    )


@pytest.mark.parametrize("scale", [64, 4096])
@pytest.mark.parametrize("pattern", ["dense", "cancellation"])
def test_dyadic_controls_have_analytical_counts_in_both_engines(core, scale, pattern):
    case = next(
        case
        for case in case_grid()
        if (
            case["architecture"] == "dyadic_dense"
            and case["pattern"] == pattern
            and case["source_scale_argument"] == scale
        )
    )
    model, inputs = build_model(case), make_inputs(case)
    imported = import_slayer_feedforward(
        model,
        input_shape=(8,),
        timestep=case["timestep"],
        acknowledge_quantization=True,
    )
    report = validate_slayer_feedforward(
        model, imported, inputs, library=core._lib._name, source_batch_size=4
    )
    assert check_analytical_control(case, inputs, report) == "passed"


@pytest.mark.parametrize(
    "case",
    [case for case in case_grid() if case["analytical_control"] is not None],
    ids=lambda case: case["id"],
)
def test_source_control_spikes_match_analytical_bin_positions(case):
    model = build_model(case)
    inputs = make_inputs(case)
    value = inputs
    with torch.no_grad():
        for block in model:
            value = block(value)
            if hasattr(block, "neuron"):
                expected = (
                    inputs
                    if case["analytical_control"] == "identity"
                    else torch.zeros_like(value)
                )
                assert torch.equal(value, expected)


def test_aggregate_does_not_treat_unlabelled_argmax_agreement_as_accuracy():
    result = summarize(
        [
            {
                "status": "completed",
                "analytical_control_result": None,
                "comparison": {
                    "samples": 2,
                    "exact_spike_match_on_batch": False,
                    "source_predictions": [0, 1],
                    "lacuna_predictions": [1, 1],
                    "off_grid_spikes": [],
                    "layers": [{"mismatched_bins": 3}],
                },
            },
            {"status": "error"},
        ]
    )
    assert result["paired_predictions_matching"] == 1
    assert result["paired_predictions"] == 2
    assert result["error_cases"] == 1
    assert "accuracy" not in result
