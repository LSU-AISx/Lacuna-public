from __future__ import annotations

import json
import random
import string

import pytest

from lacuna import Graph, parse_neuron, parse_synapse
from lacuna.errors import DSLParseError, ResolutionError

from .test_alpha import ALPHA_SYNAPSE
from .test_dsl_resolver import LIF
from .test_graph import _graph


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("junk\n" + LIF, "expected 'neuron"),
        (LIF + "\nneuron Second {}", "only one neuron definition"),
        (LIF.replace("state {", "mystery { x }\nstate {"), "unknown 'mystery' block"),
        (LIF.replace("threshold {", "state { v : membrane }\nthreshold {"), "duplicate 'state' block"),
        (LIF.replace("drive = 20.0", "drive = nan"), "finite numeric literal"),
        (LIF.replace("drive = 20.0", "drive = 1e999999"), "finite numeric literal"),
        (
            LIF.replace(
                "dv/dt = -(v - v_rest)/tau_m + drive/tau_m",
                "dv/dt = 0\ndv/dt = -(v - v_rest)/tau_m + drive/tau_m",
            ),
            "duplicate differential equation",
        ),
        (
            LIF.replace("reset { v <- v_reset }", "reset { v <- -65\nv <- v_reset }"),
            "duplicate reset statement",
        ),
    ],
)
def test_neuron_parser_rejects_ambiguous_or_nonfinite_input(
    source: str, message: str
) -> None:
    with pytest.raises(DSLParseError, match=message):
        parse_neuron(source)


def test_parsers_reject_non_text_input_with_typed_errors() -> None:
    with pytest.raises(DSLParseError, match="must be text"):
        parse_neuron(None)  # type: ignore[arg-type]
    with pytest.raises(DSLParseError, match="must be text"):
        parse_synapse(7)  # type: ignore[arg-type]
    with pytest.raises(ResolutionError, match="must be text"):
        Graph.from_text(None)  # type: ignore[arg-type]


def _mutate(source: str, randomizer: random.Random) -> str:
    result = source
    alphabet = string.ascii_letters + string.digits + "{}<>/=:+-.# \n\t"
    for _ in range(randomizer.randint(1, 8)):
        operation = randomizer.randrange(3)
        position = randomizer.randrange(len(result) + 1)
        if operation == 0:
            result = result[:position] + randomizer.choice(alphabet) + result[position:]
        elif operation == 1 and result:
            position = min(position, len(result) - 1)
            result = result[:position] + result[position + 1 :]
        elif result:
            position = min(position, len(result) - 1)
            result = result[:position] + randomizer.choice(alphabet) + result[position + 1 :]
    return result


def test_dsl_mutation_campaign_has_only_typed_rejections_or_valid_models() -> None:
    randomizer = random.Random(0x1AC0A)
    for source in (LIF, ALPHA_SYNAPSE):
        parser = parse_neuron if source is LIF else parse_synapse
        for _ in range(500):
            candidate = _mutate(source, randomizer)
            try:
                model = parser(candidate)
            except DSLParseError:
                continue
            assert model.name


def _random_json_value(randomizer: random.Random, depth: int = 0):
    atoms = [None, True, False, randomizer.randint(-10, 10), randomizer.random(), "x"]
    if depth >= 3:
        return randomizer.choice(atoms)
    kind = randomizer.randrange(4)
    if kind == 0:
        return randomizer.choice(atoms)
    if kind == 1:
        return [_random_json_value(randomizer, depth + 1) for _ in range(randomizer.randrange(5))]
    return {
        randomizer.choice(("schema", "models", "nodes", "edges", "bindings", "x")):
        _random_json_value(randomizer, depth + 1)
        for _ in range(randomizer.randrange(5))
    }


def test_graph_document_fuzz_has_only_typed_rejections_or_canonical_graphs() -> None:
    randomizer = random.Random(0xC5A)
    for _ in range(750):
        candidate = json.dumps(_random_json_value(randomizer))
        try:
            graph = Graph.from_text(candidate)
        except ResolutionError:
            continue
        assert Graph.from_text(graph.to_text()) == graph


def test_graph_serialization_canonicalizes_collection_and_key_order() -> None:
    canonical = _graph().to_text()
    document = json.loads(canonical)
    randomizer = random.Random(73)
    for key in ("models", "nodes", "edges", "input_ports", "output_ports"):
        randomizer.shuffle(document[key])
    reordered = {key: document[key] for key in reversed(tuple(document))}
    assert Graph.from_text(json.dumps(reordered)).to_text() == canonical
