"""Small, strict parser for the milestone-0 intrinsic neuron DSL."""

from __future__ import annotations

import math
import re

from .errors import DSLParseError
from .resolution_cache import _cached_resolution
from .ir import (
    FixedRefractory,
    HazardDefinition,
    NeuronModel,
    ParameterDefinition,
    ParameterDomain,
    ReactiveMode,
    StateDefinition,
    StateRole,
    SynapseModel,
    ThresholdDefinition,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DECIMAL_NUMBER = re.compile(
    r"^[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))"
    r"(?:[eE][+-]?[0-9]+)?$"
)
_HEX_NUMBER = re.compile(
    r"^[+-]?0[xX](?:(?:[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?)|"
    r"(?:\.[0-9A-Fa-f]+))[pP][+-]?[0-9]+$"
)


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _balanced_body(text: str, open_brace: int) -> tuple[str, int]:
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index], index + 1
    raise DSLParseError("unclosed '{' block")


def _definition_blocks(
    text: str,
    keyword: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> tuple[str, dict[str, str]]:
    if not isinstance(text, str):
        raise DSLParseError(f"{keyword} definition must be text")
    source = _strip_comments(text)
    header = re.match(
        rf"\s*{re.escape(keyword)}\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{{",
        source,
    )
    if not header:
        raise DSLParseError(f"expected '{keyword} <name> {{ ... }}'")
    open_brace = source.find("{", header.start())
    body, end = _balanced_body(source, open_brace)
    if source[end:].strip():
        raise DSLParseError(
            f"only one {keyword} definition is accepted by parse_{keyword}"
        )

    allowed = required | optional
    blocks: dict[str, str] = {}
    position = 0
    while position < len(body):
        while position < len(body) and body[position].isspace():
            position += 1
        if position == len(body):
            break
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", body[position:])
        if not name_match:
            raise DSLParseError(
                f"expected a block name near '{body[position:position + 24].strip()}'"
            )
        name = name_match.group(0)
        position += len(name)
        while position < len(body) and body[position].isspace():
            position += 1
        if position == len(body) or body[position] != "{":
            raise DSLParseError(f"expected '{{' after '{name}' block name")
        value, position = _balanced_body(body, position)
        if name not in allowed:
            raise DSLParseError(f"unknown '{name}' block in {keyword} definition")
        if name in blocks:
            raise DSLParseError(f"duplicate '{name}' block in {keyword} definition")
        blocks[name] = value.strip()
    missing = sorted(required - blocks.keys())
    if missing:
        raise DSLParseError(f"missing required '{missing[0]}' block")
    return header.group(1), blocks


def _statements(body: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;\n]+", body) if part.strip()]


def _number(text: str, context: str) -> float:
    value = text.strip()
    if not (_DECIMAL_NUMBER.fullmatch(value) or _HEX_NUMBER.fullmatch(value)):
        raise DSLParseError(
            f"{context} must be a finite numeric literal, got '{value}'"
        )
    try:
        result = (
            float.fromhex(value)
            if value.lower().lstrip("+-").startswith("0x")
            else float(value)
        )
    except (OverflowError, ValueError) as exc:
        raise DSLParseError(f"{context} must be a numeric literal, got '{value}'") from exc
    if not math.isfinite(result):
        raise DSLParseError(f"{context} must be a finite numeric literal")
    return result


@_cached_resolution
def parse_neuron(text: str) -> NeuronModel:
    """Parse one intrinsic neuron definition.

    This parser deliberately covers the scalar-LIF milestone only. Unsupported DSL
    syntax fails clearly instead of being guessed or silently ignored.
    """

    definition_name, blocks = _definition_blocks(
        text,
        "neuron",
        frozenset({"params", "state", "dynamics", "reset"}),
        frozenset({"threshold", "hazard", "refractory", "reactive"}),
    )

    parameters: list[ParameterDefinition] = []
    for statement in _statements(blocks["params"]):
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(finite|positive))?\s*=\s*(.+)",
            statement,
        )
        if not match:
            raise DSLParseError(f"invalid parameter declaration: '{statement}'")
        name, domain_text, value_text = match.groups()
        domain = ParameterDomain(domain_text or "finite")
        parameters.append(ParameterDefinition(name, _number(value_text, name), domain))

    states: list[StateDefinition] = []
    for statement in _statements(blocks["state"]):
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(membrane|receptor|adaptation|observer|aux)",
            statement,
        )
        if not match:
            raise DSLParseError(f"invalid state declaration: '{statement}'")
        states.append(StateDefinition(match.group(1), StateRole(match.group(2))))

    dynamics: dict[str, str] = {}
    for statement in _statements(blocks["dynamics"]):
        match = re.fullmatch(r"d([A-Za-z_][A-Za-z0-9_]*)/dt\s*=\s*(.+)", statement)
        if not match:
            raise DSLParseError(f"invalid differential equation: '{statement}'")
        if match.group(1) in dynamics:
            raise DSLParseError(f"duplicate differential equation for '{match.group(1)}'")
        dynamics[match.group(1)] = match.group(2).strip()

    if ("threshold" in blocks) == ("hazard" in blocks):
        raise DSLParseError(
            "neuron definition must contain exactly one threshold or hazard block"
        )
    threshold = None
    if "threshold" in blocks:
        threshold_statements = _statements(blocks["threshold"])
        if len(threshold_statements) != 1:
            raise DSLParseError("threshold block must contain exactly one comparison")
        threshold_match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*>\s*(.+)", threshold_statements[0]
        )
        if not threshold_match:
            raise DSLParseError(
                "slice 1 supports only rising thresholds written as "
                "'<readout> > <level>'"
            )
        threshold = ThresholdDefinition(
            threshold_match.group(1), threshold_match.group(2).strip()
        )
    hazard = None
    if "hazard" in blocks:
        hazard_statements = _statements(blocks["hazard"])
        if len(hazard_statements) != 1:
            raise DSLParseError("hazard block must contain exactly one rate equation")
        hazard_match = re.fullmatch(r"rate\s*=\s*(.+)", hazard_statements[0])
        if not hazard_match:
            raise DSLParseError("hazard must be written as 'rate = <expression>'")
        hazard = HazardDefinition(hazard_match.group(1).strip())

    reset: dict[str, str] = {}
    for statement in _statements(blocks["reset"]):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*(.+)", statement)
        if not match:
            raise DSLParseError(f"invalid reset statement: '{statement}'")
        if match.group(1) in reset:
            raise DSLParseError(f"duplicate reset statement for '{match.group(1)}'")
        reset[match.group(1)] = match.group(2).strip()

    refractory_body = blocks.get("refractory")
    refractory = None
    if refractory_body is not None:
        value = refractory_body.strip()
        if value.startswith("duration"):
            match = re.fullmatch(r"duration\s*=\s*(.+)", value)
            if not match:
                raise DSLParseError("invalid refractory duration")
            value = match.group(1)
        refractory = FixedRefractory(_number(value, "refractory duration"))

    reactive_body = blocks.get("reactive")
    reactive = None
    if reactive_body is not None:
        statements = _statements(reactive_body)
        if len(statements) != 1:
            raise DSLParseError("reactive block must contain exactly one mode")
        match = re.fullmatch(
            r"mode\s*=\s*(HOLD|RESET_BEFORE_DEPOSIT)", statements[0]
        )
        if not match:
            raise DSLParseError(
                "reactive mode must be HOLD or RESET_BEFORE_DEPOSIT"
            )
        reactive = ReactiveMode(match.group(1))

    names = [parameter.name for parameter in parameters] + [state.name for state in states]
    if len(names) != len(set(names)):
        raise DSLParseError("parameter and state names must be unique")
    if not all(_IDENTIFIER.fullmatch(name) for name in names):
        raise DSLParseError("invalid identifier")

    return NeuronModel(
        name=definition_name,
        parameters=tuple(parameters),
        states=tuple(states),
        dynamics=dynamics,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        reactive=reactive,
        hazard=hazard,
    )


@_cached_resolution
def parse_synapse(text: str) -> SynapseModel:
    """Parse one strict synapse definition for structural resolver classification."""

    definition_name, blocks = _definition_blocks(
        text,
        "synapse",
        frozenset({"params", "state", "dynamics", "on_spike", "output"}),
    )

    parameters: list[ParameterDefinition] = []
    for statement in _statements(blocks["params"]):
        match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(finite|positive))?\s*=\s*(.+)",
            statement,
        )
        if not match:
            raise DSLParseError(f"invalid parameter declaration: '{statement}'")
        name, domain_text, value_text = match.groups()
        parameters.append(
            ParameterDefinition(
                name,
                _number(value_text, name),
                ParameterDomain(domain_text or "finite"),
            )
        )

    states = tuple(_statements(blocks["state"]))
    if not states or not all(_IDENTIFIER.fullmatch(name) for name in states):
        raise DSLParseError("synapse state declarations must be identifiers")

    dynamics: dict[str, str] = {}
    for statement in _statements(blocks["dynamics"]):
        match = re.fullmatch(r"d([A-Za-z_][A-Za-z0-9_]*)/dt\s*=\s*(.+)", statement)
        if not match:
            raise DSLParseError(f"invalid differential equation: '{statement}'")
        if match.group(1) in dynamics:
            raise DSLParseError(f"duplicate differential equation for '{match.group(1)}'")
        dynamics[match.group(1)] = match.group(2).strip()

    spike_statements = _statements(blocks["on_spike"])
    if len(spike_statements) != 1:
        raise DSLParseError("on_spike block must contain exactly one update")
    spike_match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*(.+)", spike_statements[0]
    )
    if not spike_match:
        raise DSLParseError("invalid on_spike update")

    outputs: dict[str, str] = {}
    for statement in _statements(blocks["output"]):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", statement)
        if not match:
            raise DSLParseError(f"invalid output declaration: '{statement}'")
        if match.group(1) in outputs:
            raise DSLParseError(f"duplicate output declaration: '{match.group(1)}'")
        outputs[match.group(1)] = match.group(2).strip()
    if not outputs:
        raise DSLParseError("output block must contain at least one declaration")

    names = [parameter.name for parameter in parameters] + list(states)
    if len(names) != len(set(names)):
        raise DSLParseError("parameter and state names must be unique")
    if "w" in names:
        raise DSLParseError("'w' is reserved for the incoming synaptic weight")

    return SynapseModel(
        name=definition_name,
        parameters=tuple(parameters),
        states=states,
        dynamics=dynamics,
        spike_target=spike_match.group(1),
        spike_update=spike_match.group(2).strip(),
        outputs=outputs,
    )
