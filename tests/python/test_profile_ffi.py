"""Profile-owned layouts, checked boundaries, and native profile interleaving."""

import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from lacuna import (
    AugmentedState, CoreEvaluator, DecoderBinding, ExprDAG, ExprNode, ExprOp, LIF,
    MixedDriveUpdate, PrecisionProfile, PrecisionResolutionError, Presentation, RegularRateEncoder,
    ScalarState, Spike, TTFSDecoder, resolve_scalar_lif,
)
from lacuna.dsl import parse_neuron
from lacuna.errors import CapabilityError, CoreError
from lacuna.ir import ParameterDomain
from lacuna import ffi


@pytest.fixture(params=tuple(PrecisionProfile), ids=lambda item: item.value)
def profile_core(request):
    profile = request.param
    folder = "build" if profile is PrecisionProfile.FLOAT64 else "build-" + profile.value
    candidates = sorted(Path(folder).glob("liblacuna_core.*"))
    if not candidates:
        pytest.skip(f"native {profile.value} build is not available")
    return CoreEvaluator(candidates[0], precision=profile)


def _rounding_dag(large=2.0 ** 24):
    return ExprDAG((
        ExprNode(ExprOp.CONST, value=large),
        ExprNode(ExprOp.CONST, value=1.0),
        ExprNode(ExprOp.ADD, lhs=0, rhs=1),
        ExprNode(ExprOp.SUB, lhs=2, rhs=0),
    ), (), (), {"sum": 2, "difference": 3})


def test_float64_classes_and_public_records_are_not_replaced():
    original = ffi._CExprNode
    wide = ffi._native_types(PrecisionProfile.FLOAT64)
    strict = ffi._native_types(PrecisionProfile.FLOAT32)
    mixed = ffi._native_types(PrecisionProfile.FLOAT32_TIME64)
    assert wide._CExprNode is original is ffi._CExprNode
    assert wide._CTraceConsumer is ffi._CTraceConsumer
    assert strict._CExprNode is not mixed._CExprNode
    assert strict._CExprNode is not original
    assert ffi._native_types(PrecisionProfile.FLOAT32) is strict
    assert ScalarState is ffi.ScalarState
    assert AugmentedState is ffi.AugmentedState
    assert wide.real_type is wide.time_type is ctypes.c_double
    assert strict.real_type is strict.time_type is ctypes.c_float
    assert mixed.real_type is ctypes.c_float
    assert mixed.time_type is ctypes.c_double
    assert strict._CMixedNode.program_nodes.offset != wide._CMixedNode.program_nodes.offset
    assert strict._CTraceConsumer._argtypes_[0]._type_ is strict._CTraceRecord
    assert mixed._CTraceConsumer._argtypes_[0]._type_ is mixed._CTraceRecord


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_every_ctypes_structure_matches_the_selected_public_c_header(profile, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the public-layout probe")
    header = Path("c/include/lacuna.h").resolve()
    contents = re.sub(r"/\*.*?\*/", "", header.read_text(), flags=re.S)
    declarations = {}
    for name, body in re.findall(
        r"typedef struct (\w+)\s*\{(.*?)\}\s*\w+\s*;", contents, re.S,
    ):
        names = tuple(re.findall(r"\b(\w+)\s*(?:\[[^]]+\])?\s*;", body))
        declarations.setdefault(names, []).append(name)
    aliases = {
        "_COutputSpike": "lc_output_spike",
        "_CStateInspectionRequest": "lc_state_inspection_request",
        "_CEncodedSpike": "lc_encoded_spike",
        "_CEncodedDrive": "lc_encoded_drive",
    }
    types = ffi._native_types(profile)
    lines = ['#include "lacuna.h"', '#include <stddef.h>', '#include <stdio.h>',
             'int main(void) {']
    expected = []
    for name, kind in vars(types).items():
        if not isinstance(kind, type) or not issubclass(kind, ctypes.Structure):
            continue
        field_names = tuple(item[0] for item in kind._fields_)
        candidates = declarations[field_names]
        c_name = aliases.get(name, candidates[0])
        assert c_name in candidates
        lines.append(f'printf("%zu\\n", sizeof({c_name}));')
        expected.append(ctypes.sizeof(kind))
        for field_name in field_names:
            lines.append(f'printf("%zu\\n", offsetof({c_name}, {field_name}));')
            expected.append(getattr(kind, field_name).offset)
    lines.extend(['return 0;', '}'])
    source = tmp_path / "layout_probe.c"
    executable = tmp_path / "layout_probe"
    source.write_text("\n".join(lines))
    subprocess.run([
        compiler, "-std=c99", "-I", str(header.parent),
        f"-DLACUNA_REAL_BITS={profile.real_bits}",
        f"-DLACUNA_TIME_BITS={profile.time_bits}",
        str(source), "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert [int(item) for item in result.stdout.splitlines()] == expected


@pytest.mark.parametrize("profile", (PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT32_TIME64))
@pytest.mark.parametrize("value", (1e39, 1e-50, float("nan"), float("inf")))
def test_reduced_structure_array_and_function_inputs_fail_before_narrowing(profile, value):
    types = ffi._native_types(profile)
    with pytest.raises(PrecisionResolutionError):
        types._CInputSpike(0.0, 0, value)
    with pytest.raises(PrecisionResolutionError):
        types.real_array(1)(value)
    with pytest.raises(PrecisionResolutionError):
        types.real_argument.from_param(value)
    record = types._CInputSpike()
    with pytest.raises(PrecisionResolutionError):
        record.value = value


def test_mixed_clock_fields_are_not_accidentally_model_width():
    types = ffi._native_types(PrecisionProfile.FLOAT32_TIME64)
    event = types._CInputSpike(2.0 ** 24 + 1.0, 0, 0.1)
    assert event.t == 2.0 ** 24 + 1.0
    assert event.value == PrecisionProfile.FLOAT32.round_real(0.1)
    assert dict(types._CRunStats._fields_)["kernel_seconds"] is ctypes.c_double
    telemetry = types._CRunStats()
    telemetry.kernel_seconds = 2.0 ** 24 + 1.0
    assert telemetry.kernel_seconds == 2.0 ** 24 + 1.0
    strict = ffi._native_types(PrecisionProfile.FLOAT32)
    assert dict(strict._CRunStats._fields_)["kernel_seconds"] is ctypes.c_float


class _IntegerFunction:
    def __init__(self, result):
        self.result = result

    def __call__(self, *args):
        return self.result(*args) if callable(self.result) else self.result


class _MetadataLibrary:
    def __init__(self, profile, *, abi=None, properties=None):
        values = {
            1: 1, 2: profile.native_id, 3: profile.real_bits, 4: profile.time_bits,
            5: 53 if profile.real_bits == 64 else 24,
            6: 53 if profile.time_bits == 64 else 24,
            7: 1024 if profile.real_bits == 64 else 128,
            8: 1024 if profile.time_bits == 64 else 128,
            9: 32 if profile is PrecisionProfile.FLOAT32 else 64,
            10: 24 if profile is PrecisionProfile.FLOAT32 else 53,
            11: 128 if profile is PrecisionProfile.FLOAT32 else 1024,
            12: 2, 13: 1,
        }
        values.update(properties or {})
        self.lc_abi_version = _IntegerFunction(abi if abi is not None else (
            17 if profile is PrecisionProfile.FLOAT64 else 18
        ))
        self.lc_sizeof_network_error = _IntegerFunction(0)
        self.lc_numeric_property = _IntegerFunction(lambda key: values.get(key, 0))
        self.lc_numeric_profile_check = _IntegerFunction(0)

    def __getattr__(self, name):
        raise AssertionError(f"precision-bearing function was accessed: {name}")


@pytest.mark.parametrize("properties", ({2: 1}, {3: 64}, {9: 64}, {10: 53}, {13: 2}))
def test_invalid_native_metadata_fails_before_any_profile_layout(monkeypatch, properties):
    fake = _MetadataLibrary(PrecisionProfile.FLOAT32, properties=properties)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    monkeypatch.setattr(ffi, "_native_types", lambda profile: pytest.fail("layout built early"))
    with pytest.raises(CoreError, match="precision mismatch|wide arithmetic"):
        CoreEvaluator("unused", precision="float32")


def test_reduced_profile_requires_abi18_before_layout_construction(monkeypatch):
    fake = _MetadataLibrary(PrecisionProfile.FLOAT32, abi=17)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake)
    monkeypatch.setattr(ffi, "_native_types", lambda profile: pytest.fail("layout built early"))
    with pytest.raises(CoreError, match="ABI mismatch"):
        CoreEvaluator("unused", precision="float32")


def test_interleaved_native_profiles_keep_their_own_types_and_results(profile_core, core):
    selected = profile_core.precision
    dag = _rounding_dag(2.0 ** 11 if selected is PrecisionProfile.FLOAT16 else 2.0 ** 24)
    for _ in range(3):
        assert core.evaluate_expr(dag)["difference"] == 1.0
        assert profile_core.evaluate_expr(dag)["difference"] == (
            1.0 if selected is PrecisionProfile.FLOAT64 else 0.0
        )
        assert core._types._CExprNode is ffi._CExprNode
    assert profile_core.precision_info.real_bits == selected.real_bits
    assert profile_core.precision_info.time_bits == selected.time_bits
    assert profile_core.abi_version == (19 if selected is PrecisionProfile.FLOAT16 else
                                        17 if selected is PrecisionProfile.FLOAT64 else 18)


def test_bound_scalar_and_buffer_argument_roles_match_the_c_header(profile_core):
    header = re.sub(r"/\*.*?\*/", "", Path("c/include/lacuna.h").read_text(), flags=re.S)
    prototypes = re.findall(
        r"LC_API\s+[^;()]+?\s+(lc_\w+)\s*\((.*?)\)\s*;", header, re.S,
    )
    checked = 0
    for name, raw_parameters in prototypes:
        function = getattr(profile_core._lib, name, None)
        if function is None or function.argtypes is None:
            continue
        parameters = [item.strip() for item in raw_parameters.split(",")]
        if not any("lc_real_t" in item or "lc_time_t" in item for item in parameters):
            continue
        assert len(function.argtypes) == len(parameters)
        for parameter, argument in zip(parameters, function.argtypes):
            if "lc_real_t" not in parameter and "lc_time_t" not in parameter:
                continue
            expected = (profile_core._types.time_type if "lc_time_t" in parameter
                        else profile_core._types.real_type)
            if "*" in parameter:
                assert argument._type_ is expected
            else:
                assert getattr(argument, "scalar_type", argument) is expected
            checked += 1
    assert checked > 50


def test_runtime_parameter_updates_validate_values_after_target_rounding(profile_core):
    layout = ffi._RuntimeNodeLayout(1, (
        ffi._RuntimeDriveBinding(("drive",), 0, ParameterDomain.POSITIVE),
    ))
    event = MixedDriveUpdate(1.0, 0, 0.1, "drive")
    packed = profile_core._pack_mixed_drives((layout,), (event,))
    assert packed[0].value == profile_core.precision.round_real(0.1)
    if profile_core.precision.real_bits == 32:
        for value in (1e-50, 1e39):
            with pytest.raises(PrecisionResolutionError):
                profile_core._pack_mixed_drives((layout,), (
                    MixedDriveUpdate(1.0, 0, value, "drive"),
                ))


def test_selected_native_roots_skip_unrelated_invalid_dynamic_closures(profile_core):
    dag = ExprDAG((
        ExprNode(ExprOp.CONST, value=2.0),
        ExprNode(ExprOp.VAR),
        ExprNode(ExprOp.LOG, lhs=1),
        ExprNode(ExprOp.DIV, lhs=0, rhs=1),
        ExprNode(ExprOp.NEG, lhs=0),
    ), (), ("state",), {"constant": 0, "log": 2, "division": 3, "negative": 4})
    with pytest.raises(CoreError):
        profile_core.evaluate_expr(dag, variables={"state": 0.0})
    result = profile_core.evaluate_expr(
        dag, variables={"state": 0.0}, roots=("negative", "constant"),
    )
    assert list(result) == ["negative", "constant"]
    assert result == {"negative": -2.0, "constant": 2.0}
    assert profile_core.evaluate_expr(dag, variables={"state": 0.0}, roots=()) == {}


@pytest.mark.parametrize("roots,error", (
    ("sum", TypeError), (("sum", "sum"), ValueError),
    (("missing",), ValueError), ((1,), TypeError),
))
def test_invalid_selected_root_names_fail_without_native_evaluation(core, roots, error):
    with pytest.raises(error):
        core.evaluate_expr(_rounding_dag(), roots=roots)


def test_selected_inspection_matches_native_full_state_advance(profile_core):
    model = resolve_scalar_lif(parse_neuron(LIF().model.source))
    state = AugmentedState((1.0,), 0.1)
    target_time = 1.7
    full = profile_core.advance_analytical(model, state, target_time)
    selected = profile_core.advance_analytical_selected(model, state, target_time, (0,))
    assert selected == full.values


def test_selected_inspection_clock_progress_follows_the_profile(profile_core):
    model = resolve_scalar_lif(parse_neuron(LIF().model.source))
    origin = 2.0 ** 11 if profile_core.precision is PrecisionProfile.FLOAT16 else 2.0 ** 24
    def inspect():
        return profile_core.advance_analytical_selected(
            model, AugmentedState((1.0,), origin), origin + 1.0, (0,),
        )
    if profile_core.precision in (PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT16):
        with pytest.raises(PrecisionResolutionError, match="does not advance time"):
            inspect()
    else:
        assert len(inspect()) == 1


def test_native_codec_sessions_use_the_evaluators_layouts(profile_core):
    encoders = (RegularRateEncoder(1.0, 1.0),)
    presentations = (Presentation(0.0, 3.0, 0, 1.0),)
    expected = profile_core.encode_presentations(encoders, presentations)
    assert tuple(item.t for item in expected.spikes) == (1.0, 2.0)
    session = profile_core.create_encoder_session(encoders)
    session.submit(presentations)
    assert session.finish() == expected
    with profile_core.create_streaming_encoder_run(encoders) as run:
        first = run.advance_until(2.0, presentations=presentations)
        final = run.finish(3.0)
    assert first.spikes + final.spikes == expected.spikes
    spikes = tuple(Spike(item.t, 0) for item in expected.spikes)
    decoders = (DecoderBinding(0, TTFSDecoder()),)
    decoded = profile_core.decode_spikes(decoders, spikes, t_start=0.0, t_end=3.0)
    assert decoded[0].value == 1.0
    with profile_core.compile_decoders(decoders, node_count=1) as compiled:
        with compiled.create_run(t_start=0.0, t_end=3.0) as run:
            run.consume(spikes)
            run.advance(3.0)
            assert run.finalize() == decoded


def test_untagged_resolved_models_cannot_be_compiled_for_reduced_precision(profile_core):
    if profile_core.precision is PrecisionProfile.FLOAT64:
        return
    model = resolve_scalar_lif(parse_neuron(LIF().model.source))
    with pytest.raises(CapabilityError, match="target-tagged ExecutionPlan"):
        profile_core.compile_mixed((model,))
    with pytest.raises(CapabilityError, match="target-tagged ExecutionPlan"):
        profile_core.run_mixed((model,), (0.0,), t_end=1.0)
    with pytest.raises(CapabilityError, match="target-tagged ExecutionPlan"):
        profile_core.run_delta((model,), (0.0,), t_end=1.0)
