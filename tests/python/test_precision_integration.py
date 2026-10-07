"""Keep precision checks ahead of native numerical calls and graph allocation."""

import ctypes
from dataclasses import FrozenInstanceError, replace

import pytest

from lacuna import CoreEvaluator, Engine, LIF, NetworkBuilder, PrecisionProfile
from lacuna.errors import CapabilityError, CoreError
from lacuna.execution_plan import lower_execution_plan
from lacuna.ffi import _CNetworkError


class _Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _MetadataLibrary:
    def __init__(self, *, overrides=None, missing=(), handshake=0, abi=17):
        self.values = dict(enumerate(
            (1, 1, 64, 64, 53, 53, 1024, 1024, 64, 53, 1024, 2, 1), 1
        ))
        self.values.update(overrides or {})
        self.missing = missing
        self.calls = []
        self.bindings = []
        self.lc_abi_version = _Function(lambda: abi)
        self.lc_sizeof_network_error = _Function(lambda: ctypes.sizeof(_CNetworkError))
        self.numeric_query = _Function(self.query)
        self.numeric_check = _Function(lambda *args: self.check(args, handshake))

    def query(self, field):
        self.calls.append(("query", field))
        return self.values.get(field, 0)

    def check(self, args, status):
        self.calls.append(("check", args))
        return status

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        if name == "lc_numeric_property":
            return self.numeric_query
        if name == "lc_numeric_profile_check":
            return self.numeric_check
        self.bindings.append(name)
        return _Function(lambda *args: pytest.fail("unexpected numerical call"))


def _load_fake(monkeypatch, **kwargs):
    lib = _MetadataLibrary(**kwargs)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: lib)
    return lib


@pytest.mark.parametrize("field,bad_value", [
    (1, 2), (2, 0), (2, 2), (2, 3), (3, 32), (4, 32), (5, 24),
    (6, 24), (7, 128), (8, 128), (9, 0), (10, 0), (10, 1000),
    (11, 0), (12, 10), (13, 2),
])
def test_incompatible_metadata_rejected_before_numerical_binding(
    monkeypatch, field, bad_value
):
    lib = _load_fake(monkeypatch, overrides={field: bad_value})
    with pytest.raises(CoreError, match="precision mismatch|invalid wide-arithmetic"):
        CoreEvaluator("metadata-only")
    assert not lib.bindings
    assert all(kind == "query" for kind, _ in lib.calls)


@pytest.mark.parametrize("missing", [
    ("lc_numeric_property",), ("lc_numeric_profile_check",),
])
def test_partial_metadata_is_not_mistaken_for_a_legacy_library(monkeypatch, missing):
    lib = _load_fake(monkeypatch, missing=missing)
    with pytest.raises(CoreError, match="incomplete numeric metadata"):
        CoreEvaluator("metadata-only")
    assert not lib.calls
    assert not lib.bindings


def test_native_handshake_must_accept_profile_before_numerical_binding(monkeypatch):
    lib = _load_fake(monkeypatch, handshake=2)
    with pytest.raises(CoreError, match="rejected the requested precision contract"):
        CoreEvaluator("metadata-only")
    assert lib.calls[-1] == ("check", (1, 64, 64, 1))
    assert not lib.bindings


def test_abi_validation_still_precedes_precision_metadata(monkeypatch):
    lib = _load_fake(monkeypatch, abi=16)
    with pytest.raises(CoreError, match="ABI mismatch"):
        CoreEvaluator("metadata-only")
    assert not lib.calls
    assert not lib.bindings


def test_legacy_abi17_precision_is_inferred_and_labeled(monkeypatch):
    lib = _load_fake(monkeypatch, missing=(
        "lc_numeric_property", "lc_numeric_profile_check"
    ))
    core = CoreEvaluator("legacy")
    assert core.precision is PrecisionProfile.FLOAT64
    assert core.precision_info.metadata_source == "legacy-abi17"
    assert core.precision_info.metadata_version is None
    assert core.precision_info.wide_bits is None
    assert not lib.calls
    assert lib.bindings


def test_native_metadata_is_integer_only_and_read_only(monkeypatch):
    lib = _load_fake(monkeypatch)
    core = CoreEvaluator("metadata-only", precision="float64")
    assert core.precision_info.metadata_source == "native"
    assert core.precision_info.metadata_version == 1
    assert core.precision_info.arithmetic_revision == 1
    assert lib.numeric_query.argtypes == [ctypes.c_uint32]
    assert lib.numeric_query.restype is ctypes.c_uint32
    assert lib.numeric_check.argtypes == [ctypes.c_uint32] * 4
    assert lib.calls[-1] == ("check", (1, 64, 64, 1))
    with pytest.raises(AttributeError):
        core.precision = PrecisionProfile.FLOAT32
    with pytest.raises(AttributeError):
        core.precision_info = None
    with pytest.raises(FrozenInstanceError):
        core.precision_info.profile = PrecisionProfile.FLOAT32


@pytest.mark.parametrize("factory", [CoreEvaluator, Engine])
@pytest.mark.parametrize("settings", [
    {"precision": "float32"},
    {"precision": "float32", "time_precision": "float64"},
    {"precision": "float32-time64"},
    {"precision": PrecisionProfile.FLOAT32},
])
def test_reduced_precision_rejects_incompatible_library_before_binding(monkeypatch, factory, settings):
    lib = _load_fake(monkeypatch)
    with pytest.raises(CoreError, match="ABI mismatch"):
        factory("metadata-only", **settings)
    assert not lib.bindings
    assert not lib.calls


def _network():
    builder = NetworkBuilder("precision-integration")
    neuron = builder.neuron("tonic", LIF(drive=24.0))
    builder.output("spikes", neuron)
    return builder.build()


def test_native_profile_engine_plan_and_image_roundtrip(core):
    engine = Engine(core._lib._name, precision="float64", time_precision="float64")
    assert core.precision_info.metadata_source == "native"
    assert core.precision_info.real_bits == core.precision_info.time_bits == 64
    assert core.precision_info.real_mant_dig == core.precision_info.time_mant_dig == 53
    assert engine.precision is PrecisionProfile.FLOAT64
    network = _network()
    with engine.compile(network) as compiled:
        assert compiled.precision is engine.precision
        assert compiled.execution_plan.precision is engine.precision
        assert compiled.execution_plan.precision_key == ("float64", 64, 64, 1)
        result = compiled.run(100.0)
        assert result.spikes
        payload = compiled.compiled_graph_image()
        with core.load_compiled_graph_image(
            payload, execution_plan=compiled.execution_plan
        ) as loaded:
            assert loaded.precision is engine.precision
            assert loaded.to_bytes() == payload
    with Engine(core._lib._name).compile(network) as default_compiled:
        assert default_compiled.run(100.0).spikes == result.spikes
        assert default_compiled.compiled_graph_image() == payload


@pytest.mark.parametrize("precision", ["float32", "float32-time64"])
def test_lowering_requires_a_resolved_graph(precision):
    with pytest.raises(TypeError, match="ResolvedGraph"):
        lower_execution_plan(None, precision=precision)


def test_lowered_plan_cannot_be_relabelled_or_loaded_at_another_precision(core):
    plan = _network().graph.resolve().execution_plan()
    assert replace(plan, precision="float64") == plan
    with pytest.raises(FrozenInstanceError):
        plan.precision = PrecisionProfile.FLOAT32
    with pytest.raises(ValueError, match="target-aware"):
        replace(plan, precision=PrecisionProfile.FLOAT32)
    # Native entry points validate even a caller that bypasses frozen fields.
    forged = replace(plan)
    object.__setattr__(forged, "precision", PrecisionProfile.FLOAT32)
    with pytest.raises(CoreError, match="precision"):
        core.compile_execution_plan(forged)
    with pytest.raises(CoreError, match="precision"):
        core.load_compiled_graph_image(b"", execution_plan=forged)
