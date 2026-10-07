"""Evaluator-owned ctypes layouts and checked scalar boundary conversions."""

from __future__ import annotations

import ctypes
import struct
from types import SimpleNamespace
from typing import Mapping

from .errors import PrecisionResolutionError
from .precision import PrecisionProfile


# Event-time fields mirror lc_time_t in the public header. Other numeric
# fields are lc_real_t, except the explicitly classified profiling statistic.
_TIME_FIELDS = {
    "_CModel": {"refractory"},
    "_CState": {"t_last"},
    "_CEdge": {"delay"},
    "_CInputSpike": {"t"},
    "_CDriveUpdate": {"t"},
    "_COutputSpike": {"t"},
    "_CEncoderSpec": {"latency_min", "latency_max", "duration"},
    "_CEncoderState": {"last_end"},
    "_CPresentation": {"t_start", "t_end"},
    "_CEncodedSpike": {"t"},
    "_CEncodedDrive": {"t"},
    "_CDecoderSpec": {"width", "origin"},
    "_CDecodeWindow": {"t_start", "t_end"},
    "_CDecoderWindowBinding": {"t_start", "t_end"},
    "_CDecoderQueryBinding": {"t"},
    "_CDecodeResult": {"first_spike", "window_start", "window_end"},
    "_CDecodedEvent": {
        "emitted_at", "source_spike_time", "window_start", "window_end",
        "observed_through", "first_spike",
    },
    "_CRunConfig": {"t_end"},
    "_CRootHint": {"fastest_time_constant"},
    "_CTwoExpHint": {"fastest_time_constant"},
    "_CMultiExpHint": {"fastest_time_constant"},
    "_CExpPolyHint": {"fastest_time_constant"},
    "_CRootResult": {"t_spike", "horizon", "bracket_low", "bracket_high", "tolerance"},
    "_CStepConfig": {"initial_step", "minimum_step", "maximum_step", "event_tolerance"},
    "_CStepResult": {"t_reached", "t_crossing", "last_step"},
    "_CHazardConfig": {"time_tolerance"},
    "_CMixedNode": {"refractory"},
    "_CMixedEdge": {"delay"},
    "_CModulationEvent": {"t"},
    "_CPlasticityState": {
        "t_pre_fast", "t_post_fast", "t_pre_slow", "t_post_slow",
        "t_eligibility_plus", "t_eligibility_minus",
    },
    "_CLearningObserverSnapshot": {"t_activity"},
    "_CMixedInputSpike": {"t"},
    "_CMixedDriveUpdate": {"t"},
    "_CNetworkError": {"t"},
    "_CTraceRecord": {"t"},
    "_CStateInspectionRequest": {"t"},
    "_CStateInspectionResult": {"t"},
}


class _ScalarArgument:
    """Validate before ctypes can silently narrow a scalar function argument."""

    def __init__(self, scalar_type, convert):
        self.scalar_type = scalar_type
        self.convert = convert

    def from_param(self, value):
        if isinstance(value, (ctypes.c_float, ctypes.c_double)):
            value = value.value
        return self.scalar_type(self.convert(value))


def build_native_types(
    profile: PrecisionProfile, legacy: Mapping[str, type],
) -> SimpleNamespace:
    """Build one namespace without changing any legacy class or module global.

    Call only after the integer precision handshake. The caller caches these
    immutable-by-convention class definitions by complete precision identity.
    """

    scalar_types = {16: ctypes.c_uint16, 32: ctypes.c_float, 64: ctypes.c_double}
    real_type = scalar_types[profile.real_bits]
    time_type = scalar_types[profile.time_bits]
    profile_type = real_type if profile.time_bits == profile.real_bits else time_type
    half = profile is PrecisionProfile.FLOAT16

    def convert(value, *, time=False, name="native input"):
        if profile is PrecisionProfile.FLOAT64:
            return value
        try:
            round_value = profile.round_time if time else profile.round_real
            return round_value(value, name=name)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PrecisionResolutionError(str(exc)) from exc

    def real_value(value=0.0):
        return HalfValue(value) if half else real_type(convert(value))

    def time_value(value=0.0):
        return HalfValue(value) if half else time_type(convert(value, time=True))

    def half_bits(value):
        if isinstance(value, HalfValue):
            value = value.value
        return struct.unpack("=H", struct.pack("=e", convert(value)))[0]

    def half_value(bits):
        return struct.unpack("=e", struct.pack("=H", bits))[0]

    class HalfValue(ctypes.c_uint16):
        """Host-only view of native half bits, never a native half call ABI."""

        def __init__(self, value=0.0):
            ctypes.c_uint16.__init__(self, half_bits(value))

        @property
        def value(self):
            return half_value(ctypes.c_uint16.value.__get__(self))

        @value.setter
        def value(self, value):
            ctypes.c_uint16.value.__set__(self, half_bits(value))

    half_arrays = {}

    def half_array(count):
        if count not in half_arrays:
            class HalfArray(ctypes.Array):
                _type_ = ctypes.c_uint16
                _length_ = count

                def __init__(self, *values):
                    if len(values) > count:
                        raise IndexError("too many initializers")
                    for index, value in enumerate(values):
                        self[index] = value

                def __getitem__(self, key):
                    value = ctypes.Array.__getitem__(self, key)
                    return [half_value(item) for item in value] if isinstance(key, slice) else half_value(value)

                def __setitem__(self, key, value):
                    encoded = [half_bits(item) for item in value] if isinstance(key, slice) else half_bits(value)
                    ctypes.Array.__setitem__(self, key, encoded)

            half_arrays[count] = HalfArray
        return half_arrays[count]

    def array_factory(scalar_type, time):
        def sized(count):
            if half:
                return half_array(count)
            array_type = scalar_type * count

            def make(*values):
                return array_type(*(
                    convert(value, time=time, name=f"native array[{index}]")
                    for index, value in enumerate(values)
                ))

            return make

        return sized

    namespace = SimpleNamespace(
        real_type=real_type,
        time_type=time_type,
        profile_type=profile_type,
        real_value=real_value,
        time_value=time_value,
        real_array=array_factory(real_type, False),
        time_array=array_factory(time_type, True),
        real_argument=(ctypes.c_double if profile is PrecisionProfile.FLOAT64 else
                       _ScalarArgument(real_type, half_bits if half else convert)),
        time_argument=(ctypes.c_double if profile is PrecisionProfile.FLOAT64 else
                       _ScalarArgument(time_type, half_bits if half else lambda value: convert(value, time=True))),
    )
    if profile is PrecisionProfile.FLOAT64:
        vars(namespace).update(legacy)
        return namespace

    rebuilt = {}

    def clone(ctype, *, role="real"):
        if ctype is ctypes.c_double:
            return {"real": real_type, "time": time_type, "profile": profile_type}[role]
        if ctype in rebuilt:
            return rebuilt[ctype]
        if issubclass(ctype, ctypes.Array):
            if half and ctype._type_ is ctypes.c_double:
                return half_array(ctype._length_)
            return clone(ctype._type_, role=role) * ctype._length_
        if issubclass(ctype, ctypes._Pointer):
            return ctypes.POINTER(clone(ctype._type_, role=role))
        if issubclass(ctype, ctypes._CFuncPtr):
            result = ctypes.CFUNCTYPE(
                ctype._restype_, *(clone(arg) for arg in ctype._argtypes_),
            )
            rebuilt[ctype] = result
            return result
        if not issubclass(ctype, ctypes.Structure):
            return ctype
        name = ctype.__name__
        time_fields = _TIME_FIELDS.get(name, set())
        field_roles = {
            field_name: ("time" if field_name in time_fields else
                         "profile" if name == "_CRunStats" and field_name == "kernel_seconds"
                         else "real")
            for field_name, _ in ctype._fields_
        }
        numeric_fields = {
            field_name for field_name, field_type in ctype._fields_
            if field_type is ctypes.c_double
        }
        field_names = [field_name for field_name, _ in ctype._fields_]

        def checked(value, field_name):
            if field_name not in numeric_fields:
                return value
            converted = convert(value, time=field_roles[field_name] in ("time", "profile"),
                                name=f"{name}.{field_name}")
            return half_bits(converted) if half else converted

        def initialize(self, *args, **kwargs):
            if half:
                if len(args) > len(field_names):
                    raise TypeError("too many initializers")
                if set(field_names[:len(args)]) & kwargs.keys():
                    raise TypeError("duplicate field initializer")
                ctypes.Structure.__init__(self)
                for field_name, value in (*zip(field_names, args), *kwargs.items()):
                    ctypes.Structure.__setattr__(self, field_name, checked(value, field_name))
                return
            ctypes.Structure.__init__(
                self,
                *(checked(value, field_names[index]) for index, value in enumerate(args)),
                **{key: checked(value, key) for key, value in kwargs.items()},
            )

        def assign(self, field_name, value):
            ctypes.Structure.__setattr__(self, field_name, checked(value, field_name))

        def retrieve(self, field_name):
            value = ctypes.Structure.__getattribute__(self, field_name)
            return half_value(value) if half and field_name in numeric_fields else value

        result = type(name, (ctypes.Structure,), {
            "__module__": __name__, "__init__": initialize, "__setattr__": assign,
            "__getattribute__": retrieve,
        })
        rebuilt[ctype] = result
        result._fields_ = [
            (field_name, clone(field_type, role=field_roles[field_name]))
            for field_name, field_type in ctype._fields_
        ]
        return result

    vars(namespace).update({name: clone(ctype) for name, ctype in legacy.items()})
    return namespace
