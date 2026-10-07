"""Thin ctypes binding to Lacuna's sole C evaluator."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .codec import (
    BurstEncoder,
    DecodeEvent,
    DecodeEventKind,
    DecodeValue,
    DecodeWindow,
    DecoderBinding,
    DecoderQueryBinding,
    DecoderWindowBinding,
    EncodedBatch,
    EncodedDrive,
    EncodedSpike,
    Encoder,
    HeldCurrentEncoder,
    LatencyBurstEncoder,
    NativeEventEncoder,
    PoissonRateEncoder,
    Presentation,
    RateDecoder,
    RegularRateEncoder,
    TemporalSpikeMode,
    TemporalWeightDecoder,
    TTFSEncoder,
    TTFSDecoder,
)
from .errors import CapabilityError, CoreError, CoreFailureDiagnostic, PrecisionResolutionError
from .execution_plan import ArithmeticMethod, ExecutionPlan, ScalarAffineParameters
from .expr import ExprDAG, ExprNode, ExprOp
from .learning import LearningEvent
from .ir import (
    DispatchForm,
    ExpPolyRootHint,
    MultiExpRootHint,
    NeuronPolarity,
    NumericalConfig,
    ParameterDomain,
    ResolvedAdaptiveLIF,
    ResolvedAlphaLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
    RootFindHint,
    ScalarLogRootHint,
    TwoExpRootHint,
)
from .plasticity import (
    ModulatedSTDP,
    PairSTDP,
    PlasticityRule,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
)
from .precision import PrecisionProfile, normalize_precision
from .ffi_types import build_native_types

AnalyticalModel = (
    ResolvedScalarLIF
    | ResolvedAlphaLIF
    | ResolvedAdaptiveLIF
    | ResolvedPerEdgeLIF
    | ResolvedReactiveIF
)
ExecutableModel = AnalyticalModel | ResolvedSteppedNeuron
_ANALYTICAL_MODEL_TYPES = (
    ResolvedScalarLIF,
    ResolvedAlphaLIF,
    ResolvedAdaptiveLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
)
_EXECUTABLE_MODEL_TYPES = _ANALYTICAL_MODEL_TYPES + (ResolvedSteppedNeuron,)


@dataclass(frozen=True)
class _RuntimeDriveBinding:
    aliases: tuple[str, ...]
    parameter_index: int
    domain: ParameterDomain


@dataclass(frozen=True)
class _RuntimeNodeLayout:
    state_count: int
    drives: tuple[_RuntimeDriveBinding, ...]


@dataclass(frozen=True)
class _NodeLowering:
    program_identity: int | None
    polarity: NeuronPolarity
    threshold: float
    refractory: float
    dispatch: DispatchForm
    readout_index: int
    propagation_dag: ExprDAG
    normal_roots: tuple[str, ...]
    clamped_roots: tuple[str, ...]
    reset_roots: tuple[str, ...]
    parameter_values: tuple[float, ...]
    root_hint: object | None
    numerical: NumericalConfig | None
    event_batched: bool
    reset_before_deposit: bool
    scalar_affine: ScalarAffineParameters | None
    hazard: object | None
    deposit_dag: ExprDAG | None = None
    deposit_root: str | None = None
    deposit_index: int | None = None


def _scalar_affine_lowering(model: object) -> ScalarAffineParameters | None:
    """Recover the equation-derived scalar operation for the legacy packer.

    The execution-plan compiler supplies this record directly.  This structural
    compatibility bridge keeps the older compiler entry point bit-identical
    until that entry point is removed after the unified executor is proven.
    """

    roots = tuple(getattr(model, "normal_roots", ()))
    hint = getattr(model, "root_hint", None)
    if (
        len(roots) == 1
        and (
            isinstance(hint, ScalarLogRootHint)
            or getattr(model, "hazard", None) is not None
        )
        and all(hasattr(model, name) for name in ("a", "b", "reset"))
    ):
        return ScalarAffineParameters(
            float(getattr(model, "a")),
            float(getattr(model, "b")),
            float(getattr(model, "reset")),
        )
    return None

_LC_OK = 0
_LC_NO_CROSSING = 1
_LC_INVALID_ARGUMENT = 2
_LC_QUEUE_OVERFLOW = 6
_LC_OUTPUT_OVERFLOW = 7
_LC_ROOT_NONCONVERGENCE = 10
_LC_DECODER_OUTPUT_OVERFLOW = 11
_LC_TRACE_OVERFLOW = 12
_LC_INSPECTION_OVERFLOW = 13
_LC_ANALYTICAL_MAX_STATES = 8
_LC_LEARNING_EVENT_COUNT = 5
_LC_LEARNING_MAX_TRACES = 6
_UINT32_MAX = (1 << 32) - 1
_LC_ABI_VERSION = 17


class _CModel(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_double),
        ("b", ctypes.c_double),
        ("threshold", ctypes.c_double),
        ("reset", ctypes.c_double),
        ("refractory", ctypes.c_double),
        ("polarity", ctypes.c_uint32),
    ]


class _CState(ctypes.Structure):
    _fields_ = [("value", ctypes.c_double), ("t_last", ctypes.c_double)]


class _CCompiledGraphInfo(ctypes.Structure):
    _fields_ = [
        ("node_count", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
        ("parameter_count", ctypes.c_uint32),
        ("edge_count", ctypes.c_uint32),
        ("plastic_edge_count", ctypes.c_uint32),
    ]


class _CCompiledNodeLayout(ctypes.Structure):
    _fields_ = [
        ("state_offset", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
    ]


class _CEdge(ctypes.Structure):
    _fields_ = [
        ("pre", ctypes.c_uint32),
        ("post", ctypes.c_uint32),
        ("weight", ctypes.c_double),
        ("delay", ctypes.c_double),
    ]


class _CInputSpike(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("node", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CDriveUpdate(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("node", ctypes.c_uint32),
        ("b", ctypes.c_double),
    ]


class _COutputSpike(ctypes.Structure):
    _fields_ = [("t", ctypes.c_double), ("node", ctypes.c_uint32)]


class _CEncoderSpec(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("stream", ctypes.c_uint64),
        ("amplitude", ctypes.c_double),
        ("rate_min", ctypes.c_double),
        ("rate_max", ctypes.c_double),
        ("latency_min", ctypes.c_double),
        ("latency_max", ctypes.c_double),
        ("duration", ctypes.c_double),
        ("silence_threshold", ctypes.c_double),
        ("gain", ctypes.c_double),
        ("offset", ctypes.c_double),
        ("baseline", ctypes.c_double),
    ]


class _CEncoderState(ctypes.Structure):
    _fields_ = [
        ("phase_remaining", ctypes.c_double),
        ("poisson_remaining", ctypes.c_double),
        ("last_end", ctypes.c_double),
        ("draw_index", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("initialized", ctypes.c_uint32),
        ("poisson_ready", ctypes.c_uint32),
    ]


class _CPresentation(ctypes.Structure):
    _fields_ = [
        ("t_start", ctypes.c_double),
        ("t_end", ctypes.c_double),
        ("encoder", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CEncodedSpike(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("encoder", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CEncodedDrive(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("encoder", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CDecoderSpec(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("node", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("first_only", ctypes.c_uint32),
        ("normalize", ctypes.c_uint32),
        ("emission", ctypes.c_uint32),
        ("width", ctypes.c_double),
        ("origin", ctypes.c_double),
        ("tau", ctypes.c_double),
    ]


class _CDecodeWindow(ctypes.Structure):
    _fields_ = [("t_start", ctypes.c_double), ("t_end", ctypes.c_double)]


class _CDecoderWindowBinding(ctypes.Structure):
    _fields_ = [
        ("decoder", ctypes.c_uint32),
        ("window", ctypes.c_uint32),
        ("t_start", ctypes.c_double),
        ("t_end", ctypes.c_double),
    ]


class _CDecoderQueryBinding(ctypes.Structure):
    _fields_ = [
        ("decoder", ctypes.c_uint32),
        ("window", ctypes.c_uint32),
        ("t", ctypes.c_double),
    ]


class _CDecodeResult(ctypes.Structure):
    _fields_ = [
        ("decoder", ctypes.c_uint32),
        ("window", ctypes.c_uint32),
        ("valid", ctypes.c_uint32),
        ("count", ctypes.c_uint64),
        ("value", ctypes.c_double),
        ("first_spike", ctypes.c_double),
        ("window_start", ctypes.c_double),
        ("window_end", ctypes.c_double),
    ]


class _CDecodedEvent(ctypes.Structure):
    _fields_ = [
        ("decoder", ctypes.c_uint32),
        ("window", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("valid", ctypes.c_uint32),
        ("has_source", ctypes.c_uint32),
        ("has_first_spike", ctypes.c_uint32),
        ("count", ctypes.c_uint64),
        ("emitted_at", ctypes.c_double),
        ("source_spike_time", ctypes.c_double),
        ("window_start", ctypes.c_double),
        ("window_end", ctypes.c_double),
        ("observed_through", ctypes.c_double),
        ("value", ctypes.c_double),
        ("first_spike", ctypes.c_double),
    ]


class _CRunConfig(ctypes.Structure):
    _fields_ = [
        ("t_end", ctypes.c_double),
        ("queue_capacity", ctypes.c_uint64),
        ("output_capacity", ctypes.c_uint64),
        ("same_time_cascade_limit", ctypes.c_uint32),
        ("stochastic_seed", ctypes.c_uint64),
    ]


class _CRunStats(ctypes.Structure):
    _fields_ = [
        ("events_popped", ctypes.c_uint64),
        ("stale_predictions", ctypes.c_uint64),
        ("peak_queue_occupancy", ctypes.c_uint64),
        ("deliveries_scheduled", ctypes.c_uint64),
        ("deliveries_processed", ctypes.c_uint64),
        ("input_spikes_processed", ctypes.c_uint64),
        ("drive_updates_processed", ctypes.c_uint64),
        ("autonomous_spikes_confirmed", ctypes.c_uint64),
        ("output_spikes", ctypes.c_uint64),
        ("refractory_releases_processed", ctypes.c_uint64),
        ("max_same_time_cascade_depth", ctypes.c_uint32),
        ("kernel_seconds", ctypes.c_double),
    ]


class _CExprNode(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_uint32),
        ("lhs", ctypes.c_uint32),
        ("rhs", ctypes.c_uint32),
        ("binding", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CRootHint(ctypes.Structure):
    _fields_ = [
        ("g_root", ctypes.c_uint32),
        ("g_prime_root", ctypes.c_uint32),
        ("extremum_root", ctypes.c_uint32),
        ("extremum_prime_root", ctypes.c_uint32),
        ("asymptote_root", ctypes.c_uint32),
        ("membrane_coefficient_root", ctypes.c_uint32),
        ("synapse_constant_root", ctypes.c_uint32),
        ("synapse_linear_root", ctypes.c_uint32),
        ("membrane_rate_root", ctypes.c_uint32),
        ("synapse_rate_root", ctypes.c_uint32),
        ("threshold_root", ctypes.c_uint32),
        ("relative_tolerance", ctypes.c_double),
        ("fastest_time_constant", ctypes.c_double),
    ]


class _CTwoExpHint(ctypes.Structure):
    _fields_ = [
        ("g_root", ctypes.c_uint32),
        ("g_prime_root", ctypes.c_uint32),
        ("limit_root", ctypes.c_uint32),
        ("coefficient_one_root", ctypes.c_uint32),
        ("coefficient_two_root", ctypes.c_uint32),
        ("rate_one_root", ctypes.c_uint32),
        ("rate_two_root", ctypes.c_uint32),
        ("relative_tolerance", ctypes.c_double),
        ("fastest_time_constant", ctypes.c_double),
    ]


class _CScalarLogHint(ctypes.Structure):
    _fields_ = [
        ("decay_root", ctypes.c_uint32),
        ("affine_root", ctypes.c_uint32),
        ("threshold_root", ctypes.c_uint32),
    ]


class _CMultiExpHint(ctypes.Structure):
    _fields_ = [
        ("limit_root", ctypes.c_uint32),
        (
            "coefficient_roots",
            ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES,
        ),
        ("rate_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("mode_count", ctypes.c_uint32),
        ("iteration_cap", ctypes.c_uint32),
        ("relative_tolerance", ctypes.c_double),
        ("fastest_time_constant", ctypes.c_double),
    ]


class _CExpPolyHint(ctypes.Structure):
    _fields_ = [
        ("limit_root", ctypes.c_uint32),
        ("rate_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("coefficient_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        (
            "coefficient_offsets",
            ctypes.c_uint32 * (_LC_ANALYTICAL_MAX_STATES + 1),
        ),
        ("block_count", ctypes.c_uint32),
        ("coefficient_count", ctypes.c_uint32),
        ("iteration_cap", ctypes.c_uint32),
        ("relative_tolerance", ctypes.c_double),
        ("fastest_time_constant", ctypes.c_double),
    ]


class _CRootResult(ctypes.Structure):
    _fields_ = [
        ("t_spike", ctypes.c_double),
        ("horizon", ctypes.c_double),
        ("bracket_low", ctypes.c_double),
        ("bracket_high", ctypes.c_double),
        ("residual", ctypes.c_double),
        ("tolerance", ctypes.c_double),
        ("iterations", ctypes.c_uint32),
        ("extrema_count", ctypes.c_uint32),
    ]


class _CStepConfig(ctypes.Structure):
    _fields_ = [
        ("relative_tolerance", ctypes.c_double),
        ("absolute_tolerance", ctypes.c_double),
        ("initial_step", ctypes.c_double),
        ("minimum_step", ctypes.c_double),
        ("maximum_step", ctypes.c_double),
        ("event_tolerance", ctypes.c_double),
        ("maximum_steps", ctypes.c_uint32),
        ("maximum_rhs_evaluations", ctypes.c_uint32),
    ]


class _CStepResult(ctypes.Structure):
    _fields_ = [
        ("t_reached", ctypes.c_double),
        ("t_crossing", ctypes.c_double),
        ("last_step", ctypes.c_double),
        ("error_norm", ctypes.c_double),
        ("accepted_steps", ctypes.c_uint32),
        ("rejected_steps", ctypes.c_uint32),
        ("rhs_evaluations", ctypes.c_uint32),
        ("event_iterations", ctypes.c_uint32),
    ]


class _CHazardConfig(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("trajectory_mode_count", ctypes.c_uint32),
        ("trajectory_limit_root", ctypes.c_uint32),
        (
            "trajectory_coefficient_roots",
            ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES,
        ),
        ("trajectory_rate_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("log_scale", ctypes.c_double),
        ("voltage_gain", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("absolute_tolerance", ctypes.c_double),
        ("time_tolerance", ctypes.c_double),
        ("maximum_quadrature_depth", ctypes.c_uint32),
        ("maximum_root_iterations", ctypes.c_uint32),
    ]


class _CMixedNode(ctypes.Structure):
    _fields_ = [
        ("dispatch", ctypes.c_uint32),
        ("crossing_kind", ctypes.c_uint32),
        ("state_offset", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
        ("readout", ctypes.c_uint32),
        ("parameter_offset", ctypes.c_uint32),
        ("parameter_count", ctypes.c_uint32),
        ("threshold", ctypes.c_double),
        ("refractory", ctypes.c_double),
        ("program_nodes", ctypes.POINTER(_CExprNode)),
        ("program_node_count", ctypes.c_uint32),
        ("normal_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("clamped_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("reset_roots", ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES),
        ("scalar_log_hint", _CScalarLogHint),
        ("root_hint", _CRootHint),
        ("two_exp_hint", _CTwoExpHint),
        ("multi_exp_hint", _CMultiExpHint),
        ("exp_poly_hint", _CExpPolyHint),
        ("step_config", _CStepConfig),
        ("deposit_nodes", ctypes.POINTER(_CExprNode)),
        ("deposit_node_count", ctypes.c_uint32),
        ("deposit_root", ctypes.c_uint32),
        ("deposit_target", ctypes.c_uint32),
        ("polarity", ctypes.c_uint32),
        ("reset_before_deposit", ctypes.c_uint32),
        ("arithmetic_kind", ctypes.c_uint32),
        ("scalar_affine_decay", ctypes.c_double),
        ("scalar_affine_drive", ctypes.c_double),
        ("scalar_affine_reset", ctypes.c_double),
        ("hazard", _CHazardConfig),
    ]


class _CMixedEdge(ctypes.Structure):
    _fields_ = [
        ("pre", ctypes.c_uint32),
        ("post", ctypes.c_uint32),
        ("deposit_kind", ctypes.c_uint32),
        ("target", ctypes.c_uint32),
        ("weight", ctypes.c_double),
        ("delay", ctypes.c_double),
        ("deposit_scale", ctypes.c_double),
    ]


class _CPlasticityRule(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("modulator", ctypes.c_uint32),
        ("consume_on_modulation", ctypes.c_uint32),
        ("weight_group", ctypes.c_uint32),
        ("tau_pre", ctypes.c_double),
        ("tau_post", ctypes.c_double),
        ("tau_pre_slow", ctypes.c_double),
        ("tau_post_slow", ctypes.c_double),
        ("a2_plus", ctypes.c_double),
        ("a2_minus", ctypes.c_double),
        ("a3_plus", ctypes.c_double),
        ("a3_minus", ctypes.c_double),
        ("tau_eligibility_plus", ctypes.c_double),
        ("tau_eligibility_minus", ctypes.c_double),
        ("positive_plus", ctypes.c_double),
        ("positive_minus", ctypes.c_double),
        ("negative_plus", ctypes.c_double),
        ("negative_minus", ctypes.c_double),
        ("learning_rate", ctypes.c_double),
        ("weight_min", ctypes.c_double),
        ("weight_max", ctypes.c_double),
    ]


class _CLearningEventProgram(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(_CExprNode)),
        ("node_count", ctypes.c_uint32),
        ("advance_mask", ctypes.c_uint32),
        ("variable_mask", ctypes.c_uint32),
        ("weight_root", ctypes.c_uint32),
        ("trace_update_count", ctypes.c_uint32),
        ("trace_indices", ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES),
        ("trace_roots", ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES),
    ]


class _CLearningObserverProgram(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(_CExprNode)),
        ("node_count", ctypes.c_uint32),
        ("variable_mask", ctypes.c_uint32),
        ("parameter_mask", ctypes.c_uint32),
        ("voltage_tau_parameter", ctypes.c_uint32),
        ("fast_activity_tau_parameter", ctypes.c_uint32),
        ("slow_activity_tau_parameter", ctypes.c_uint32),
        ("band_width_parameter", ctypes.c_uint32),
        ("fast_activity_root", ctypes.c_uint32),
        ("slow_activity_root", ctypes.c_uint32),
        ("gain_root", ctypes.c_uint32),
    ]


class _CLearningProgram(ctypes.Structure):
    _fields_ = [
        ("parameter_count", ctypes.c_uint32),
        ("variable_count", ctypes.c_uint32),
        ("trace_count", ctypes.c_uint32),
        ("clamp_normalized_weight", ctypes.c_uint32),
        ("compatibility_kind", ctypes.c_uint32),
        ("trace_tau_parameters", ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES),
        ("trace_storage_slots", ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES),
        ("events", _CLearningEventProgram * _LC_LEARNING_EVENT_COUNT),
        ("observer", _CLearningObserverProgram),
    ]


class _CLearningBinding(ctypes.Structure):
    _fields_ = [
        ("program", ctypes.c_uint32),
        ("parameter_offset", ctypes.c_uint32),
        ("modulator", ctypes.c_uint32),
        ("weight_group", ctypes.c_uint32),
        ("weight_min", ctypes.c_double),
        ("weight_max", ctypes.c_double),
    ]


class _CModulationEvent(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("modulator", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CPlasticityState(ctypes.Structure):
    _fields_ = [
        ("edge", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("weight", ctypes.c_double),
        ("pre_fast", ctypes.c_double),
        ("post_fast", ctypes.c_double),
        ("pre_slow", ctypes.c_double),
        ("post_slow", ctypes.c_double),
        ("eligibility_plus", ctypes.c_double),
        ("eligibility_minus", ctypes.c_double),
        ("t_pre_fast", ctypes.c_double),
        ("t_post_fast", ctypes.c_double),
        ("t_pre_slow", ctypes.c_double),
        ("t_post_slow", ctypes.c_double),
        ("t_eligibility_plus", ctypes.c_double),
        ("t_eligibility_minus", ctypes.c_double),
    ]


class _CLearningObserverSnapshot(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("slow_voltage", ctypes.c_double),
        ("fast_activity", ctypes.c_double),
        ("slow_activity", ctypes.c_double),
        ("sensitivity", ctypes.c_double),
        ("t_activity", ctypes.c_double),
    ]


class _CMixedInputSpike(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("node", ctypes.c_uint32),
        ("deposit_kind", ctypes.c_uint32),
        ("target", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CMixedDriveUpdate(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("node", ctypes.c_uint32),
        ("binding", ctypes.c_uint32),
        ("value", ctypes.c_double),
    ]


class _CNetworkError(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_uint32),
        ("t", ctypes.c_double),
        ("root", _CRootResult),
        ("resource", ctypes.c_uint32),
        ("event_kind", ctypes.c_uint32),
        ("event_phase", ctypes.c_uint32),
        ("has_event", ctypes.c_uint32),
        ("event_index", ctypes.c_uint32),
        ("capacity", ctypes.c_uint64),
        ("occupancy", ctypes.c_uint64),
        ("peak", ctypes.c_uint64),
    ]


class _CTraceRecord(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("sequence", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("kind", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("node", ctypes.c_uint32),
        ("subject", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
        ("value", ctypes.c_double),
        ("before", ctypes.c_double * _LC_ANALYTICAL_MAX_STATES),
        ("after", ctypes.c_double * _LC_ANALYTICAL_MAX_STATES),
    ]


_CTraceConsumer = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_CTraceRecord), ctypes.c_void_p
)


class _CTraceConfig(ctypes.Structure):
    _fields_ = [
        ("kind_mask", ctypes.c_uint64),
        ("node_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("node_mask_count", ctypes.c_uint32),
        ("capture_state", ctypes.c_uint32),
        ("records", ctypes.POINTER(_CTraceRecord)),
        ("capacity", ctypes.c_uint64),
        ("count", ctypes.POINTER(ctypes.c_uint64)),
        ("consumer", _CTraceConsumer),
        ("consumer_context", ctypes.c_void_p),
        ("sequence_base", ctypes.c_uint64),
    ]


class _CStateInspectionRequest(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("node", ctypes.c_uint32),
    ]


class _CStateInspectionResult(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_double),
        ("generation", ctypes.c_uint64),
        ("node", ctypes.c_uint32),
        ("state_count", ctypes.c_uint32),
        ("clamped", ctypes.c_uint32),
        ("values", ctypes.c_double * _LC_ANALYTICAL_MAX_STATES),
    ]


class _CStateInspectionConfig(ctypes.Structure):
    _fields_ = [
        ("requests", ctypes.POINTER(_CStateInspectionRequest)),
        ("request_count", ctypes.c_uint32),
        ("results", ctypes.POINTER(_CStateInspectionResult)),
        ("capacity", ctypes.c_uint64),
        ("count", ctypes.POINTER(ctypes.c_uint64)),
    ]


@dataclass(frozen=True)
class ScalarState:
    """Scalar neuron state and its last update time."""

    value: float
    t_last: float


@dataclass(frozen=True)
class AugmentedState:
    """Multivariable neuron state and its last update time."""

    values: tuple[float, ...]
    t_last: float


class DepositKind(IntEnum):
    """Operation applied when a spike reaches its target."""

    STATE_ADD = 0
    PROGRAM = 1
    FOLDED_ALPHA = PROGRAM


class TraceKind(IntEnum):
    """Causal operation represented by a trace record."""

    INPUT_SPIKE = 0
    DELIVERY = 1
    DRIVE_UPDATE = 2
    REFRACTORY_RELEASE = 3
    STALE_PREDICTION = 4
    PREDICTION_CONFIRMED = 5
    DEPOSIT_APPLY = 6
    SPIKE = 7
    RESET = 8
    REFRACTORY_ENTER = 9
    FINAL_STATE = 10
    MODULATION = 11


class TracePhase(IntEnum):
    """Same-time event phase represented by a trace record."""

    BOUNDARY = 0
    DEPOSIT = 1
    PREDICTION = 2
    FIRE = 3
    FINAL = 4


@dataclass(frozen=True)
class TraceRecord:
    """One causal event with optional before and after state snapshots."""

    t: float
    sequence: int
    generation: int
    kind: TraceKind
    phase: TracePhase
    node: int
    subject: int | None
    value: float
    state_indices: tuple[int, ...] = ()
    state_names: tuple[str, ...] = ()
    before: tuple[float, ...] = ()
    after: tuple[float, ...] = ()


@dataclass(frozen=True)
class RecordingConfig:
    """Optional causal trace selection for one analytical network run.

    ``nodes`` are zero-based node indices at the low-level evaluator boundary.
    Graph-level runs translate graph node identifiers to these indices.
    """

    kinds: frozenset[TraceKind] | None = None
    nodes: tuple[int, ...] | None = None
    capture_state: bool = True
    state_indices: tuple[int, ...] | None = None
    capacity: int = 4096
    consumer: Callable[[TraceRecord], None] | None = None


@dataclass(frozen=True)
class StateInspectionRequest:
    """Read one node's settled state at an explicit simulation timestamp."""

    t: float
    node: int
    state_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class StateInspection:
    """Settled node state sampled at an explicit simulation time."""

    t: float
    node: int
    generation: int
    clamped: bool
    state_indices: tuple[int, ...]
    values: tuple[float, ...]
    state_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    """Predicted scalar threshold crossing and dispatch method."""

    dispatch: DispatchForm
    t_spike: float | None


@dataclass(frozen=True)
class RootDiagnostics:
    """Numerical evidence returned by an analytical root finder."""

    horizon: float
    bracket_low: float
    bracket_high: float
    residual: float
    tolerance: float
    iterations: int
    extrema_count: int


@dataclass(frozen=True)
class AlphaPrediction:
    """Crossing prediction for a folded alpha trajectory."""

    dispatch: DispatchForm
    t_spike: float | None
    diagnostics: RootDiagnostics


@dataclass(frozen=True)
class AdaptivePrediction:
    """Crossing prediction for an adaptive analytical trajectory."""

    dispatch: DispatchForm
    t_spike: float | None
    diagnostics: RootDiagnostics


@dataclass(frozen=True)
class StepDiagnostics:
    """Work and error summary from adaptive ODE integration."""

    t_reached: float
    last_step: float
    error_norm: float
    accepted_steps: int
    rejected_steps: int
    rhs_evaluations: int
    event_iterations: int


@dataclass(frozen=True)
class SteppedPrediction:
    """Crossing prediction from controlled numerical integration."""

    dispatch: DispatchForm
    t_spike: float | None
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class DeltaEdge:
    """Chemical connection with a magnitude or signed MIXED-source weight."""

    pre: int
    post: int
    weight: float
    delay: float = 0.0


@dataclass(frozen=True)
class MixedEdge:
    """Chemical connection with a magnitude or signed MIXED-source weight."""

    pre: int
    post: int
    weight: float
    delay: float = 0.0
    deposit_kind: DepositKind = DepositKind.STATE_ADD
    target: int = 0
    deposit_scale: float = 1.0


@dataclass(frozen=True)
class InputSpike:
    """External scalar spike submitted directly to a node."""

    t: float
    node: int
    value: float


@dataclass(frozen=True)
class MixedInputSpike:
    """External spike with an explicit deposit operation and target."""

    t: float
    node: int
    value: float
    deposit_kind: DepositKind = DepositKind.STATE_ADD
    target: int = 0


@dataclass(frozen=True)
class DriveUpdate:
    """Boundary-time replacement of a scalar node's affine b coefficient."""

    t: float
    node: int
    b: float


@dataclass(frozen=True)
class MixedDriveUpdate:
    """Boundary update of a validated affine-drive parameter."""

    t: float
    node: int
    value: float
    binding: str | None = None


@dataclass(frozen=True)
class ModulationEvent:
    """Timestamped third factor routed by modulator identifier."""

    t: float
    modulator: int
    value: float


@dataclass(frozen=True)
class PlasticityState:
    """Observable weight and trace state for one plastic edge."""

    edge: int
    kind: str
    weight: float
    pre_fast: float
    post_fast: float
    pre_slow: float
    post_slow: float
    eligibility_plus: float
    eligibility_minus: float
    t_pre_fast: float
    t_post_fast: float
    t_pre_slow: float
    t_post_slow: float
    t_eligibility_plus: float
    t_eligibility_minus: float

    @property
    def voltage_eligibility(self) -> float:
        """Voltage eligibility stored in the first auxiliary trace slot."""
        return self.pre_slow

    @property
    def t_voltage_eligibility(self) -> float:
        """Timestamp of the voltage eligibility auxiliary trace."""
        return self.t_pre_slow


@dataclass(frozen=True)
class Spike:
    """Timestamped output spike emitted by a node."""

    t: float
    node: int


@dataclass(frozen=True)
class LearningObserverState:
    """Read-only neuron-local state retained by a learning observer."""

    node: int
    active: bool
    slow_voltage: float
    fast_activity: float
    slow_activity: float
    sensitivity: float
    t_activity: float


@dataclass(frozen=True)
class RunStats:
    """Event and queue counters collected by one C execution."""

    events_popped: int
    stale_predictions: int
    peak_queue_occupancy: int
    deliveries_scheduled: int
    deliveries_processed: int
    input_spikes_processed: int
    drive_updates_processed: int
    autonomous_spikes_confirmed: int
    output_spikes: int
    refractory_releases_processed: int
    max_same_time_cascade_depth: int


@dataclass(frozen=True)
class RunResult:
    """Final state, spikes, and statistics from a scalar graph run."""

    states: tuple[ScalarState, ...]
    spikes: tuple[Spike, ...]
    stats: RunStats
    kernel_seconds: float = field(default=0.0, compare=False)
    weights: tuple[float, ...] = ()
    plasticity: tuple[PlasticityState, ...] = ()


@dataclass(frozen=True)
class MixedRunResult:
    """Final state, outputs, recordings, and learning state from a mixed run."""

    states: tuple[AugmentedState, ...]
    spikes: tuple[Spike, ...]
    stats: RunStats
    kernel_seconds: float = field(default=0.0, compare=False)
    decoded: tuple[DecodeValue, ...] = ()
    decoded_events: tuple[DecodeEvent, ...] = ()
    trace: tuple[TraceRecord, ...] = ()
    inspections: tuple[StateInspection, ...] = ()
    weights: tuple[float, ...] = ()
    plasticity: tuple[PlasticityState, ...] = ()
    learning_observers: tuple[LearningObserverState, ...] = ()


@dataclass(frozen=True)
class NativePrecisionInfo:
    """Validated arithmetic metadata, with legacy ABI inference labeled explicitly."""

    profile: PrecisionProfile
    metadata_source: str
    metadata_version: int | None
    real_bits: int
    time_bits: int
    real_mant_dig: int
    time_mant_dig: int
    real_max_exp: int
    time_max_exp: int
    wide_bits: int | None
    wide_mant_dig: int | None
    wide_max_exp: int | None
    radix: int
    arithmetic_revision: int


def _native_precision_info(lib, profile: PrecisionProfile) -> NativePrecisionInfo:
    """Check only integer entry points before binding precision-bearing calls."""

    query = getattr(lib, "lc_numeric_property", None)
    check = getattr(lib, "lc_numeric_profile_check", None)
    if (query is None) != (check is None):
        raise CoreError("C evaluator exposes incomplete numeric metadata")
    if query is None:
        # Original ABI17 libraries use fixed binary64 layouts without this query.
        if profile is not PrecisionProfile.FLOAT64:
            raise CoreError("reduced precision requires native numeric metadata")
        return NativePrecisionInfo(
            profile=profile,
            metadata_source="legacy-abi17",
            metadata_version=None,
            real_bits=64,
            time_bits=64,
            real_mant_dig=53,
            time_mant_dig=53,
            real_max_exp=1024,
            time_max_exp=1024,
            wide_bits=None,
            wide_mant_dig=None,
            wide_max_exp=None,
            radix=2,
            arithmetic_revision=1,
        )
    query.argtypes = [ctypes.c_uint32]
    query.restype = ctypes.c_uint32
    check.argtypes = [ctypes.c_uint32] * 4
    check.restype = ctypes.c_int
    # Field order is the stable integer contract in lacuna_numeric.h.
    fields = (
        "version", "profile", "real_bits", "time_bits", "real_mant_dig",
        "time_mant_dig", "real_max_exp", "time_max_exp", "wide_bits",
        "wide_mant_dig", "wide_max_exp", "radix", "arithmetic_revision",
    )
    values = {name: int(query(index)) for index, name in enumerate(fields, 1)}
    expected = {
        "version": 1,
        "profile": profile.native_id,
        "real_bits": profile.real_bits,
        "time_bits": profile.time_bits,
        "real_mant_dig": {16: 11, 32: 24, 64: 53}[profile.real_bits],
        "time_mant_dig": {16: 11, 32: 24, 64: 53}[profile.time_bits],
        "real_max_exp": {16: 16, 32: 128, 64: 1024}[profile.real_bits],
        "time_max_exp": {16: 16, 32: 128, 64: 1024}[profile.time_bits],
        "radix": 2,
        "arithmetic_revision": profile.arithmetic_revision,
    }
    for name, required in expected.items():
        if values[name] != required:
            raise CoreError(
                f"C evaluator precision mismatch for {name}: "
                f"expected {required}, received {values[name]}"
            )
    if profile is PrecisionProfile.FLOAT64:
        if (
            values["wide_bits"] < values["real_bits"]
            or not 53 <= values["wide_mant_dig"] <= values["wide_bits"]
            or values["wide_max_exp"] < values["real_max_exp"]
        ):
            raise CoreError("C evaluator exposes invalid wide-arithmetic metadata")
    else:
        wide = ((16, 11, 16) if profile is PrecisionProfile.FLOAT16 else
                (32, 24, 128) if profile is PrecisionProfile.FLOAT32 else (64, 53, 1024))
        if tuple(values[name] for name in (
            "wide_bits", "wide_mant_dig", "wide_max_exp",
        )) != wide:
            raise CoreError("C evaluator wide arithmetic does not match the profile")
    if check(
        profile.native_id, profile.real_bits, profile.time_bits,
        profile.arithmetic_revision,
    ) != _LC_OK:
        raise CoreError("C evaluator rejected the requested precision contract")
    return NativePrecisionInfo(
        profile=profile,
        metadata_source="native",
        metadata_version=values["version"],
        **{name: values[name] for name in fields if name not in ("version", "profile")},
    )


_PROFILE_CTYPES = {}
_HALF_SCALAR_CALLS = (
    "encoder_run_create", "encoder_run_advance", "decoder_run_advance",
    "decoder_run_advance_before", "scalar_advance", "expr_state_advance",
    "expr_state_deposit", "expr_step_advance", "expr_step_predict",
    "expr_alpha_predict", "expr_two_exp_predict", "expr_multi_exp_predict",
    "expr_exp_poly_predict", "mixed_run_advance_incremental",
)


class _HalfHostFunction:
    """Bind bit transport to this evaluator's exact native entry point."""

    def __init__(self, native, transport):
        self._native = native
        self._address = ctypes.cast(native, ctypes.c_void_p)
        self._transport = transport
        self._argtypes = None

    @property
    def argtypes(self):
        return self._argtypes

    @argtypes.setter
    def argtypes(self, value):
        self._argtypes = value
        self._transport.argtypes = [ctypes.c_void_p, *value]

    @property
    def restype(self):
        return self._transport.restype

    @restype.setter
    def restype(self, value):
        self._transport.restype = value

    def __call__(self, *args):
        return self._transport(self._address, *args)


def _native_types(profile: PrecisionProfile):
    """Reuse a profile's private types without altering the legacy classes."""

    if profile.cache_key not in _PROFILE_CTYPES:
        legacy = {
            name: value for name, value in globals().items()
            if name.startswith("_C") and isinstance(value, type)
            and issubclass(value, (ctypes.Structure, ctypes._CFuncPtr))
        }
        _PROFILE_CTYPES[profile.cache_key] = build_native_types(profile, legacy)
    return _PROFILE_CTYPES[profile.cache_key]


class CoreEvaluator:
    """Call the C core without implementing simulation arithmetic in Python."""

    def __init__(
        self,
        library: str | os.PathLike[str] | None = None,
        *,
        precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
        time_precision: str | None = None,
    ):
        profile = normalize_precision(precision, time_precision=time_precision)
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        filename = "lacuna_core.dll" if os.name == "nt" else "liblacuna_core" + suffix
        directory = "build" if profile is PrecisionProfile.FLOAT64 else "build-" + profile.value
        path = Path(library or os.environ.get("LACUNA_CORE_LIB", str(Path(directory) / filename)))
        self._lib = ctypes.CDLL(str(path))
        try:
            self._lib.lc_abi_version.argtypes = []
            self._lib.lc_abi_version.restype = ctypes.c_uint32
            self._lib.lc_sizeof_network_error.argtypes = []
            self._lib.lc_sizeof_network_error.restype = ctypes.c_uint64
        except AttributeError as exc:
            raise CoreError("C evaluator does not expose the required ABI metadata") from exc
        actual_abi = int(self._lib.lc_abi_version())
        actual_error_size = int(self._lib.lc_sizeof_network_error())
        expected_abi = (19 if profile is PrecisionProfile.FLOAT16 else
                        _LC_ABI_VERSION if profile is PrecisionProfile.FLOAT64 else 18)
        if actual_abi != expected_abi:
            raise CoreError(
                f"C evaluator ABI mismatch: expected version={expected_abi}, "
                f"received version={actual_abi}"
            )
        self._precision_info = _native_precision_info(self._lib, profile)
        if profile is PrecisionProfile.FLOAT16:
            bridge_name = "lacuna_half_host.dll" if os.name == "nt" else "liblacuna_half_host" + suffix
            bridge_path = path.with_name(bridge_name)
            try:
                self._half_host = ctypes.CDLL(str(bridge_path))
                version = self._half_host.lc_host_half_transport_version
                version.argtypes = []
                version.restype = ctypes.c_uint32
                if version() != 1:
                    raise CoreError("unsupported half host transport version")
                check = self._half_host.lc_host_half_profile_check
                check.argtypes = [ctypes.c_void_p]
                check.restype = ctypes.c_int
                if check(ctypes.cast(self._lib.lc_numeric_profile_check, ctypes.c_void_p)) != _LC_OK:
                    raise CoreError("half host bridge does not match the native precision")
                for name in _HALF_SCALAR_CALLS:
                    setattr(self._lib, "lc_" + name, _HalfHostFunction(
                        getattr(self._lib, "lc_" + name),
                        getattr(self._half_host, "lc_host_half_" + name),
                    ))
            except (OSError, AttributeError) as exc:
                raise CoreError("float16 requires its matching host-only scalar transport bridge") from exc
        self._types = _native_types(profile)
        expected_error_size = ctypes.sizeof(self._types._CNetworkError)
        if actual_error_size != expected_error_size:
            raise CoreError(
                "C evaluator ABI mismatch: "
                f"expected network_error_size={expected_error_size}, "
                f"received network_error_size={actual_error_size}"
            )
        self.abi_version = actual_abi
        self.network_error_size = actual_error_size
        self._lib.lc_status_string.argtypes = [ctypes.c_int]
        self._lib.lc_status_string.restype = ctypes.c_char_p
        self._lib.lc_encoder_state_reset.argtypes = [
            ctypes.POINTER(self._types._CEncoderState),
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        self._lib.lc_encoder_state_reset.restype = ctypes.c_int
        self._lib.lc_encode_presentations.argtypes = [
            ctypes.POINTER(self._types._CEncoderSpec),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CEncoderState),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CPresentation),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CEncodedSpike),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CEncodedDrive),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_encode_presentations.restype = ctypes.c_int
        self._lib.lc_encoder_run_create.argtypes = [
            ctypes.POINTER(self._types._CEncoderSpec),
            ctypes.c_uint32,
            ctypes.c_uint64,
            self._types.time_argument,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_encoder_run_create.restype = ctypes.c_int
        self._lib.lc_encoder_run_advance.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CPresentation),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CEncodedSpike),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CEncodedDrive),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_encoder_run_advance.restype = ctypes.c_int
        self._lib.lc_encoder_run_reset_episode.argtypes = [ctypes.c_void_p]
        self._lib.lc_encoder_run_reset_episode.restype = ctypes.c_int
        self._lib.lc_encoder_run_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lc_encoder_run_destroy.restype = None
        self._lib.lc_decode_spikes.argtypes = [
            ctypes.POINTER(self._types._CDecoderSpec),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.c_uint64,
            ctypes.POINTER(self._types._CDecodeWindow),
            ctypes.POINTER(self._types._CDecodeResult),
            ctypes.c_uint32,
        ]
        self._lib.lc_decode_spikes.restype = ctypes.c_int
        self._lib.lc_decoder_bank_compile.argtypes = [
            ctypes.POINTER(self._types._CDecoderSpec),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_decoder_bank_compile.restype = ctypes.c_int
        self._lib.lc_decoder_bank_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lc_decoder_bank_destroy.restype = None
        self._lib.lc_decoder_run_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_decoder_run_create.restype = ctypes.c_int
        self._lib.lc_decoder_run_reserve_events.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._lib.lc_decoder_run_reserve_events.restype = ctypes.c_int
        self._lib.lc_decoder_run_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecodeWindow),
        ]
        self._lib.lc_decoder_run_reset.restype = ctypes.c_int
        self._lib.lc_decoder_run_reset_windows.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecodeWindow),
            ctypes.c_uint32,
        ]
        self._lib.lc_decoder_run_reset_windows.restype = ctypes.c_int
        self._lib.lc_decoder_run_reset_schedule.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecoderWindowBinding),
            ctypes.c_uint64,
        ]
        self._lib.lc_decoder_run_reset_schedule.restype = ctypes.c_int
        self._lib.lc_decoder_run_set_queries.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecoderQueryBinding),
            ctypes.c_uint64,
        ]
        self._lib.lc_decoder_run_set_queries.restype = ctypes.c_int
        self._lib.lc_decoder_run_advance.argtypes = [
            ctypes.c_void_p,
            self._types.time_argument,
        ]
        self._lib.lc_decoder_run_advance.restype = ctypes.c_int
        self._lib.lc_decoder_run_advance_before.argtypes = [
            ctypes.c_void_p,
            self._types.time_argument,
        ]
        self._lib.lc_decoder_run_advance_before.restype = ctypes.c_int
        self._lib.lc_decoder_run_consume.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._COutputSpike),
        ]
        self._lib.lc_decoder_run_consume.restype = ctypes.c_int
        self._lib.lc_decoder_run_result_count.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_decoder_run_result_count.restype = ctypes.c_int
        self._lib.lc_decoder_run_finalize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecodeResult),
            ctypes.c_uint64,
        ]
        self._lib.lc_decoder_run_finalize.restype = ctypes.c_int
        self._lib.lc_decoder_run_copy_events.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CDecodedEvent),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_decoder_run_copy_events.restype = ctypes.c_int
        self._lib.lc_decoder_run_event_usage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_decoder_run_event_usage.restype = ctypes.c_int
        self._lib.lc_decoder_run_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lc_decoder_run_destroy.restype = None
        self._lib.lc_scalar_advance.argtypes = [
            ctypes.POINTER(self._types._CModel),
            ctypes.POINTER(self._types._CState),
            self._types.time_argument,
        ]
        self._lib.lc_scalar_advance.restype = ctypes.c_int
        self._lib.lc_scalar_predict.argtypes = [
            ctypes.POINTER(self._types._CModel),
            ctypes.POINTER(self._types._CState),
            ctypes.POINTER(self._types.time_type),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.lc_scalar_predict.restype = ctypes.c_int
        self._lib.lc_delta_network_run.argtypes = [
            ctypes.POINTER(self._types._CModel),
            ctypes.POINTER(self._types._CState),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CEdge),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
        ]
        self._lib.lc_delta_network_run.restype = ctypes.c_int
        self._lib.lc_expr_evaluate.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_evaluate.restype = ctypes.c_int
        self._lib.lc_expr_evaluate_selected.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_evaluate_selected.restype = ctypes.c_int
        self._lib.lc_expr_state_advance.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.POINTER(self._types.time_type),
            self._types.time_argument,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_state_advance.restype = ctypes.c_int
        self._lib.lc_expr_state_map.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_state_map.restype = ctypes.c_int
        self._lib.lc_expr_state_deposit.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.c_uint32,
            self._types.real_argument,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_state_deposit.restype = ctypes.c_int
        self._lib.lc_expr_step_advance.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CStepConfig),
            ctypes.POINTER(self._types.real_type),
            ctypes.POINTER(self._types.time_type),
            self._types.time_argument,
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CStepResult),
        ]
        self._lib.lc_expr_step_advance.restype = ctypes.c_int
        self._lib.lc_expr_step_predict.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            self._types.real_argument,
            ctypes.POINTER(self._types._CStepConfig),
            ctypes.POINTER(self._types.real_type),
            self._types.time_argument,
            self._types.time_argument,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CStepResult),
        ]
        self._lib.lc_expr_step_predict.restype = ctypes.c_int
        self._lib.lc_expr_alpha_predict.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRootHint),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.POINTER(self._types._CRootResult),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_alpha_predict.restype = ctypes.c_int
        self._lib.lc_expr_two_exp_predict.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CTwoExpHint),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.POINTER(self._types._CRootResult),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_two_exp_predict.restype = ctypes.c_int
        self._lib.lc_expr_multi_exp_predict.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMultiExpHint),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.POINTER(self._types._CRootResult),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_multi_exp_predict.restype = ctypes.c_int
        self._lib.lc_expr_exp_poly_predict.argtypes = [
            ctypes.POINTER(self._types._CExprNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CExpPolyHint),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.POINTER(self._types._CRootResult),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_expr_exp_poly_predict.restype = ctypes.c_int
        self._lib.lc_mixed_network_run.argtypes = [
            ctypes.POINTER(self._types._CMixedNode),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.time_type),
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedEdge),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
        ]
        self._lib.lc_mixed_network_run.restype = ctypes.c_int
        self._lib.lc_mixed_graph_compile.argtypes = [
            ctypes.POINTER(self._types._CMixedNode),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedEdge),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_mixed_graph_compile.restype = ctypes.c_int
        self._lib.lc_mixed_graph_compile_plastic.argtypes = [
            ctypes.POINTER(self._types._CMixedNode),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedEdge),
            ctypes.POINTER(self._types._CPlasticityRule),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_mixed_graph_compile_plastic.restype = ctypes.c_int
        self._lib.lc_mixed_graph_compile_learning.argtypes = [
            ctypes.POINTER(self._types._CMixedNode),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedEdge),
            ctypes.POINTER(self._types._CLearningProgram),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CLearningBinding),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_mixed_graph_compile_learning.restype = ctypes.c_int
        self._lib.lc_mixed_graph_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lc_mixed_graph_destroy.restype = None
        self._lib.lc_compiled_graph_image_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_compiled_graph_image_size.restype = ctypes.c_int
        self._lib.lc_compiled_graph_serialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.lc_compiled_graph_serialize.restype = ctypes.c_int
        self._lib.lc_compiled_graph_deserialize.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_compiled_graph_deserialize.restype = ctypes.c_int
        self._lib.lc_compiled_graph_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CCompiledGraphInfo),
        ]
        self._lib.lc_compiled_graph_get_info.restype = ctypes.c_int
        self._lib.lc_compiled_graph_copy_node_layouts.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CCompiledNodeLayout),
            ctypes.c_uint32,
        ]
        self._lib.lc_compiled_graph_copy_node_layouts.restype = ctypes.c_int
        self._lib.lc_mixed_run_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.time_type),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.lc_mixed_run_create.restype = ctypes.c_int
        self._lib.lc_mixed_run_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.time_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_mixed_run_reset.restype = ctypes.c_int
        self._lib.lc_mixed_run_begin_incremental.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._CNetworkError),
        ]
        self._lib.lc_mixed_run_begin_incremental.restype = ctypes.c_int
        self._lib.lc_mixed_run_advance_incremental.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            self._types.time_argument,
            ctypes.c_uint32,
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CTraceConfig),
            ctypes.POINTER(self._types._CStateInspectionConfig),
        ]
        self._lib.lc_mixed_run_advance_incremental.restype = ctypes.c_int
        self._lib.lc_mixed_run_reset_episode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CNetworkError),
        ]
        self._lib.lc_mixed_run_reset_episode.restype = ctypes.c_int
        self._lib.lc_mixed_run_execute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
        ]
        self._lib.lc_mixed_run_execute.restype = ctypes.c_int
        self._lib.lc_mixed_run_execute_recorded.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
            ctypes.POINTER(self._types._CTraceConfig),
        ]
        self._lib.lc_mixed_run_execute_recorded.restype = ctypes.c_int
        self._lib.lc_mixed_run_execute_with_decoders.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
            ctypes.c_void_p,
        ]
        self._lib.lc_mixed_run_execute_with_decoders.restype = ctypes.c_int
        self._lib.lc_mixed_run_execute_with_decoders_recorded.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CTraceConfig),
        ]
        self._lib.lc_mixed_run_execute_with_decoders_recorded.restype = ctypes.c_int
        self._lib.lc_mixed_run_execute_observed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CMixedInputSpike),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CMixedDriveUpdate),
            ctypes.c_uint32,
            ctypes.POINTER(self._types._CRunConfig),
            ctypes.POINTER(self._types._COutputSpike),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(self._types._CRunStats),
            ctypes.POINTER(self._types._CNetworkError),
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CTraceConfig),
            ctypes.POINTER(self._types._CStateInspectionConfig),
        ]
        self._lib.lc_mixed_run_execute_observed.restype = ctypes.c_int
        self._lib.lc_mixed_run_copy_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
            ctypes.POINTER(self._types.time_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_mixed_run_copy_state.restype = ctypes.c_int
        self._lib.lc_mixed_run_schedule_modulations.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CModulationEvent),
            ctypes.c_uint32,
        ]
        self._lib.lc_mixed_run_schedule_modulations.restype = ctypes.c_int
        self._lib.lc_mixed_run_copy_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types.real_type),
            ctypes.c_uint32,
        ]
        self._lib.lc_mixed_run_copy_weights.restype = ctypes.c_int
        self._lib.lc_mixed_run_copy_plasticity.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CPlasticityState),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._lib.lc_mixed_run_copy_plasticity.restype = ctypes.c_int
        self._lib.lc_mixed_run_copy_learning_observers.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._types._CLearningObserverSnapshot),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._lib.lc_mixed_run_copy_learning_observers.restype = ctypes.c_int
        self._lib.lc_mixed_run_destroy.argtypes = [ctypes.c_void_p]
        self._lib.lc_mixed_run_destroy.restype = None

    @property
    def precision(self) -> PrecisionProfile:
        """Return the validated model and timestamp precision profile."""

        return self._precision_info.profile

    @property
    def precision_info(self) -> NativePrecisionInfo:
        """Return native numeric metadata or explicitly labeled legacy inference."""

        return self._precision_info

    def _check_plan_precision(self, plan: ExecutionPlan) -> None:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("compiled plan must be an ExecutionPlan")
        profile = normalize_precision(plan.precision)
        if profile is not self.precision:
            raise CoreError("execution plan precision does not match the C evaluator")

    def _check_decoder_run(self, decoder_run: "StreamingDecoderRun | None") -> None:
        """Reject incompatible opaque decoder handles before native calls."""

        if decoder_run is None:
            return
        if not isinstance(decoder_run, StreamingDecoderRun):
            raise TypeError("decoder_run must be a StreamingDecoderRun")
        if decoder_run.closed:
            raise RuntimeError("streaming decoder run is closed")
        if decoder_run._compiled._core.precision is not self.precision:
            raise CoreError("decoder precision does not match the C evaluator")

    def _model(
        self, model: ResolvedScalarLIF,
        polarity: NeuronPolarity = NeuronPolarity.EXCITATORY,
    ) -> _CModel:
        if not isinstance(polarity, NeuronPolarity):
            raise ValueError(
                "polarity must be EXCITATORY, INHIBITORY, or MIXED"
            )
        return self._types._CModel(
            model.a,
            model.b,
            model.threshold,
            model.reset,
            model.refractory,
            polarity.runtime_code,
        )

    def _raise(self, status: int) -> None:
        message = self._lib.lc_status_string(status).decode("utf-8")
        raise CoreError(message, status=status)

    def _encoder_spec(self, encoder: Encoder, stream: int) -> _CEncoderSpec:
        values = dict(
            kind=int(encoder.kind),
            stream=stream,
            amplitude=float(getattr(encoder, "amplitude", 1.0)),
            rate_min=0.0,
            rate_max=0.0,
            latency_min=0.0,
            latency_max=0.0,
            duration=0.0,
            silence_threshold=0.0,
            gain=0.0,
            offset=0.0,
            baseline=0.0,
        )
        if isinstance(encoder, (RegularRateEncoder, PoissonRateEncoder, BurstEncoder)):
            values["rate_min"] = float(encoder.min_rate)
            values["rate_max"] = float(encoder.max_rate)
        if isinstance(encoder, (TTFSEncoder, LatencyBurstEncoder)):
            values["latency_min"] = float(encoder.min_latency)
            values["latency_max"] = float(encoder.max_latency)
            values["silence_threshold"] = float(encoder.silence_threshold)
        if isinstance(encoder, BurstEncoder):
            values["duration"] = float(encoder.duration)
        elif isinstance(encoder, LatencyBurstEncoder):
            values["rate_max"] = float(encoder.rate)
            values["duration"] = float(encoder.duration)
        elif isinstance(encoder, HeldCurrentEncoder):
            values["gain"] = float(encoder.gain)
            values["offset"] = float(encoder.offset)
            values["baseline"] = float(encoder.baseline)
        elif not isinstance(
            encoder,
            (NativeEventEncoder, RegularRateEncoder, PoissonRateEncoder, TTFSEncoder),
        ):
            raise TypeError(f"unsupported encoder type: {type(encoder).__name__}")
        return self._types._CEncoderSpec(**values)

    def encode_presentations(
        self,
        encoders: Sequence[Encoder],
        presentations: Sequence[Presentation],
        *,
        seed: int = 0,
        spike_capacity: int = 4096,
        drive_capacity: int | None = None,
    ) -> EncodedBatch:
        """Run configured encoders in C and return core input primitives."""

        if not isinstance(seed, int) or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if (
            not isinstance(spike_capacity, int)
            or not 0 <= spike_capacity <= 2**64 - 1
        ):
            raise ValueError(
                "spike_capacity must be an unsigned 64-bit integer"
            )
        if drive_capacity is None:
            drive_capacity = 2 * len(presentations)
        if (
            not isinstance(drive_capacity, int)
            or not 0 <= drive_capacity <= 2**64 - 1
        ):
            raise ValueError(
                "drive_capacity must be an unsigned 64-bit integer"
            )
        specs = (self._types._CEncoderSpec * len(encoders))(
            *(self._encoder_spec(item, index) for index, item in enumerate(encoders))
        )
        states = (self._types._CEncoderState * len(encoders))()
        status = self._lib.lc_encoder_state_reset(
            states, len(states), ctypes.c_uint64(seed)
        )
        if status != _LC_OK:
            self._raise(status)
        items = (self._types._CPresentation * len(presentations))(
            *(
                self._types._CPresentation(item.t_start, item.t_end, item.encoder, item.value)
                for item in presentations
            )
        )
        spikes = (self._types._CEncodedSpike * max(spike_capacity, 1))()
        drives = (self._types._CEncodedDrive * max(drive_capacity, 1))()
        spike_count = ctypes.c_uint64()
        drive_count = ctypes.c_uint64()
        status = self._lib.lc_encode_presentations(
            specs,
            len(specs),
            states,
            len(states),
            items,
            len(items),
            spikes,
            spike_capacity,
            ctypes.byref(spike_count),
            drives,
            drive_capacity,
            ctypes.byref(drive_count),
        )
        if status != _LC_OK:
            self._raise(status)
        return EncodedBatch(
            tuple(
                EncodedSpike(spikes[index].t, spikes[index].encoder, spikes[index].value)
                for index in range(spike_count.value)
            ),
            tuple(
                EncodedDrive(drives[index].t, drives[index].encoder, drives[index].value)
                for index in range(drive_count.value)
            ),
        )

    def create_encoder_session(
        self,
        encoders: Sequence[Encoder],
        *,
        seed: int = 0,
        spike_capacity: int = 4096,
        drive_capacity: int | None = None,
    ) -> "EncoderSession":
        """Create a stateful, watermark-driven scalar encoder session."""

        return EncoderSession(
            self,
            encoders,
            seed=seed,
            spike_capacity=spike_capacity,
            drive_capacity=drive_capacity,
        )

    def create_streaming_encoder_run(
        self,
        encoders: Sequence[Encoder],
        *,
        initial_frontier: float = 0.0,
        seed: int = 0,
        spike_capacity: int = 4096,
        drive_capacity: int = 4096,
    ) -> "StreamingEncoderRun":
        """Create a live encoder run whose active presentations span pauses."""

        return StreamingEncoderRun(
            self,
            encoders,
            initial_frontier=initial_frontier,
            seed=seed,
            spike_capacity=spike_capacity,
            drive_capacity=drive_capacity,
        )

    def _decoder_spec(self, binding: DecoderBinding) -> _CDecoderSpec:
        decoder = binding.decoder
        if isinstance(decoder, RateDecoder):
            return self._types._CDecoderSpec(
                0,
                binding.node,
                int(decoder.mode),
                0,
                0,
                int(decoder.emission),
                float(decoder.width or 0.0),
                float(decoder.origin),
                0.0,
            )
        if isinstance(decoder, TTFSDecoder):
            return self._types._CDecoderSpec(
                1,
                binding.node,
                0,
                0,
                int(decoder.normalize),
                int(decoder.emission),
                0.0,
                0.0,
                0.0,
            )
        if isinstance(decoder, TemporalWeightDecoder):
            return self._types._CDecoderSpec(
                2,
                binding.node,
                0,
                int(decoder.spikes is TemporalSpikeMode.FIRST),
                int(decoder.normalize),
                int(decoder.emission),
                0.0,
                0.0,
                float(decoder.tau),
            )
        raise TypeError(f"unsupported decoder type: {type(decoder).__name__}")

    def decode_spikes(
        self,
        decoders: Sequence[DecoderBinding],
        spikes: Sequence[Spike],
        *,
        t_start: float,
        t_end: float,
    ) -> tuple[DecodeValue, ...]:
        """Decode a chronological spike stream in C over ``[t_start, t_end)``."""

        specs = (self._types._CDecoderSpec * len(decoders))(
            *(self._decoder_spec(item) for item in decoders)
        )
        input_spikes = (self._types._COutputSpike * len(spikes))(
            *(self._types._COutputSpike(item.t, item.node) for item in spikes)
        )
        window = self._types._CDecodeWindow(float(t_start), float(t_end))
        results = (self._types._CDecodeResult * len(decoders))()
        status = self._lib.lc_decode_spikes(
            specs,
            len(specs),
            input_spikes,
            len(input_spikes),
            ctypes.byref(window),
            results,
            len(results),
        )
        if status != _LC_OK:
            self._raise(status)
        return tuple(
            DecodeValue(
                result.decoder,
                bool(result.valid),
                result.count,
                result.value if result.valid else None,
                result.first_spike if result.count else None,
                result.window,
                result.window_start,
                result.window_end,
            )
            for result in results
        )

    def compile_decoders(
        self,
        decoders: Sequence[DecoderBinding],
        *,
        node_count: int,
    ) -> "CompiledDecoderBank":
        """Compile immutable decoder specs and their node-to-decoder index in C."""

        if not decoders:
            raise ValueError("at least one decoder is required")
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        specs = (self._types._CDecoderSpec * len(decoders))(
            *(self._decoder_spec(item) for item in decoders)
        )
        handle = ctypes.c_void_p()
        status = self._lib.lc_decoder_bank_compile(
            specs, len(specs), node_count, ctypes.byref(handle)
        )
        if status != _LC_OK:
            self._raise(status)
        return CompiledDecoderBank(self, handle, tuple(decoders))

    def advance(self, model: ResolvedScalarLIF, state: ScalarState, t: float) -> ScalarState:
        """Advance one scalar state with the C evaluator."""

        c_model = self._model(model)
        c_state = self._types._CState(state.value, state.t_last)
        status = self._lib.lc_scalar_advance(
            ctypes.byref(c_model), ctypes.byref(c_state), self._types.time_value(t)
        )
        if status != _LC_OK:
            self._raise(status)
        return ScalarState(c_state.value, c_state.t_last)

    def advance_analytical(
        self,
        model: AnalyticalModel,
        state: AugmentedState,
        t: float,
        *,
        clamped: bool = False,
        parameter_bindings: Mapping[str, float] | None = None,
    ) -> AugmentedState:
        """Advance any resolved analytical state through its C expression program."""

        if not isinstance(
            model, _ANALYTICAL_MODEL_TYPES
        ):
            raise TypeError("model must be a resolved analytical node")
        if not isinstance(state, AugmentedState):
            raise TypeError("state must be an AugmentedState")
        dag = model.propagation_dag
        state_names = (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else tuple(model.state_names)
        )
        state_count = len(state_names)
        expected_variables = ("Delta",) + tuple(
            f"x{index}" for index in range(state_count)
        )
        if len(state.values) != state_count or dag.variables != expected_variables:
            raise ValueError(
                "analytical state and propagation variable layouts disagree"
            )
        root_names = model.clamped_roots if clamped else model.normal_roots
        if len(root_names) != state_count or any(
            name not in dag.roots for name in root_names
        ):
            raise ValueError("analytical propagation program is missing a state root")
        bindings = (
            model.bindings
            if parameter_bindings is None
            else dict(parameter_bindings)
        )
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, bindings)
        roots = (ctypes.c_uint32 * state_count)(
            *(dag.roots[name] for name in root_names)
        )
        values = (self._types.real_array(state_count))(*map(float, state.values))
        t_last = self._types.time_value(float(state.t_last))
        variables = (self._types.real_array(state_count + 1))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_state_advance(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            state_count,
            values,
            ctypes.byref(t_last),
            float(t),
            variables,
            len(variables),
            workspace,
            len(workspace),
        )
        if status != _LC_OK:
            self._raise(status)
        return AugmentedState(tuple(values), t_last.value)

    def advance_analytical_selected(
        self,
        model: AnalyticalModel,
        state: AugmentedState,
        t: float,
        state_indices: Sequence[int],
        *,
        clamped: bool = False,
        parameter_bindings: Mapping[str, float] | None = None,
    ) -> tuple[float, ...]:
        """Evaluate selected propagated state roots and only their DAG closure in C.

        This read-only inspection rounds clock operands before constructing the
        model-width Delta input. Half elapsed-time arithmetic also executes in C.
        """

        if not isinstance(
            model, _ANALYTICAL_MODEL_TYPES
        ):
            raise TypeError("model must be a resolved analytical node")
        if not isinstance(state, AugmentedState):
            raise TypeError("state must be an AugmentedState")
        dag = model.propagation_dag
        state_names = (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else tuple(model.state_names)
        )
        state_count = len(state_names)
        selected = tuple(state_indices)
        if not selected:
            return ()
        if (
            len(selected) != len(set(selected))
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < state_count
                for index in selected
            )
        ):
            raise ValueError("state indices must be unique valid local indices")
        expected_variables = ("Delta",) + tuple(
            f"x{index}" for index in range(state_count)
        )
        if len(state.values) != state_count or dag.variables != expected_variables:
            raise ValueError(
                "analytical state and propagation variable layouts disagree"
            )
        if not math.isfinite(state.t_last) or not math.isfinite(t):
            raise ValueError("analytical times must be finite")
        if t < state.t_last:
            raise CoreError("time reversed")
        if self.precision is PrecisionProfile.FLOAT64:
            delta = float(t) - float(state.t_last)
        else:
            try:
                target_time = self.precision.round_time(t, name="inspection time")
                target_last = self.precision.round_time(state.t_last, name="state time")
                if t > state.t_last and target_time <= target_last:
                    raise ValueError("positive inspection interval does not advance time")
                if self.precision is PrecisionProfile.FLOAT16:
                    elapsed = ExprDAG(
                        (ExprNode(ExprOp.VAR, binding=0),
                         ExprNode(ExprOp.VAR, binding=1),
                         ExprNode(ExprOp.SUB, lhs=0, rhs=1)),
                        (), ("time", "last"), {"elapsed": 2},
                    )
                    delta = self.evaluate_expr(
                        elapsed, variables={"time": target_time, "last": target_last},
                        roots=("elapsed",),
                    )["elapsed"]
                else:
                    delta = self.precision.round_real(
                        target_time - target_last, name="inspection interval",
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                raise PrecisionResolutionError(str(exc)) from exc
        root_names = model.clamped_roots if clamped else model.normal_roots
        if len(root_names) != state_count or any(
            name not in dag.roots for name in root_names
        ):
            raise ValueError("analytical propagation program is missing a state root")
        bindings = (
            model.bindings
            if parameter_bindings is None
            else dict(parameter_bindings)
        )
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, bindings)
        variable_array = (self._types.real_array(state_count + 1))(
            delta, *map(float, state.values)
        )
        roots = (ctypes.c_uint32 * len(selected))(
            *(dag.roots[root_names[index]] for index in selected)
        )
        outputs = (self._types.real_array(len(selected)))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        active = (ctypes.c_uint8 * len(dag.nodes))()
        status = self._lib.lc_expr_evaluate_selected(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            variable_array,
            len(variable_array),
            roots,
            len(roots),
            outputs,
            workspace,
            len(workspace),
            active,
            len(active),
        )
        if status != _LC_OK:
            self._raise(status)
        return tuple(outputs)

    def _expr_nodes(self, dag: ExprDAG):
        return (self._types._CExprNode * len(dag.nodes))(
            *(
                self._types._CExprNode(int(node.op), node.lhs, node.rhs, node.binding, node.value)
                for node in dag.nodes
            )
        )

    def _expr_parameters(self, dag: ExprDAG, bindings: Mapping[str, float]):
        values = dict(bindings)
        if not set(dag.parameters).issubset(values):
            missing = set(dag.parameters).difference(values)
            raise ValueError(
                "missing expression parameter binding(s): " + ", ".join(sorted(missing))
            )
        return (self._types.real_array(len(dag.parameters)))(
            *(float(values[name]) for name in dag.parameters)
        )

    def advance_alpha(
        self,
        model: ResolvedAlphaLIF,
        state: AugmentedState,
        t: float,
        *,
        clamped: bool = False,
    ) -> AugmentedState:
        """Advance one folded-alpha state through its lowered program in C."""

        dag = model.propagation_dag
        if len(state.values) != len(model.state_names):
            raise ValueError("augmented state length must match the resolved state layout")
        if dag.variables != ("Delta", "x0", "x1", "x2") or len(state.values) != 3:
            raise ValueError("folded-alpha propagation program has an invalid variable layout")
        root_names = model.clamped_roots if clamped else model.normal_roots
        if any(name not in dag.roots for name in root_names):
            raise ValueError("folded-alpha propagation program is missing a state root")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        roots = (ctypes.c_uint32 * len(root_names))(
            *(dag.roots[name] for name in root_names)
        )
        values = (self._types.real_array(len(state.values)))(*map(float, state.values))
        t_last = self._types.time_value(float(state.t_last))
        variables = (self._types.real_array(len(state.values) + 1))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_state_advance(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            len(state.values),
            values,
            ctypes.byref(t_last),
            float(t),
            variables,
            len(variables),
            workspace,
            len(workspace),
        )
        if status != _LC_OK:
            self._raise(status)
        return AugmentedState(tuple(values), t_last.value)

    def advance_adaptive(
        self,
        model: ResolvedAdaptiveLIF,
        state: AugmentedState,
        t: float,
        *,
        clamped: bool = False,
    ) -> AugmentedState:
        """Advance one adaptive state through its exact C-evaluated program."""

        dag = model.propagation_dag
        if len(state.values) != 2 or len(state.values) != len(model.state_names):
            raise ValueError("adaptive state must contain [v, w]")
        if dag.variables != ("Delta", "x0", "x1"):
            raise ValueError("adaptive propagation program has an invalid variable layout")
        root_names = model.clamped_roots if clamped else model.normal_roots
        if any(name not in dag.roots for name in root_names):
            raise ValueError("adaptive propagation program is missing a state root")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        roots = (ctypes.c_uint32 * len(root_names))(
            *(dag.roots[name] for name in root_names)
        )
        values = (self._types.real_array(2))(*map(float, state.values))
        t_last = self._types.time_value(float(state.t_last))
        variables = (self._types.real_array(3))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_state_advance(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            2,
            values,
            ctypes.byref(t_last),
            float(t),
            variables,
            3,
            workspace,
            len(workspace),
        )
        if status != _LC_OK:
            self._raise(status)
        return AugmentedState(tuple(values), t_last.value)

    def reset_adaptive(
        self, model: ResolvedAdaptiveLIF, state: AugmentedState
    ) -> AugmentedState:
        """Apply the adaptive neuron's simultaneous own-spike reset in C."""

        dag = model.propagation_dag
        if len(state.values) != 2 or dag.variables != ("Delta", "x0", "x1"):
            raise ValueError("adaptive reset requires a [v, w] state")
        if any(name not in dag.roots for name in model.reset_roots):
            raise ValueError("adaptive program is missing a reset root")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        roots = (ctypes.c_uint32 * 2)(
            *(dag.roots[name] for name in model.reset_roots)
        )
        values = (self._types.real_array(2))(*map(float, state.values))
        variables = (self._types.real_array(3))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_state_map(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            2,
            values,
            variables,
            3,
            workspace,
            len(workspace),
        )
        if status != _LC_OK:
            self._raise(status)
        return AugmentedState(tuple(values), state.t_last)

    def deposit_alpha(
        self,
        model: ResolvedAlphaLIF,
        state: AugmentedState,
        weight: float,
    ) -> AugmentedState:
        """Apply one normalized alpha deposit to the resolved target in C."""

        dag = model.deposit_dag
        if len(state.values) != len(model.state_names):
            raise ValueError("augmented state length must match the resolved state layout")
        if dag.variables != ("w",) or model.deposit_root not in dag.roots:
            raise ValueError("folded-alpha deposit program has an invalid layout")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        values = (self._types.real_array(len(state.values)))(*map(float, state.values))
        variables = (self._types.real_array(1))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_state_deposit(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            dag.roots[model.deposit_root],
            float(weight),
            values,
            len(state.values),
            model.deposit_index,
            variables,
            len(variables),
            workspace,
            len(workspace),
        )
        if status != _LC_OK:
            self._raise(status)
        return AugmentedState(tuple(values), state.t_last)

    def predict_alpha(
        self,
        model: ResolvedAlphaLIF,
        state: AugmentedState,
    ) -> AlphaPrediction:
        """Predict the earliest certified rising crossing for one alpha node."""

        dag = model.propagation_dag
        hint = model.root_hint
        if len(state.values) != 3 or len(state.values) != len(model.state_names):
            raise ValueError("folded-alpha prediction requires the resolved three-state layout")
        if dag.variables != ("Delta", "x0", "x1", "x2"):
            raise ValueError("folded-alpha crossing program has an invalid variable layout")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        values = (self._types.real_array(len(state.values)))(*map(float, state.values))
        result = self._types._CRootResult()
        variables = (self._types.real_array(4))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        if isinstance(hint, RootFindHint):
            root_names = (
                hint.g_root,
                hint.g_prime_root,
                hint.extremum_root,
                hint.extremum_prime_root,
                hint.asymptote_root,
                hint.membrane_coefficient_root,
                hint.synapse_constant_root,
                hint.synapse_linear_root,
                hint.membrane_rate_root,
                hint.synapse_rate_root,
                hint.threshold_root,
            )
            if any(name not in dag.roots for name in root_names):
                raise ValueError("folded-alpha crossing program is missing a required root")
            if hint.iteration_cap != 192:
                raise ValueError("resolved iteration cap does not match the C solver contract")
            c_hint = self._types._CRootHint(
                *(dag.roots[name] for name in root_names),
                hint.relative_tolerance,
                hint.fastest_time_constant,
            )
            status = self._lib.lc_expr_alpha_predict(
                node_array,
                len(dag.nodes),
                parameter_array,
                len(dag.parameters),
                ctypes.byref(c_hint),
                values,
                len(values),
                state.t_last,
                ctypes.byref(result),
                variables,
                len(variables),
                workspace,
                len(workspace),
            )
        else:
            flattened = [
                name
                for block in hint.coefficient_roots
                for name in block
            ]
            offsets = [0]
            for block in hint.coefficient_roots:
                offsets.append(offsets[-1] + len(block))
            if (
                len(hint.rate_roots) != len(hint.coefficient_roots)
                or not 0 < len(hint.rate_roots) <= 8
                or len(flattened) > 8
            ):
                raise ValueError("invalid folded-alpha exponential-polynomial layout")
            rates = [dag.roots[name] for name in hint.rate_roots]
            coefficients = [dag.roots[name] for name in flattened]
            rates.extend([0] * (8 - len(rates)))
            coefficients.extend([0] * (8 - len(coefficients)))
            offsets.extend([offsets[-1]] * (9 - len(offsets)))
            c_hint = self._types._CExpPolyHint(
                dag.roots[hint.limit_root],
                (ctypes.c_uint32 * 8)(*rates),
                (ctypes.c_uint32 * 8)(*coefficients),
                (ctypes.c_uint32 * 9)(*offsets),
                len(hint.rate_roots),
                len(flattened),
                hint.iteration_cap,
                hint.relative_tolerance,
                hint.fastest_time_constant,
            )
            status = self._lib.lc_expr_exp_poly_predict(
                node_array,
                len(dag.nodes),
                parameter_array,
                len(dag.parameters),
                ctypes.byref(c_hint),
                values,
                len(values),
                state.t_last,
                ctypes.byref(result),
                variables,
                len(variables),
                workspace,
                len(workspace),
            )
        diagnostics = RootDiagnostics(
            horizon=result.horizon,
            bracket_low=result.bracket_low,
            bracket_high=result.bracket_high,
            residual=result.residual,
            tolerance=result.tolerance,
            iterations=result.iterations,
            extrema_count=result.extrema_count,
        )
        if status == _LC_NO_CROSSING:
            return AlphaPrediction(DispatchForm.ROOT_FIND, None, diagnostics)
        if status == _LC_ROOT_NONCONVERGENCE:
            width = result.bracket_high - result.bracket_low
            raise CoreError(
                "certified root finder did not converge: "
                f"t={state.t_last}, bracket_width={width}, residual={result.residual}, "
                f"tolerance={result.tolerance}, iteration_cap={hint.iteration_cap}"
            )
        if status != _LC_OK:
            self._raise(status)
        return AlphaPrediction(DispatchForm.ROOT_FIND, result.t_spike, diagnostics)

    def predict_adaptive(
        self, model: ResolvedAdaptiveLIF, state: AugmentedState
    ) -> AdaptivePrediction:
        """Predict the earliest certified crossing of a two-mode adaptive node."""

        dag = model.propagation_dag
        hint = model.root_hint
        if len(state.values) != 2 or len(state.values) != len(model.state_names):
            raise ValueError("adaptive prediction requires a [v, w] state")
        if dag.variables != ("Delta", "x0", "x1"):
            raise ValueError("adaptive crossing program has an invalid variable layout")
        root_names = (
            hint.g_root,
            hint.g_prime_root,
            hint.limit_root,
            hint.coefficient_one_root,
            hint.coefficient_two_root,
            hint.rate_one_root,
            hint.rate_two_root,
        )
        if any(name not in dag.roots for name in root_names):
            raise ValueError("adaptive crossing program is missing a required root")
        if hint.iteration_cap != 192:
            raise ValueError("resolved iteration cap does not match the C solver contract")
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, model.bindings)
        c_hint = self._types._CTwoExpHint(
            *(dag.roots[name] for name in root_names),
            hint.relative_tolerance,
            hint.fastest_time_constant,
        )
        values = (self._types.real_array(2))(*map(float, state.values))
        result = self._types._CRootResult()
        variables = (self._types.real_array(3))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        status = self._lib.lc_expr_two_exp_predict(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            ctypes.byref(c_hint),
            values,
            2,
            state.t_last,
            ctypes.byref(result),
            variables,
            3,
            workspace,
            len(workspace),
        )
        diagnostics = RootDiagnostics(
            horizon=result.horizon,
            bracket_low=result.bracket_low,
            bracket_high=result.bracket_high,
            residual=result.residual,
            tolerance=result.tolerance,
            iterations=result.iterations,
            extrema_count=result.extrema_count,
        )
        if status == _LC_NO_CROSSING:
            return AdaptivePrediction(DispatchForm.ROOT_FIND, None, diagnostics)
        if status == _LC_ROOT_NONCONVERGENCE:
            width = result.bracket_high - result.bracket_low
            raise CoreError(
                "adaptive root finder did not converge: "
                f"t={state.t_last}, bracket_width={width}, residual={result.residual}, "
                f"tolerance={result.tolerance}, iteration_cap={hint.iteration_cap}"
            )
        if status != _LC_OK:
            self._raise(status)
        return AdaptivePrediction(DispatchForm.ROOT_FIND, result.t_spike, diagnostics)

    def _step_config(self, model: ResolvedSteppedNeuron) -> _CStepConfig:
        numerical = model.numerical
        return self._types._CStepConfig(
            numerical.relative_tolerance,
            numerical.absolute_tolerance,
            numerical.initial_step,
            numerical.minimum_step,
            numerical.maximum_step,
            numerical.event_tolerance,
            numerical.maximum_steps,
            numerical.maximum_rhs_evaluations,
        )

    @staticmethod
    def _step_diagnostics(result: _CStepResult) -> StepDiagnostics:
        return StepDiagnostics(
            result.t_reached,
            result.last_step,
            result.error_norm,
            result.accepted_steps,
            result.rejected_steps,
            result.rhs_evaluations,
            result.event_iterations,
        )

    def advance_stepped(
        self,
        model: ResolvedSteppedNeuron,
        state: AugmentedState,
        t: float,
        *,
        clamped: bool = False,
        parameter_bindings: Mapping[str, float] | None = None,
    ) -> tuple[AugmentedState, StepDiagnostics]:
        """Advance one generic ODE node through the allocation-free C stepper."""

        if not isinstance(model, ResolvedSteppedNeuron):
            raise TypeError("model must be a resolved stepped neuron")
        if not isinstance(state, AugmentedState):
            raise TypeError("state must be an AugmentedState")
        state_count = len(model.state_names)
        dag = model.propagation_dag
        expected_variables = ("Time",) + tuple(f"x{i}" for i in range(state_count))
        if len(state.values) != state_count or dag.variables != expected_variables:
            raise ValueError("stepped state and expression variable layouts disagree")
        bindings = model.bindings if parameter_bindings is None else dict(parameter_bindings)
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, bindings)
        roots = (ctypes.c_uint32 * state_count)(
            *(dag.roots[name] for name in model.normal_roots)
        )
        values = (self._types.real_array(state_count))(*map(float, state.values))
        t_last = self._types.time_value(float(state.t_last))
        variables = (self._types.real_array(state_count + 1))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        result = self._types._CStepResult()
        config = self._step_config(model)
        status = self._lib.lc_expr_step_advance(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            state_count,
            model.readout_index,
            ctypes.byref(config),
            values,
            ctypes.byref(t_last),
            float(t),
            int(bool(clamped)),
            variables,
            len(variables),
            workspace,
            len(workspace),
            ctypes.byref(result),
        )
        if status != _LC_OK:
            self._raise(status)
        return (
            AugmentedState(tuple(values), t_last.value),
            self._step_diagnostics(result),
        )

    def predict_stepped(
        self,
        model: ResolvedSteppedNeuron,
        state: AugmentedState,
        horizon: float,
        *,
        parameter_bindings: Mapping[str, float] | None = None,
    ) -> SteppedPrediction:
        """Predict the first rising fixed-threshold crossing up to ``horizon``."""

        if not isinstance(model, ResolvedSteppedNeuron):
            raise TypeError("model must be a resolved stepped neuron")
        if not isinstance(state, AugmentedState):
            raise TypeError("state must be an AugmentedState")
        state_count = len(model.state_names)
        dag = model.propagation_dag
        expected_variables = ("Time",) + tuple(f"x{i}" for i in range(state_count))
        if len(state.values) != state_count or dag.variables != expected_variables:
            raise ValueError("stepped state and expression variable layouts disagree")
        bindings = model.bindings if parameter_bindings is None else dict(parameter_bindings)
        node_array = self._expr_nodes(dag)
        parameter_array = self._expr_parameters(dag, bindings)
        roots = (ctypes.c_uint32 * state_count)(
            *(dag.roots[name] for name in model.normal_roots)
        )
        values = (self._types.real_array(state_count))(*map(float, state.values))
        variables = (self._types.real_array(state_count + 1))()
        workspace = (self._types.real_array(len(dag.nodes)))()
        result = self._types._CStepResult()
        config = self._step_config(model)
        status = self._lib.lc_expr_step_predict(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            roots,
            state_count,
            model.readout_index,
            model.threshold,
            ctypes.byref(config),
            values,
            float(state.t_last),
            float(horizon),
            variables,
            len(variables),
            workspace,
            len(workspace),
            ctypes.byref(result),
        )
        diagnostics = self._step_diagnostics(result)
        if status == _LC_NO_CROSSING:
            return SteppedPrediction(DispatchForm.STEPPED, None, diagnostics)
        if status != _LC_OK:
            self._raise(status)
        return SteppedPrediction(
            DispatchForm.STEPPED, result.t_crossing, diagnostics
        )

    def predict(self, model: ResolvedScalarLIF, state: ScalarState) -> Prediction:
        """Predict one scalar threshold crossing with the C evaluator."""

        c_model = self._model(model)
        c_state = self._types._CState(state.value, state.t_last)
        t_spike = self._types.time_value()
        dispatch = ctypes.c_int()
        status = self._lib.lc_scalar_predict(
            ctypes.byref(c_model),
            ctypes.byref(c_state),
            ctypes.byref(t_spike),
            ctypes.byref(dispatch),
        )
        if status == _LC_NO_CROSSING:
            return Prediction(DispatchForm.REACTIVE, None)
        if status != _LC_OK:
            self._raise(status)
        return Prediction(DispatchForm.CLOSED_FORM, t_spike.value)

    def _pack_mixed_graph(
        self,
        models: Sequence[ExecutableModel],
        edges: Sequence[MixedEdge],
        polarities: Sequence[NeuronPolarity] | None = None,
        plasticity: Sequence[PlasticityRule | None] | None = None,
        modulators: Sequence[int | None] | None = None,
        weight_groups: Sequence[int | None] | None = None,
    ):
        if not models:
            raise ValueError("mixed graph must contain at least one model")
        if polarities is None:
            normalized_polarities = (NeuronPolarity.EXCITATORY,) * len(models)
        else:
            normalized_polarities = tuple(polarities)
            if len(normalized_polarities) != len(models):
                raise ValueError("polarities must contain one value per model")
            if any(
                not isinstance(polarity, NeuronPolarity)
                for polarity in normalized_polarities
            ):
                raise ValueError(
                    "polarity must be EXCITATORY, INHIBITORY, or MIXED"
                )
        lowerings = tuple(
            _NodeLowering(
                program_identity=None,
                polarity=polarity,
                threshold=model.threshold,
                refractory=model.refractory,
                dispatch=model.dispatch,
                readout_index=model.readout_index,
                propagation_dag=model.propagation_dag,
                normal_roots=tuple(model.normal_roots),
                clamped_roots=tuple(model.clamped_roots),
                reset_roots=tuple(model.reset_roots),
                parameter_values=tuple(
                    float(model.bindings[name])
                    for name in model.propagation_dag.parameters
                ),
                root_hint=getattr(model, "root_hint", None),
                numerical=getattr(model, "numerical", None),
                event_batched=getattr(model, "reactive_mode", None) is not None,
                reset_before_deposit=(
                    getattr(getattr(model, "reactive_mode", None), "value", None)
                    == "RESET_BEFORE_DEPOSIT"
                ),
                scalar_affine=_scalar_affine_lowering(model),
                hazard=getattr(model, "hazard", None),
                deposit_dag=getattr(model, "deposit_dag", None),
                deposit_root=getattr(model, "deposit_root", None),
                deposit_index=getattr(model, "deposit_index", None),
            )
            for model, polarity in zip(models, normalized_polarities)
        )
        return self._pack_lowered_graph(
            lowerings,
            edges,
            plasticity,
            modulators,
            weight_groups,
        )

    def _pack_lowered_graph(
        self,
        models: Sequence[_NodeLowering],
        edges: Sequence[MixedEdge],
        plasticity: Sequence[PlasticityRule | None] | None = None,
        modulators: Sequence[int | None] | None = None,
        weight_groups: Sequence[int | None] | None = None,
        *,
        include_legacy_plasticity: bool = True,
    ):
        if not models:
            raise ValueError("mixed graph must contain at least one equation lowering")
        if any(count > 0xFFFFFFFF for count in (len(models), len(edges))):
            raise ValueError("mixed graph counts must fit uint32")
        if any(
            not isinstance(model.polarity, NeuronPolarity) for model in models
        ):
            raise ValueError("invalid neuron polarity in graph lowering")
        if any(
            not 0 <= edge.pre < len(models) or not 0 <= edge.post < len(models)
            for edge in edges
        ):
            raise ValueError("mixed edge references an invalid node")
        if any(
            not math.isfinite(edge.weight)
            or (
                edge.weight < 0.0
                and models[edge.pre].polarity is not NeuronPolarity.MIXED
            )
            or not math.isfinite(edge.deposit_scale)
            or edge.deposit_scale <= 0.0
            for edge in edges
        ):
            raise ValueError(
                "mixed edge weights must be finite with positive deposit scales, "
                "and signed weights require a MIXED presynaptic neuron"
            )
        if not include_legacy_plasticity:
            if plasticity is not None or modulators is not None or weight_groups is not None:
                raise ValueError(
                    "legacy plasticity descriptors cannot accompany plan-native packing"
                )
            normalized_plasticity = ()
            normalized_modulators = ()
            normalized_weight_groups = ()
            dense_weight_groups: dict[int, int] = {}
        else:
            normalized_plasticity = (
                (None,) * len(edges) if plasticity is None else tuple(plasticity)
            )
            normalized_modulators = (
                (None,) * len(edges) if modulators is None else tuple(modulators)
            )
            normalized_weight_groups = (
                (None,) * len(edges) if weight_groups is None else tuple(weight_groups)
            )
            if (
                len(normalized_plasticity) != len(edges)
                or len(normalized_modulators) != len(edges)
                or len(normalized_weight_groups) != len(edges)
            ):
                raise ValueError(
                    "plasticity, modulator, and weight-group layouts must match the edge layout"
                )
            if any(
                rule is not None
                and models[edge.pre].polarity is NeuronPolarity.MIXED
                for edge, rule in zip(edges, normalized_plasticity)
            ):
                raise ValueError(
                    "online plasticity is not supported for "
                    "MIXED presynaptic neurons"
                )
            if any(
                group is not None
                and (
                    not isinstance(group, int)
                    or isinstance(group, bool)
                    or group < 0
                )
                for group in normalized_weight_groups
            ):
                raise ValueError("weight groups must be nonnegative integers or None")
            group_ids = sorted(
                {group for group in normalized_weight_groups if group is not None}
            )
            dense_weight_groups = {
                group: index + 1 for index, group in enumerate(group_ids)
            }
            grouped_members: dict[int, list[int]] = {}
            for index, group in enumerate(normalized_weight_groups):
                if group is not None:
                    grouped_members.setdefault(group, []).append(index)
            for group, members in grouped_members.items():
                reference = members[0]
                for member in members[1:]:
                    if (
                        edges[member].weight != edges[reference].weight
                        or normalized_plasticity[member]
                        != normalized_plasticity[reference]
                        or normalized_modulators[member]
                        != normalized_modulators[reference]
                        or models[edges[member].pre].polarity
                        is not models[edges[reference].pre].polarity
                    ):
                        raise ValueError(
                            f"weight group {group} must use one initial weight, "
                            "polarity, plasticity rule, and modulator"
                        )
        descriptors: list[self._types._CMixedNode] = []
        program_storage = []
        deposit_storage = []
        program_cache: dict[int, object] = {}
        deposit_cache: dict[int, object] = {}
        parameter_values: list[float] = []
        state_layouts: list[tuple[int, int]] = []
        state_offset = 0
        parameter_offset = 0
        dispatch_values = {
            DispatchForm.REACTIVE: 0,
            DispatchForm.CLOSED_FORM: 1,
            DispatchForm.ROOT_FIND: 2,
            DispatchForm.STEPPED: 3,
        }

        def root_array(dag: ExprDAG, names: Sequence[str]):
            """Map named roots into a fixed-width C array."""

            if len(names) > _LC_ANALYTICAL_MAX_STATES:
                raise ValueError("analytical state count exceeds the bounded C descriptor")
            values = [dag.roots[name] for name in names]
            values.extend([0] * (_LC_ANALYTICAL_MAX_STATES - len(values)))
            return (ctypes.c_uint32 * _LC_ANALYTICAL_MAX_STATES)(*values)

        def exp_poly_hint(dag: ExprDAG, hint: ExpPolyRootHint) -> _CExpPolyHint:
            """Pack one bounded exponential-polynomial crossing hint."""

            block_count = len(hint.rate_roots)
            if block_count != len(hint.coefficient_roots) or not 0 < block_count <= 8:
                raise ValueError("invalid bounded exponential-polynomial block layout")
            flattened: list[int] = []
            offsets = [0]
            for names in hint.coefficient_roots:
                if not names:
                    raise ValueError("exponential-polynomial blocks cannot be empty")
                flattened.extend(dag.roots[name] for name in names)
                offsets.append(len(flattened))
            if len(flattened) > 8:
                raise ValueError("exponential-polynomial coefficient bound exceeded")
            rates = [dag.roots[name] for name in hint.rate_roots]
            rates.extend([0] * (8 - len(rates)))
            flattened.extend([0] * (8 - len(flattened)))
            offsets.extend([offsets[-1]] * (9 - len(offsets)))
            return self._types._CExpPolyHint(
                dag.roots[hint.limit_root],
                (ctypes.c_uint32 * 8)(*rates),
                (ctypes.c_uint32 * 8)(*flattened),
                (ctypes.c_uint32 * 9)(*offsets),
                block_count,
                offsets[block_count],
                hint.iteration_cap,
                hint.relative_tolerance,
                hint.fastest_time_constant,
            )

        for model in models:
            descriptor = self._types._CMixedNode()
            descriptor.polarity = model.polarity.runtime_code
            descriptor.reset_before_deposit = 0
            descriptor.arithmetic_kind = 0
            descriptor.state_offset = state_offset
            descriptor.parameter_offset = parameter_offset
            descriptor.threshold = model.threshold
            descriptor.refractory = model.refractory
            descriptor.dispatch = dispatch_values[model.dispatch]
            descriptor.readout = model.readout_index
            if model.scalar_affine is not None:
                descriptor.arithmetic_kind = 1
                descriptor.scalar_affine_decay = model.scalar_affine.decay
                descriptor.scalar_affine_drive = model.scalar_affine.drive
                descriptor.scalar_affine_reset = model.scalar_affine.reset
            dag = model.propagation_dag
            program_array = (
                program_cache.get(model.program_identity)
                if model.program_identity is not None
                else None
            )
            if program_array is None:
                program_array = self._expr_nodes(dag)
                program_storage.append(program_array)
                if model.program_identity is not None:
                    program_cache[model.program_identity] = program_array
            descriptor.program_nodes = program_array
            descriptor.program_node_count = len(dag.nodes)
            descriptor.state_count = len(model.normal_roots)
            state_count = descriptor.state_count
            descriptor.parameter_count = len(dag.parameters)
            descriptor.normal_roots = root_array(dag, model.normal_roots)
            descriptor.clamped_roots = root_array(dag, model.clamped_roots)
            descriptor.reset_roots = root_array(dag, model.reset_roots)
            deposit_dag = getattr(model, "deposit_dag", None)
            if deposit_dag is not None:
                if deposit_dag.parameters != dag.parameters:
                    raise ValueError(
                        "propagation and deposit bindings must share one layout"
                    )
                deposit_array = (
                    deposit_cache.get(model.program_identity)
                    if model.program_identity is not None
                    else None
                )
                if deposit_array is None:
                    deposit_array = self._expr_nodes(deposit_dag)
                    deposit_storage.append(deposit_array)
                    if model.program_identity is not None:
                        deposit_cache[model.program_identity] = deposit_array
                descriptor.deposit_nodes = deposit_array
                descriptor.deposit_node_count = len(deposit_dag.nodes)
                if model.deposit_root is None or model.deposit_index is None:
                    raise ValueError("deposit equation requires a root and target")
                descriptor.deposit_root = deposit_dag.roots[model.deposit_root]
                descriptor.deposit_target = model.deposit_index

            hint = model.root_hint
            if model.hazard is not None:
                hazard = model.hazard
                descriptor.crossing_kind = 8
                coefficient_roots = [
                    dag.roots[name]
                    for name in hazard.trajectory_coefficient_roots
                ]
                rate_roots = [
                    dag.roots[name] for name in hazard.trajectory_rate_roots
                ]
                mode_count = len(coefficient_roots)
                if mode_count != len(rate_roots) or mode_count > 8:
                    raise ValueError("invalid bounded hazard trajectory layout")
                if mode_count and hazard.trajectory_limit_root is None:
                    raise ValueError("hazard trajectory modes require a limit root")
                coefficient_roots.extend([0] * (8 - mode_count))
                rate_roots.extend([0] * (8 - mode_count))
                descriptor.hazard = self._types._CHazardConfig(
                    1,
                    mode_count,
                    0
                    if hazard.trajectory_limit_root is None
                    else dag.roots[hazard.trajectory_limit_root],
                    (ctypes.c_uint32 * 8)(*coefficient_roots),
                    (ctypes.c_uint32 * 8)(*rate_roots),
                    hazard.log_scale,
                    hazard.voltage_gain,
                    hazard.relative_tolerance,
                    hazard.absolute_tolerance,
                    hazard.time_tolerance,
                    hazard.maximum_quadrature_depth,
                    hazard.maximum_root_iterations,
                )
            elif model.event_batched:
                descriptor.crossing_kind = 7
                descriptor.reset_before_deposit = int(model.reset_before_deposit)
            elif model.numerical is not None:
                descriptor.crossing_kind = 6
                numerical = model.numerical
                descriptor.step_config = self._types._CStepConfig(
                    numerical.relative_tolerance,
                    numerical.absolute_tolerance,
                    numerical.initial_step,
                    numerical.minimum_step,
                    numerical.maximum_step,
                    numerical.event_tolerance,
                    numerical.maximum_steps,
                    numerical.maximum_rhs_evaluations,
                )
            elif isinstance(hint, ScalarLogRootHint):
                descriptor.crossing_kind = 0
                descriptor.scalar_log_hint = self._types._CScalarLogHint(
                    dag.roots[hint.decay_root],
                    dag.roots[hint.affine_root],
                    dag.roots[hint.threshold_root],
                )
            elif isinstance(hint, TwoExpRootHint):
                descriptor.crossing_kind = 2
                root_names = (
                    hint.g_root,
                    hint.g_prime_root,
                    hint.limit_root,
                    hint.coefficient_one_root,
                    hint.coefficient_two_root,
                    hint.rate_one_root,
                    hint.rate_two_root,
                )
                descriptor.two_exp_hint = self._types._CTwoExpHint(
                    *(dag.roots[name] for name in root_names),
                    hint.relative_tolerance,
                    hint.fastest_time_constant,
                )
            elif isinstance(hint, RootFindHint):
                descriptor.crossing_kind = 1
                root_names = (
                    hint.g_root,
                    hint.g_prime_root,
                    hint.extremum_root,
                    hint.extremum_prime_root,
                    hint.asymptote_root,
                    hint.membrane_coefficient_root,
                    hint.synapse_constant_root,
                    hint.synapse_linear_root,
                    hint.membrane_rate_root,
                    hint.synapse_rate_root,
                    hint.threshold_root,
                )
                descriptor.root_hint = self._types._CRootHint(
                    *(dag.roots[name] for name in root_names),
                    hint.relative_tolerance,
                    hint.fastest_time_constant,
                )
            elif isinstance(hint, MultiExpRootHint):
                descriptor.crossing_kind = 4
                roots = [dag.roots[name] for name in hint.coefficient_roots]
                rates = [dag.roots[name] for name in hint.rate_roots]
                mode_count = len(roots)
                if mode_count != len(rates) or not 0 < mode_count <= 8:
                    raise ValueError("invalid bounded multi-exponential root layout")
                roots.extend([0] * (8 - mode_count))
                rates.extend([0] * (8 - mode_count))
                descriptor.multi_exp_hint = self._types._CMultiExpHint(
                    dag.roots[hint.limit_root],
                    (ctypes.c_uint32 * 8)(*roots),
                    (ctypes.c_uint32 * 8)(*rates),
                    mode_count,
                    hint.iteration_cap,
                    hint.relative_tolerance,
                    hint.fastest_time_constant,
                )
            elif isinstance(hint, ExpPolyRootHint):
                descriptor.crossing_kind = 3 if len(hint.rate_roots) == 1 else 5
                descriptor.exp_poly_hint = exp_poly_hint(dag, hint)
            else:
                raise TypeError("equation program has no supported crossing description")
            if len(model.parameter_values) != len(dag.parameters):
                raise ValueError("equation parameter values do not match the program")
            parameter_values.extend(model.parameter_values)
            parameter_offset += len(dag.parameters)
            state_layouts.append((state_offset, state_count))
            state_offset += state_count
            descriptors.append(descriptor)
        descriptor_array = (self._types._CMixedNode * len(descriptors))(*descriptors)
        parameter_array = (self._types.real_array(len(parameter_values)))(*parameter_values)
        edge_array = (self._types._CMixedEdge * len(edges))(
            *(
                self._types._CMixedEdge(
                    edge.pre,
                    edge.post,
                    int(edge.deposit_kind),
                    edge.target,
                    edge.weight,
                    edge.delay,
                    edge.deposit_scale,
                )
                for edge in edges
            )
        )
        if not include_legacy_plasticity:
            return (
                descriptor_array,
                parameter_array,
                edge_array,
                tuple(state_layouts),
                program_storage,
                deposit_storage,
                None,
            )
        rules = []
        for edge, rule, modulator, weight_group in zip(
            edges,
            normalized_plasticity,
            normalized_modulators,
            normalized_weight_groups,
        ):
            values = dict(
                kind=0,
                modulator=0,
                consume_on_modulation=0,
                weight_group=(
                    0 if rule is None or weight_group is None
                    else dense_weight_groups[weight_group]
                ),
                tau_pre=0.0,
                tau_post=0.0,
                tau_pre_slow=0.0,
                tau_post_slow=0.0,
                a2_plus=0.0,
                a2_minus=0.0,
                a3_plus=0.0,
                a3_minus=0.0,
                tau_eligibility_plus=0.0,
                tau_eligibility_minus=0.0,
                positive_plus=0.0,
                positive_minus=0.0,
                negative_plus=0.0,
                negative_minus=0.0,
                learning_rate=0.0,
                weight_min=0.0,
                weight_max=0.0,
            )
            if isinstance(rule, PairSTDP):
                values.update(
                    kind=1,
                    tau_pre=rule.tau_pre,
                    tau_post=rule.tau_post,
                    a2_plus=rule.a_plus,
                    a2_minus=rule.a_minus,
                    learning_rate=rule.learning_rate,
                    weight_min=rule.bounds[0],
                    weight_max=rule.bounds[1],
                )
            elif isinstance(rule, TripletSTDP):
                values.update(
                    kind=2,
                    tau_pre=rule.tau_plus,
                    tau_post=rule.tau_minus,
                    tau_pre_slow=rule.tau_x,
                    tau_post_slow=rule.tau_y,
                    a2_plus=rule.a2_plus,
                    a2_minus=rule.a2_minus,
                    a3_plus=rule.a3_plus,
                    a3_minus=rule.a3_minus,
                    learning_rate=rule.learning_rate,
                    weight_min=rule.bounds[0],
                    weight_max=rule.bounds[1],
                )
            elif isinstance(rule, ModulatedSTDP):
                if not isinstance(modulator, int) or isinstance(modulator, bool) or modulator < 0:
                    raise ValueError("ModulatedSTDP edges require a nonnegative modulator index")
                values.update(
                    kind=3,
                    modulator=modulator,
                    consume_on_modulation=int(rule.consume_on_modulation),
                    tau_pre=rule.tau_pre,
                    tau_post=rule.tau_post,
                    tau_eligibility_plus=rule.tau_eligibility_plus,
                    tau_eligibility_minus=rule.tau_eligibility_minus,
                    positive_plus=rule.positive_plus,
                    positive_minus=rule.positive_minus,
                    negative_plus=rule.negative_plus,
                    negative_minus=rule.negative_minus,
                    learning_rate=rule.learning_rate,
                    weight_min=rule.bounds[0],
                    weight_max=rule.bounds[1],
                )
            elif rule is not None:
                raise TypeError(f"unsupported plasticity rule {type(rule).__name__}")
            if rule is not None and not rule.bounds[0] <= edge.weight <= rule.bounds[1]:
                raise ValueError("initial edge weight lies outside plasticity bounds")
            rules.append(self._types._CPlasticityRule(**values))
        plasticity_array = (self._types._CPlasticityRule * len(rules))(*rules)
        return (
            descriptor_array,
            parameter_array,
            edge_array,
            tuple(state_layouts),
            program_storage,
            deposit_storage,
            plasticity_array,
        )

    def _pack_execution_plan(self, plan: ExecutionPlan):
        """Pack the shared C scheduler exclusively from equation-plan records."""

        if not plan.nodes:
            raise ValueError("execution plan must contain at least one node")
        lowerings = []
        for node in plan.nodes:
            program = plan.programs[node.program]
            if program.evolution.value == "EVENT_BATCHED":
                dispatch = DispatchForm.REACTIVE
            elif program.evolution.value == "NUMERICAL_ODE":
                dispatch = DispatchForm.STEPPED
            elif not node.autonomous_crossing:
                dispatch = DispatchForm.REACTIVE
            elif program.crossing.value == "SCALAR_LOG":
                dispatch = DispatchForm.CLOSED_FORM
            else:
                dispatch = DispatchForm.ROOT_FIND
            deposit_dag = None
            deposit_root = None
            deposit_index = None
            if program.deposit is not None:
                if node.deposit_parameter_values != node.parameter_values:
                    raise ValueError(
                        "compiled lowering requires shared propagation/deposit bindings"
                    )
                deposit_dag = program.deposit.expressions
                deposit_root = program.deposit.root
                deposit_index = program.deposit.target_state
            lowerings.append(_NodeLowering(
                program_identity=node.program,
                polarity=node.polarity,
                threshold=node.threshold,
                refractory=node.refractory,
                dispatch=dispatch,
                readout_index=node.readout_index,
                propagation_dag=program.propagation,
                normal_roots=tuple(program.normal_roots),
                clamped_roots=tuple(program.clamped_roots),
                reset_roots=tuple(program.reset_roots),
                parameter_values=node.parameter_values,
                root_hint=node.crossing_hint,
                numerical=node.numerical,
                event_batched=program.evolution.value == "EVENT_BATCHED",
                reset_before_deposit=(
                    getattr(node.reactive_mode, "value", None)
                    == "RESET_BEFORE_DEPOSIT"
                ),
                scalar_affine=(
                    node.scalar_affine
                    if node.arithmetic is ArithmeticMethod.SCALAR_AFFINE
                    else None
                ),
                hazard=node.hazard,
                deposit_dag=deposit_dag,
                deposit_root=deposit_root,
                deposit_index=deposit_index,
            ))

        edges = tuple(
            MixedEdge(
                connection.pre,
                connection.post,
                connection.weight,
                connection.delay,
                (
                    DepositKind.STATE_ADD
                    if connection.operation.value == "ADD_STATE"
                    else DepositKind.PROGRAM
                ),
                connection.target_state,
                connection.scale,
            )
            for connection in plan.connections
        )
        packed = self._pack_lowered_graph(
            lowerings,
            edges,
            None,
            None,
            None,
            include_legacy_plasticity=False,
        )
        learning = self._pack_learning_plan(plan)
        return packed + learning

    def _pack_learning_plan(self, plan: ExecutionPlan):
        """Pack structural trace/event programs without inspecting rule classes."""

        event_indices = {
            LearningEvent.PRE_SPIKE: 0,
            LearningEvent.POST_SPIKE: 1,
            LearningEvent.MODULATION_POSITIVE: 2,
            LearningEvent.MODULATION_NEGATIVE: 3,
            LearningEvent.OBSERVATION: 4,
        }
        canonical_storage = {
            "pre_fast": 0,
            "post_fast": 1,
            "pre_slow": 2,
            "post_slow": 3,
            "eligibility_plus": 4,
            "eligibility_minus": 5,
            "voltage_eligibility": 2,
            "voltage_sum": 1,
            "voltage_mass": 3,
            "soft_eligibility": 2,
            "trace_eligibility": 4,
        }
        learning_expression_storage: list[object] = []
        packed_programs: list[self._types._CLearningProgram] = []
        for program in plan.learning_programs:
            if len(program.traces) > _LC_LEARNING_MAX_TRACES:
                raise ValueError("learning trace count exceeds the bounded C runtime")
            base_variables = (
                "weight",
                "modulation",
                "learning_scale",
                "post_readout",
            )
            if program.observer is not None:
                base_variables += ("observation_gain", "event_amplitude")
            if "input_accepted" in program.variable_names:
                base_variables += ("input_accepted",)
            if program.variable_names != base_variables + tuple(
                trace.name for trace in program.traces
            ):
                raise ValueError("learning variable layout is not canonical")
            used_storage: set[int] = set()
            trace_storage: list[int] = []
            for trace in program.traces:
                slot = canonical_storage.get(trace.name)
                if slot is None or slot in used_storage:
                    slot = next(
                        (
                            candidate
                            for candidate in range(_LC_LEARNING_MAX_TRACES)
                            if candidate not in used_storage
                        ),
                        None,
                    )
                if slot is None:
                    raise ValueError("learning trace storage bound exceeded")
                used_storage.add(slot)
                trace_storage.append(slot)
            trace_tau = [trace.tau_parameter for trace in program.traces]
            trace_tau.extend([0] * (_LC_LEARNING_MAX_TRACES - len(trace_tau)))
            trace_storage.extend(
                [0] * (_LC_LEARNING_MAX_TRACES - len(trace_storage))
            )
            trace_names = tuple(trace.name for trace in program.traces)
            compatibility_kind = {
                ("pre_fast", "post_fast"): 1,
                ("pre_fast", "post_fast", "pre_slow", "post_slow"): 2,
                (
                    "pre_fast",
                    "post_fast",
                    "eligibility_plus",
                    "eligibility_minus",
                ): 3,
                (
                    "pre_fast",
                    "post_fast",
                    "eligibility_plus",
                    "eligibility_minus",
                    "voltage_eligibility",
                ): 3,
                (
                    "pre_fast",
                    "voltage_sum",
                    "voltage_mass",
                    "soft_eligibility",
                ): 3,
            }.get(trace_names, 0)
            packed_events = [
                self._types._CLearningEventProgram(weight_root=_UINT32_MAX)
                for _ in range(_LC_LEARNING_EVENT_COUNT)
            ]
            seen_events: set[LearningEvent] = set()
            for event in program.events:
                if event.event in seen_events:
                    raise ValueError("learning program repeats an event map")
                seen_events.add(event.event)
                dag = event.expressions
                if dag.parameters != program.parameter_names or (
                    dag.variables != program.variable_names
                ):
                    raise ValueError("learning event bindings do not match its program")
                nodes = self._expr_nodes(dag)
                learning_expression_storage.append(nodes)
                indices = [index for index, _ in event.trace_roots]
                roots = [dag.roots[root] for _, root in event.trace_roots]
                indices.extend([0] * (_LC_LEARNING_MAX_TRACES - len(indices)))
                roots.extend([0] * (_LC_LEARNING_MAX_TRACES - len(roots)))
                advance_mask = sum(1 << index for index in event.advance_traces)
                variable_mask = sum(
                    1 << node.binding
                    for node in dag.nodes
                    if node.op is ExprOp.VAR
                )
                packed_events[event_indices[event.event]] = self._types._CLearningEventProgram(
                    nodes,
                    len(dag.nodes),
                    advance_mask,
                    variable_mask,
                    (
                        _UINT32_MAX
                        if event.weight_root is None
                        else dag.roots[event.weight_root]
                    ),
                    len(event.trace_roots),
                    (ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES)(*indices),
                    (ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES)(*roots),
                )
            if program.observer is None:
                packed_observer = self._types._CLearningObserverProgram()
            else:
                observer = program.observer
                dag = observer.expressions
                if dag.parameters != program.parameter_names or (
                    dag.variables != observer.variable_names
                ):
                    raise ValueError(
                        "learning observer bindings do not match its program"
                    )
                observer_nodes = self._expr_nodes(dag)
                learning_expression_storage.append(observer_nodes)
                observer_variable_mask = sum(
                    1 << node.binding
                    for node in dag.nodes
                    if node.op is ExprOp.VAR
                )
                observer_parameter_mask = sum(
                    1 << node.binding
                    for node in dag.nodes
                    if node.op is ExprOp.PARAM
                )
                for parameter_index in (
                    observer.voltage_tau_parameter,
                    observer.fast_activity_tau_parameter,
                    observer.slow_activity_tau_parameter,
                    observer.band_width_parameter,
                ):
                    observer_parameter_mask |= 1 << parameter_index
                packed_observer = self._types._CLearningObserverProgram(
                    observer_nodes,
                    len(dag.nodes),
                    observer_variable_mask,
                    observer_parameter_mask,
                    observer.voltage_tau_parameter,
                    observer.fast_activity_tau_parameter,
                    observer.slow_activity_tau_parameter,
                    observer.band_width_parameter,
                    dag.roots[observer.fast_activity_root],
                    dag.roots[observer.slow_activity_root],
                    dag.roots[observer.gain_root],
                )
            packed_programs.append(
                self._types._CLearningProgram(
                    len(program.parameter_names),
                    len(program.variable_names),
                    len(program.traces),
                    int(program.clamp_normalized_weight),
                    compatibility_kind,
                    (ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES)(*trace_tau),
                    (ctypes.c_uint32 * _LC_LEARNING_MAX_TRACES)(*trace_storage),
                    (self._types._CLearningEventProgram * _LC_LEARNING_EVENT_COUNT)(
                        *packed_events
                    ),
                    packed_observer,
                )
            )

        group_ids = sorted(
            {
                connection.weight_group
                for connection in plan.connections
                if connection.weight_group is not None
                and connection.learning_program is not None
            }
        )
        dense_groups = {group: index + 1 for index, group in enumerate(group_ids)}
        group_reference: dict[int, tuple[object, ...]] = {}
        learning_parameters: list[float] = []
        bindings: list[self._types._CLearningBinding] = []
        for connection in plan.connections:
            parameter_offset = len(learning_parameters)
            if connection.learning_program is None:
                if connection.modulator is not None:
                    raise ValueError("modulators require a learning program")
                bindings.append(
                    self._types._CLearningBinding(
                        _UINT32_MAX,
                        parameter_offset,
                        _UINT32_MAX,
                        0,
                        0.0,
                        0.0,
                    )
                )
                continue
            if plan.nodes[connection.pre].polarity is NeuronPolarity.MIXED:
                raise ValueError(
                    "online plasticity is not supported for "
                    "MIXED presynaptic neurons"
                )
            if connection.learning_weight_bounds is None:
                raise ValueError("learning connection has no weight bounds")
            program = plan.learning_programs[connection.learning_program]
            if len(connection.learning_parameter_values) != len(
                program.parameter_names
            ):
                raise ValueError("learning parameter values do not match the program")
            lower, upper = connection.learning_weight_bounds
            if not lower <= connection.weight <= upper:
                raise ValueError("initial edge weight lies outside learning bounds")
            learning_parameters.extend(connection.learning_parameter_values)
            encoded_group = (
                0
                if connection.weight_group is None
                else dense_groups[connection.weight_group]
            )
            if connection.weight_group is not None:
                signature = (
                    connection.learning_program,
                    connection.learning_parameter_values,
                    connection.learning_weight_bounds,
                    connection.weight,
                )
                previous = group_reference.setdefault(
                    connection.weight_group, signature
                )
                if previous != signature:
                    raise ValueError(
                        f"weight group {connection.weight_group} must share one "
                        "learning program, parameters, bounds, and weight"
                    )
            bindings.append(
                self._types._CLearningBinding(
                    connection.learning_program,
                    parameter_offset,
                    (
                        _UINT32_MAX
                        if connection.modulator is None
                        else connection.modulator
                    ),
                    encoded_group,
                    lower,
                    upper,
                )
            )
        return (
            (self._types._CLearningProgram * len(packed_programs))(*packed_programs),
            (self._types.real_array(len(learning_parameters)))(*learning_parameters),
            (self._types._CLearningBinding * len(bindings))(*bindings),
            learning_expression_storage,
        )

    @staticmethod
    def _runtime_layouts_from_models(
        models: Sequence[ExecutableModel],
    ) -> tuple[_RuntimeNodeLayout, ...]:
        layouts = []
        for model in models:
            dag = model.propagation_dag
            drives = []
            for name in model.drive_parameters:
                aliases = (
                    (name, name.split(".", 1)[1])
                    if name.startswith("neuron.")
                    else (name,)
                )
                drives.append(
                    _RuntimeDriveBinding(
                        aliases,
                        dag.parameters.index(name),
                        model.parameter_domains[name],
                    )
                )
            layouts.append(_RuntimeNodeLayout(len(model.normal_roots), tuple(drives)))
        return tuple(layouts)

    @staticmethod
    def _runtime_layouts_from_plan(
        plan: ExecutionPlan,
    ) -> tuple[_RuntimeNodeLayout, ...]:
        return tuple(
            _RuntimeNodeLayout(
                node.state_count,
                tuple(
                    _RuntimeDriveBinding(
                        binding.aliases,
                        binding.parameter_index,
                        binding.domain,
                    )
                    for binding in plan.programs[node.program].drive_bindings
                ),
            )
            for node in plan.nodes
        )

    def _pack_mixed_initial(
        self, layouts: Sequence[_RuntimeNodeLayout],
        initial_states: Sequence[float | Sequence[float]],
    ):
        if len(layouts) != len(initial_states):
            raise ValueError("node layouts and initial states must have the same length")
        flat_state: list[float] = []
        for layout, initial in zip(layouts, initial_states):
            state_count = layout.state_count
            values = (
                (float(initial),) + (0.0,) * (state_count - 1)
                if isinstance(initial, (int, float))
                else tuple(float(value) for value in initial)
            )
            if len(values) != state_count:
                raise ValueError(
                    f"analytical initial state must contain {state_count} value(s)"
                )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("initial state values must be finite")
            flat_state.extend(values)
        return (
            (self._types.real_array(len(flat_state)))(*flat_state),
            (self._types.time_array(len(layouts)))(*([0.0] * len(layouts))),
        )

    def _pack_mixed_inputs(self, inputs: Sequence[MixedInputSpike]):
        if len(inputs) > 0xFFFFFFFF:
            raise ValueError("mixed input count must fit uint32")
        return (self._types._CMixedInputSpike * len(inputs))(
            *(
                self._types._CMixedInputSpike(
                    event.t,
                    event.node,
                    int(event.deposit_kind),
                    event.target,
                    event.value,
                )
                for event in inputs
            )
        )

    def _pack_mixed_drives(
        self, layouts: Sequence[_RuntimeNodeLayout],
        drive_updates: Sequence[MixedDriveUpdate],
    ):
        if len(drive_updates) > 0xFFFFFFFF:
            raise ValueError("mixed drive update count must fit uint32")
        c_drive_updates = []
        for update in drive_updates:
            if update.node < 0 or update.node >= len(layouts):
                raise ValueError("mixed drive update references an invalid node")
            layout = layouts[update.node]
            value = float(update.value)
            if not math.isfinite(value):
                raise ValueError("mixed drive update values must be finite")
            if update.binding is None:
                raise ValueError("analytical drive updates require a parameter binding")
            match = next(
                (binding for binding in layout.drives if update.binding in binding.aliases),
                None,
            )
            if match is None:
                raise ValueError("drive binding is not an approved runtime parameter")
            if match.domain is ParameterDomain.POSITIVE and value <= 0.0:
                raise ValueError("positive drive parameters must remain positive")
            c_drive_updates.append(
                self._types._CMixedDriveUpdate(update.t, update.node, match.parameter_index, value)
            )
        return (self._types._CMixedDriveUpdate * len(c_drive_updates))(*c_drive_updates)

    def _pack_modulations(self, events: Sequence[ModulationEvent]):
        items = tuple(events)
        if len(items) > 0xFFFFFFFF or any(
            not isinstance(item, ModulationEvent)
            or not math.isfinite(float(item.t))
            or not isinstance(item.modulator, int)
            or isinstance(item.modulator, bool)
            or not 0 <= item.modulator <= 0xFFFFFFFF
            or not math.isfinite(float(item.value))
            for item in items
        ):
            raise ValueError("modulation events must have finite values and uint32 groups")
        return (self._types._CModulationEvent * len(items))(
            *(
                self._types._CModulationEvent(item.t, item.modulator, item.value)
                for item in items
            )
        )

    def _copy_learning_state(
        self,
        handle: ctypes.c_void_p,
        edge_count: int,
        plastic_edge_count: int,
        node_count: int,
    ) -> tuple[
        tuple[float, ...],
        tuple[PlasticityState, ...],
        tuple[LearningObserverState, ...],
    ]:
        weights_array = (self._types.real_array(max(edge_count, 1)))()
        if edge_count > 0:
            status = self._lib.lc_mixed_run_copy_weights(
                handle, weights_array, edge_count
            )
            if status != _LC_OK:
                self._raise(status)
        states_array = (self._types._CPlasticityState * max(plastic_edge_count, 1))()
        count = ctypes.c_uint32()
        status = self._lib.lc_mixed_run_copy_plasticity(
            handle, states_array, plastic_edge_count, ctypes.byref(count)
        )
        if status != _LC_OK:
            self._raise(status)
        observer_array = (self._types._CLearningObserverSnapshot * max(node_count, 1))()
        observer_count = ctypes.c_uint32()
        status = self._lib.lc_mixed_run_copy_learning_observers(
            handle, observer_array, node_count, ctypes.byref(observer_count)
        )
        if status != _LC_OK:
            self._raise(status)
        if observer_count.value != node_count:
            raise CoreError("learning observer state count mismatch")
        kinds = {
            0: "EQUATION_PROGRAM",
            1: "PAIR_STDP",
            2: "TRIPLET_STDP",
            3: "MODULATED_STDP",
        }
        return (
            tuple(float(weights_array[index]) for index in range(edge_count)),
            tuple(
                PlasticityState(
                    edge=int(item.edge),
                    kind=kinds[int(item.kind)],
                    weight=float(item.weight),
                    pre_fast=float(item.pre_fast),
                    post_fast=float(item.post_fast),
                    pre_slow=float(item.pre_slow),
                    post_slow=float(item.post_slow),
                    eligibility_plus=float(item.eligibility_plus),
                    eligibility_minus=float(item.eligibility_minus),
                    t_pre_fast=float(item.t_pre_fast),
                    t_post_fast=float(item.t_post_fast),
                    t_pre_slow=float(item.t_pre_slow),
                    t_post_slow=float(item.t_post_slow),
                    t_eligibility_plus=float(item.t_eligibility_plus),
                    t_eligibility_minus=float(item.t_eligibility_minus),
                )
                for item in states_array[: count.value]
            ),
            tuple(
                LearningObserverState(
                    node=int(item.node),
                    active=bool(item.active),
                    slow_voltage=float(item.slow_voltage),
                    fast_activity=float(item.fast_activity),
                    slow_activity=float(item.slow_activity),
                    sensitivity=float(item.sensitivity),
                    t_activity=float(item.t_activity),
                )
                for item in observer_array[: observer_count.value]
            ),
        )

    @staticmethod
    def _mixed_stats(stats: _CRunStats) -> RunStats:
        return RunStats(
            events_popped=stats.events_popped,
            stale_predictions=stats.stale_predictions,
            peak_queue_occupancy=stats.peak_queue_occupancy,
            deliveries_scheduled=stats.deliveries_scheduled,
            deliveries_processed=stats.deliveries_processed,
            input_spikes_processed=stats.input_spikes_processed,
            drive_updates_processed=stats.drive_updates_processed,
            autonomous_spikes_confirmed=stats.autonomous_spikes_confirmed,
            output_spikes=stats.output_spikes,
            refractory_releases_processed=stats.refractory_releases_processed,
            max_same_time_cascade_depth=stats.max_same_time_cascade_depth,
        )

    @staticmethod
    def _trace_record(
        record: _CTraceRecord,
        state_indices: tuple[int, ...] | None = None,
    ) -> TraceRecord:
        count = int(record.state_count)
        selected = tuple(range(count)) if state_indices is None else tuple(
            index for index in state_indices if index < count
        )
        return TraceRecord(
            t=float(record.t),
            sequence=int(record.sequence),
            generation=int(record.generation),
            kind=TraceKind(record.kind),
            phase=TracePhase(record.phase),
            node=int(record.node),
            subject=None if record.subject == 0xFFFFFFFF else int(record.subject),
            value=float(record.value),
            state_indices=selected,
            before=tuple(float(record.before[index]) for index in selected),
            after=tuple(float(record.after[index]) for index in selected),
        )

    def _raise_mixed_status(self, status: int, error: _CNetworkError) -> None:
        if status == _LC_ROOT_NONCONVERGENCE:
            width = error.root.bracket_high - error.root.bracket_low
            raise CoreError(
                "mixed network root finder did not converge: "
                f"node={error.node}, t={error.t}, bracket_width={width}, "
                f"residual={error.root.residual}, tolerance={error.root.tolerance}",
                status=status,
            )
        resources = {
            1: "queue",
            2: "output",
            3: "decoder_output",
            4: "trace",
            5: "inspection",
        }
        event_kinds = {
            0: "REFRACTORY_RELEASE",
            1: "DRIVE_UPDATE",
            2: "DELIVERY",
            3: "INPUT_SPIKE",
            4: "AUTONOMOUS_SPIKE",
            5: "OUTPUT_SPIKE",
            6: "DECODER_EVENT",
        }
        event_phases = {
            0: "BOUNDARY",
            1: "DEPOSIT",
            2: "PREDICTION",
        }
        if error.resource in resources:
            has_event = bool(error.has_event)
            event_kind = event_kinds.get(error.event_kind) if has_event else None
            event_phase = event_phases.get(error.event_phase) if has_event else None
            if has_event and error.resource == 4:
                try:
                    event_kind = f"TRACE_{TraceKind(error.event_index).name}"
                except ValueError:
                    event_kind = "TRACE_UNKNOWN"
                try:
                    event_phase = TracePhase(error.event_phase).name
                except ValueError:
                    event_phase = None
            elif has_event and error.resource == 5:
                event_kind = "INSPECTION_REQUEST"
                event_phase = None
            diagnostic = CoreFailureDiagnostic(
                resource=resources[error.resource],
                capacity=int(error.capacity),
                occupancy=int(error.occupancy),
                peak=int(error.peak),
                event_kind=event_kind,
                event_phase=event_phase,
                event_index=(
                    None
                    if not has_event or error.event_index == 0xFFFFFFFF
                    else int(error.event_index)
                ),
                node=(
                    None
                    if not has_event or error.node == 0xFFFFFFFF
                    else int(error.node)
                ),
                t=float(error.t) if has_event else None,
            )
            message = self._lib.lc_status_string(status).decode("utf-8")
            details = (
                f"resource={diagnostic.resource}, capacity={diagnostic.capacity}, "
                f"occupancy={diagnostic.occupancy}, peak={diagnostic.peak}"
            )
            if has_event:
                details += (
                    f", event={diagnostic.event_kind or 'UNKNOWN'}, "
                    f"phase={diagnostic.event_phase or 'UNKNOWN'}, "
                    f"index={diagnostic.event_index}, node={diagnostic.node}, "
                    f"t={diagnostic.t}"
                )
            raise CoreError(
                f"{message}: {details}",
                status=status,
                diagnostic=diagnostic,
            )
        self._raise(status)

    def compile_mixed(
        self,
        models: Sequence[ExecutableModel],
        *,
        edges: Sequence[MixedEdge] = (),
        polarities: Sequence[NeuronPolarity] | None = None,
        plasticity: Sequence[PlasticityRule | None] | None = None,
        modulators: Sequence[int | None] | None = None,
        weight_groups: Sequence[int | None] | None = None,
    ) -> "CompiledGraph":
        """Prepare a legacy float64 mixed graph and its CSR index in the C core."""

        if self.precision is not PrecisionProfile.FLOAT64:
            raise CapabilityError(
                "reduced-precision network compilation requires a target-tagged "
                "ExecutionPlan, not untagged resolved models"
            )
        (
            descriptor_array,
            parameter_array,
            edge_array,
            state_layouts,
            program_storage,
            deposit_storage,
            plasticity_array,
        ) = self._pack_mixed_graph(
            models, edges, polarities, plasticity, modulators, weight_groups
        )
        handle = ctypes.c_void_p()
        state_count = sum(count for _, count in state_layouts)
        status = self._lib.lc_mixed_graph_compile_plastic(
            descriptor_array,
            len(models),
            state_count,
            parameter_array,
            len(parameter_array),
            edge_array,
            plasticity_array,
            len(edge_array),
            ctypes.byref(handle),
        )
        if status != _LC_OK:
            self._raise(status)
        return CompiledGraph(
            self,
            handle,
            tuple(models),
            state_layouts,
            edge_count=len(edge_array),
            plastic_edge_count=sum(rule is not None for rule in (plasticity or ())),
        )

    def compile_execution_plan(
        self,
        plan: ExecutionPlan,
    ) -> "CompiledGraph":
        """Compile the shared sparse runtime from the equation-derived plan."""

        self._check_plan_precision(plan)
        (
            descriptor_array,
            parameter_array,
            edge_array,
            state_layouts,
            program_storage,
            deposit_storage,
            _no_legacy_plasticity,
            learning_program_array,
            learning_parameter_array,
            learning_binding_array,
            learning_expression_storage,
        ) = self._pack_execution_plan(plan)
        if _no_legacy_plasticity is not None:
            raise RuntimeError("plan-native packing produced a legacy descriptor array")
        handle = ctypes.c_void_p()
        state_count = sum(count for _, count in state_layouts)
        status = self._lib.lc_mixed_graph_compile_learning(
            descriptor_array,
            len(plan.nodes),
            state_count,
            parameter_array,
            len(parameter_array),
            edge_array,
            learning_program_array,
            len(learning_program_array),
            learning_parameter_array,
            len(learning_parameter_array),
            learning_binding_array,
            len(edge_array),
            ctypes.byref(handle),
        )
        if status != _LC_OK:
            self._raise(status)
        return CompiledGraph(
            self,
            handle,
            None,
            state_layouts,
            runtime_layouts=self._runtime_layouts_from_plan(plan),
            edge_count=len(edge_array),
            plastic_edge_count=plan.requirements.learning_connections,
        )

    def load_compiled_graph_image(
        self,
        image: bytes | bytearray | memoryview,
        *,
        execution_plan: ExecutionPlan | None = None,
    ) -> "CompiledGraph":
        """Load a compiler-free C graph image into the ordinary runtime."""

        if execution_plan is not None:
            self._check_plan_precision(execution_plan)
        try:
            payload = memoryview(image).tobytes()
        except TypeError as exc:
            raise TypeError("compiled graph image must support the buffer protocol") from exc
        if not payload:
            raise ValueError("compiled graph image cannot be empty")
        storage = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        handle = ctypes.c_void_p()
        status = self._lib.lc_compiled_graph_deserialize(
            storage, len(storage), ctypes.byref(handle)
        )
        if status != _LC_OK:
            self._raise(status)
        try:
            info = self._types._CCompiledGraphInfo()
            status = self._lib.lc_compiled_graph_get_info(
                handle, ctypes.byref(info)
            )
            if status != _LC_OK:
                self._raise(status)
            layouts = (self._types._CCompiledNodeLayout * info.node_count)()
            status = self._lib.lc_compiled_graph_copy_node_layouts(
                handle, layouts, info.node_count
            )
            if status != _LC_OK:
                self._raise(status)
            state_layouts = tuple(
                (int(layout.state_offset), int(layout.state_count))
                for layout in layouts
            )
            if execution_plan is None:
                runtime_layouts = tuple(
                    _RuntimeNodeLayout(count, ())
                    for _, count in state_layouts
                )
            else:
                runtime_layouts = self._runtime_layouts_from_plan(execution_plan)
                if (
                    len(execution_plan.nodes) != info.node_count
                    or sum(node.state_count for node in execution_plan.nodes)
                    != info.state_count
                    or len(execution_plan.connections) != info.edge_count
                    or execution_plan.requirements.learning_connections
                    != info.plastic_edge_count
                    or tuple(node.state_count for node in execution_plan.nodes)
                    != tuple(count for _, count in state_layouts)
                ):
                    raise ValueError(
                        "execution plan layout does not match the compiled graph image"
                    )
            return CompiledGraph(
                self,
                handle,
                None,
                state_layouts,
                runtime_layouts=runtime_layouts,
                edge_count=int(info.edge_count),
                plastic_edge_count=int(info.plastic_edge_count),
            )
        except Exception:
            self._lib.lc_mixed_graph_destroy(handle)
            raise

    def load_compiled_graph_file(
        self,
        path: str | os.PathLike[str],
        *,
        execution_plan: ExecutionPlan | None = None,
    ) -> "CompiledGraph":
        """Load a graph image written by :meth:`CompiledGraph.save_image`."""

        return self.load_compiled_graph_image(
            Path(path).read_bytes(), execution_plan=execution_plan
        )

    def run_mixed(
        self,
        models: Sequence[ExecutableModel],
        initial_states: Sequence[float | Sequence[float]],
        *,
        edges: Sequence[MixedEdge] = (),
        polarities: Sequence[NeuronPolarity] | None = None,
        plasticity: Sequence[PlasticityRule | None] | None = None,
        modulators: Sequence[int | None] | None = None,
        weight_groups: Sequence[int | None] | None = None,
        inputs: Sequence[MixedInputSpike] = (),
        drive_updates: Sequence[MixedDriveUpdate] = (),
        modulations: Sequence[ModulationEvent] = (),
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
        return_final_state: bool = True,
    ) -> MixedRunResult:
        """Compatibility path using a temporary compiled graph and run session."""

        with self.compile_mixed(
            models,
            edges=edges,
            polarities=polarities,
            plasticity=plasticity,
            modulators=modulators,
            weight_groups=weight_groups,
        ) as compiled:
            return compiled.run(
                initial_states,
                inputs=inputs,
                drive_updates=drive_updates,
                modulations=modulations,
                t_end=t_end,
                queue_capacity=queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=same_time_cascade_limit,
                stochastic_seed=stochastic_seed,
                recording=recording,
                inspections=inspections,
                return_final_state=return_final_state,
            )

    def run_delta(
        self,
        models: list[ResolvedScalarLIF] | tuple[ResolvedScalarLIF, ...],
        initial_values: list[float] | tuple[float, ...],
        *,
        edges: list[DeltaEdge] | tuple[DeltaEdge, ...] = (),
        polarities: Sequence[NeuronPolarity] | None = None,
        inputs: list[InputSpike] | tuple[InputSpike, ...] = (),
        drive_updates: list[DriveUpdate] | tuple[DriveUpdate, ...] = (),
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
    ) -> RunResult:
        """Run a flat delta-connected scalar-LIF graph entirely in the C core."""

        if self.precision is not PrecisionProfile.FLOAT64:
            raise CapabilityError(
                "reduced-precision network execution requires a target-tagged "
                "ExecutionPlan, not untagged resolved models"
            )
        if not models or len(models) != len(initial_values):
            raise ValueError("models and initial_values must have the same nonzero length")
        if polarities is None:
            normalized_polarities = (NeuronPolarity.EXCITATORY,) * len(models)
        else:
            normalized_polarities = tuple(polarities)
            if len(normalized_polarities) != len(models):
                raise ValueError("polarities must contain one value per model")
        if any(
            not isinstance(polarity, NeuronPolarity)
            for polarity in normalized_polarities
        ):
            raise ValueError(
                "polarity must be EXCITATORY, INHIBITORY, or MIXED"
            )
        if any(
            not 0 <= edge.pre < len(models) or not 0 <= edge.post < len(models)
            for edge in edges
        ):
            raise ValueError("delta edge references an invalid node")
        if any(
            not math.isfinite(edge.weight)
            or (
                edge.weight < 0.0
                and normalized_polarities[edge.pre] is not NeuronPolarity.MIXED
            )
            for edge in edges
        ):
            raise ValueError(
                "delta edge weights must be finite and typed sources require "
                "nonnegative magnitudes. Signed weights require a MIXED "
                "presynaptic neuron"
            )
        return self._run_packed_delta(
            tuple(
                self._model(model, polarity)
                for model, polarity in zip(models, normalized_polarities)
            ),
            tuple(float(value) for value in initial_values),
            tuple(
                self._types._CEdge(edge.pre, edge.post, edge.weight, edge.delay)
                for edge in edges
            ),
            tuple(edge.weight for edge in edges),
            inputs=inputs,
            drive_updates=drive_updates,
            t_end=t_end,
            queue_capacity=queue_capacity,
            output_capacity=output_capacity,
            same_time_cascade_limit=same_time_cascade_limit,
        )

    def run_affine_delta_plan(
        self,
        plan: ExecutionPlan,
        *,
        inputs: Sequence[InputSpike] = (),
        drive_updates: Sequence[DriveUpdate] = (),
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
    ) -> RunResult:
        """Lower an equation-derived affine/delta plan into the compact C ABI."""

        self._check_plan_precision(plan)
        if not plan.compact_scalar_delta_compatible:
            raise ValueError("execution plan is not compatible with affine/delta lowering")
        models = []
        initial_values = []
        for node in plan.nodes:
            affine = node.scalar_affine
            if affine is None:
                raise ValueError("affine/delta plan node is missing affine parameters")
            models.append(
                self._types._CModel(
                    affine.decay,
                    affine.drive,
                    node.threshold,
                    affine.reset,
                    node.refractory,
                    node.polarity.runtime_code,
                )
            )
            initial_values.append(plan.states[node.state_offset].initial)
        edges = tuple(
            self._types._CEdge(
                connection.pre,
                connection.post,
                connection.weight,
                connection.delay,
            )
            for connection in plan.connections
        )
        return self._run_packed_delta(
            tuple(models),
            tuple(initial_values),
            edges,
            tuple(connection.weight for connection in plan.connections),
            inputs=inputs,
            drive_updates=drive_updates,
            t_end=t_end,
            queue_capacity=queue_capacity,
            output_capacity=output_capacity,
            same_time_cascade_limit=same_time_cascade_limit,
        )

    def _run_packed_delta(
        self,
        models: Sequence[_CModel],
        initial_values: Sequence[float],
        edges: Sequence[_CEdge],
        weights: Sequence[float],
        *,
        inputs: Sequence[InputSpike],
        drive_updates: Sequence[DriveUpdate],
        t_end: float,
        queue_capacity: int,
        output_capacity: int,
        same_time_cascade_limit: int,
    ) -> RunResult:
        """Execute records already lowered to the compact scalar/delta ABI."""

        if not models or len(models) != len(initial_values):
            raise ValueError("packed models and initial values must have equal nonzero length")
        if len(edges) != len(weights):
            raise ValueError("packed edges and reported weights must have equal length")
        if any(
            count > 0xFFFFFFFF
            for count in (len(models), len(edges), len(inputs), len(drive_updates))
        ):
            raise ValueError("node, edge, input, and drive-update counts must fit uint32")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if output_capacity < 0:
            raise ValueError("output_capacity must be nonnegative")
        if same_time_cascade_limit <= 0:
            raise ValueError("same_time_cascade_limit must be positive")
        model_array = (self._types._CModel * len(models))(*models)
        state_array = (self._types._CState * len(models))(
            *(self._types._CState(float(value), 0.0) for value in initial_values)
        )
        edge_array = (self._types._CEdge * len(edges))(*edges)
        input_array = (self._types._CInputSpike * len(inputs))(
            *(self._types._CInputSpike(event.t, event.node, event.value) for event in inputs)
        )
        drive_array = (self._types._CDriveUpdate * len(drive_updates))(
            *(self._types._CDriveUpdate(event.t, event.node, event.b) for event in drive_updates)
        )
        output_storage = max(output_capacity, 1)
        output_array = (self._types._COutputSpike * output_storage)()
        config = self._types._CRunConfig(
            t_end,
            queue_capacity,
            output_capacity,
            same_time_cascade_limit,
            0,
        )
        output_count = ctypes.c_uint64()
        stats = self._types._CRunStats()
        status = self._lib.lc_delta_network_run(
            model_array,
            state_array,
            len(models),
            edge_array,
            len(edges),
            input_array,
            len(inputs),
            drive_array,
            len(drive_updates),
            ctypes.byref(config),
            output_array,
            ctypes.byref(output_count),
            ctypes.byref(stats),
        )
        if status != _LC_OK:
            self._raise(status)
        return RunResult(
            states=tuple(ScalarState(item.value, item.t_last) for item in state_array),
            spikes=tuple(
                Spike(output_array[index].t, output_array[index].node)
                for index in range(output_count.value)
            ),
            stats=RunStats(
                events_popped=stats.events_popped,
                stale_predictions=stats.stale_predictions,
                peak_queue_occupancy=stats.peak_queue_occupancy,
                deliveries_scheduled=stats.deliveries_scheduled,
                deliveries_processed=stats.deliveries_processed,
                input_spikes_processed=stats.input_spikes_processed,
                drive_updates_processed=stats.drive_updates_processed,
                autonomous_spikes_confirmed=stats.autonomous_spikes_confirmed,
                output_spikes=stats.output_spikes,
                refractory_releases_processed=stats.refractory_releases_processed,
                max_same_time_cascade_depth=stats.max_same_time_cascade_depth,
            ),
            kernel_seconds=float(stats.kernel_seconds),
            weights=tuple(float(weight) for weight in weights),
        )

    def evaluate_expr(
        self,
        dag: ExprDAG,
        *,
        parameters: dict[str, float] | None = None,
        variables: dict[str, float] | None = None,
        roots: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate named roots in C, optionally restricting work to their closure.

        With no selection the existing whole-DAG evaluation order is retained.
        Selected roots are returned in caller order without evaluating unrelated
        state-dependent expressions at placeholder variable values.
        """

        if not dag.nodes:
            raise ValueError("expression DAG cannot be empty")
        if len(dag.nodes) > 0xFFFFFFFF:
            raise ValueError("expression DAG node count must fit uint32")
        parameter_values = parameters or {}
        variable_values = variables or {}
        if set(parameter_values) != set(dag.parameters):
            raise ValueError("parameter bindings must match the DAG parameter names exactly")
        if set(variable_values) != set(dag.variables):
            raise ValueError("variable bindings must match the DAG variable names exactly")
        selected = None
        if roots is not None:
            if isinstance(roots, (str, bytes)):
                raise TypeError("roots must be a sequence of root names")
            selected = tuple(roots)
            if any(not isinstance(name, str) for name in selected):
                raise TypeError("selected root names must be strings")
            if len(set(selected)) != len(selected) or any(
                name not in dag.roots for name in selected
            ):
                raise ValueError("selected roots must be unique DAG root names")
            if not selected:
                return {}
        node_array = self._expr_nodes(dag)
        parameter_array = (self._types.real_array(len(dag.parameters)))(
            *(float(parameter_values[name]) for name in dag.parameters)
        )
        variable_array = (self._types.real_array(len(dag.variables)))(
            *(float(variable_values[name]) for name in dag.variables)
        )
        workspace = (self._types.real_array(len(dag.nodes)))()
        if selected is not None:
            indices = (ctypes.c_uint32 * len(selected))(
                *(dag.roots[name] for name in selected)
            )
            output = self._types.real_array(len(selected))()
            active = (ctypes.c_uint8 * len(dag.nodes))()
            status = self._lib.lc_expr_evaluate_selected(
                node_array, len(dag.nodes), parameter_array, len(dag.parameters),
                variable_array, len(dag.variables), indices, len(indices), output,
                workspace, len(workspace), active, len(active),
            )
            if status != _LC_OK:
                self._raise(status)
            return dict(zip(selected, output))
        status = self._lib.lc_expr_evaluate(
            node_array,
            len(dag.nodes),
            parameter_array,
            len(dag.parameters),
            variable_array,
            len(dag.variables),
            workspace,
            len(dag.nodes),
        )
        if status != _LC_OK:
            self._raise(status)
        return {name: workspace[index] for name, index in dag.roots.items()}


class StreamingEncoderRun:
    """C-owned live encoder state aligned to an open simulation frontier."""

    def __init__(
        self,
        core: CoreEvaluator,
        encoders: Sequence[Encoder],
        *,
        initial_frontier: float,
        seed: int,
        spike_capacity: int,
        drive_capacity: int,
    ):
        self._types = core._types
        if not isinstance(seed, int) or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for label, capacity in (
            ("spike_capacity", spike_capacity),
            ("drive_capacity", drive_capacity),
        ):
            if (
                not isinstance(capacity, int)
                or not 0 <= capacity <= 2**64 - 1
            ):
                raise ValueError(f"{label} must be an unsigned 64-bit integer")
        initial_frontier = self._types.time_value(float(initial_frontier)).value
        if not math.isfinite(initial_frontier):
            raise ValueError("initial_frontier must be finite")
        self._core = core
        self.encoders = tuple(encoders)
        if len(self.encoders) > 2**32 - 1:
            raise ValueError("encoder count exceeds the unsigned 32-bit limit")
        specs = (self._types._CEncoderSpec * len(self.encoders))(
            *(
                core._encoder_spec(encoder, index)
                for index, encoder in enumerate(self.encoders)
            )
        )
        handle = ctypes.c_void_p()
        status = core._lib.lc_encoder_run_create(
            specs,
            len(specs),
            ctypes.c_uint64(seed),
            initial_frontier,
            ctypes.byref(handle),
        )
        if status != _LC_OK:
            core._raise(status)
        self._handle = handle
        self._frontier = initial_frontier
        self._spike_capacity = spike_capacity
        self._drive_capacity = drive_capacity
        self._finished = False
        self._closed = False

    @property
    def frontier(self) -> float:
        """Return the open presentation frontier."""

        return self._frontier

    @property
    def finished(self) -> bool:
        """Return whether the encoder run has been sealed."""

        return self._finished

    @property
    def closed(self) -> bool:
        """Return whether the C encoder run has been released."""

        return self._closed

    def close(self) -> None:
        """Release the C encoder run once."""

        if self._closed:
            return
        self._core._lib.lc_encoder_run_destroy(self._handle)
        self._handle = ctypes.c_void_p()
        self._closed = True

    def __enter__(self) -> "StreamingEncoderRun":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("streaming encoder run is closed")
        if self._finished:
            raise RuntimeError("streaming encoder run is finished")

    def _advance(
        self,
        until: float,
        presentations: Sequence[Presentation],
        *,
        seal: bool,
    ) -> EncodedBatch:
        self._require_active()
        requested_until = float(until)
        until = self._types.time_value(requested_until).value
        if requested_until > self._frontier and until <= self._frontier:
            raise PrecisionResolutionError("positive encoder interval does not advance time")
        if not math.isfinite(until) or until < self._frontier:
            raise ValueError("until must be finite and not precede the frontier")
        normalized = tuple(presentations)
        if len(normalized) > 2**32 - 1:
            raise ValueError("presentation batch exceeds the unsigned 32-bit limit")
        last_key: tuple[float, int] | None = None
        per_encoder_start: dict[int, float] = {}
        for index, item in enumerate(normalized):
            if not isinstance(item, Presentation):
                raise ValueError("streaming encoder runs accept Presentation values")
            start = float(item.t_start)
            end = float(item.t_end)
            value = float(item.value)
            if (
                not isinstance(item.encoder, int)
                or not 0 <= item.encoder < len(self.encoders)
                or isinstance(self.encoders[item.encoder], NativeEventEncoder)
                or not math.isfinite(start)
                or not math.isfinite(end)
                or not start < end
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                or start < self._frontier
                or (start > until if seal else start >= until)
            ):
                interval = "[frontier, until]" if seal else "[frontier, until)"
                raise ValueError(
                    f"presentation {index} requires a scalar encoder, finite "
                    f"start < end and normalized value; start must be in {interval}"
                )
            key = (start, item.encoder)
            if last_key is not None and key < last_key:
                raise ValueError(
                    "presentations must be ordered by (t_start, encoder)"
                )
            prior = per_encoder_start.get(item.encoder)
            if prior is not None and start <= prior:
                raise ValueError(
                    "presentation starts must increase strictly per encoder"
                )
            last_key = key
            per_encoder_start[item.encoder] = start
        items = (self._types._CPresentation * len(normalized))(
            *(
                self._types._CPresentation(item.t_start, item.t_end, item.encoder, item.value)
                for item in normalized
            )
        )
        spikes = (self._types._CEncodedSpike * max(self._spike_capacity, 1))()
        drives = (self._types._CEncodedDrive * max(self._drive_capacity, 1))()
        spike_count = ctypes.c_uint64()
        drive_count = ctypes.c_uint64()
        status = self._core._lib.lc_encoder_run_advance(
            self._handle,
            items,
            len(items),
            until,
            int(seal),
            spikes,
            self._spike_capacity,
            ctypes.byref(spike_count),
            drives,
            self._drive_capacity,
            ctypes.byref(drive_count),
        )
        if status != _LC_OK:
            self._core._raise(status)
        self._frontier = until
        self._finished = seal
        return EncodedBatch(
            tuple(
                EncodedSpike(
                    spikes[index].t,
                    spikes[index].encoder,
                    spikes[index].value,
                )
                for index in range(spike_count.value)
            ),
            tuple(
                EncodedDrive(
                    drives[index].t,
                    drives[index].encoder,
                    drives[index].value,
                )
                for index in range(drive_count.value)
            ),
        )

    def advance_until(
        self,
        until: float,
        presentations: Sequence[Presentation] = (),
    ) -> EncodedBatch:
        """Emit primitives strictly before an open right boundary."""

        return self._advance(until, presentations, seal=False)

    def reset_episode(self) -> None:
        """Clear presentation/phase state at the current frontier."""

        self._require_active()
        status = self._core._lib.lc_encoder_run_reset_episode(self._handle)
        if status != _LC_OK:
            self._core._raise(status)

    def finish(
        self,
        until: float,
        presentations: Sequence[Presentation] = (),
    ) -> EncodedBatch:
        """Emit through the inclusive run horizon and seal the encoder run."""

        return self._advance(until, presentations, seal=True)


class EncoderSession:
    """Incrementally encode ordered presentations while preserving C state.

    ``advance`` is a finality watermark: it returns every globally ordered
    presentation prefix whose effective end can no longer be shortened by a
    future presentation. Returned events can therefore predate the watermark.
    """

    def __init__(
        self,
        core: CoreEvaluator,
        encoders: Sequence[Encoder],
        *,
        seed: int,
        spike_capacity: int,
        drive_capacity: int | None,
    ):
        self._types = core._types
        if not isinstance(seed, int) or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if (
            not isinstance(spike_capacity, int)
            or not 0 <= spike_capacity <= 2**64 - 1
        ):
            raise ValueError("spike_capacity must be an unsigned 64-bit integer")
        if drive_capacity is not None and (
            not isinstance(drive_capacity, int)
            or not 0 <= drive_capacity <= 2**64 - 1
        ):
            raise ValueError(
                "drive_capacity must be an unsigned 64-bit integer or None"
            )
        self._core = core
        self.encoders = tuple(encoders)
        if len(self.encoders) > 2**32 - 1:
            raise ValueError("encoder count exceeds the unsigned 32-bit limit")
        self._specs = (self._types._CEncoderSpec * len(self.encoders))(
            *(
                core._encoder_spec(encoder, index)
                for index, encoder in enumerate(self.encoders)
            )
        )
        self._states = (self._types._CEncoderState * len(self.encoders))()
        status = core._lib.lc_encoder_state_reset(
            self._states, len(self._states), ctypes.c_uint64(seed)
        )
        if status != _LC_OK:
            core._raise(status)
        empty_count = ctypes.c_uint64()
        status = core._lib.lc_encode_presentations(
            self._specs,
            len(self._specs),
            self._states,
            len(self._states),
            None,
            0,
            None,
            0,
            ctypes.byref(empty_count),
            None,
            0,
            ctypes.byref(empty_count),
        )
        if status != _LC_OK:
            core._raise(status)
        self._spike_capacity = spike_capacity
        self._drive_capacity = drive_capacity
        self._pending: list[Presentation] = []
        self._last_key: tuple[float, int] | None = None
        self._last_starts: dict[int, float] = {}
        self._watermark = -math.inf
        self._finished = False
        self._poisoned = False

    @property
    def finished(self) -> bool:
        """Return whether the high-level encoder session is sealed."""

        return self._finished

    def _require_active(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "encoder session is unusable after a failed C encoding call"
            )
        if self._finished:
            raise RuntimeError("encoder session is finished")

    def submit(self, presentations: Sequence[Presentation]) -> None:
        """Queue presentations ordered by ``(t_start, encoder)``."""

        self._require_active()
        normalized = tuple(presentations)
        staged_last_key = self._last_key
        staged_starts = dict(self._last_starts)
        for index, item in enumerate(normalized):
            if not isinstance(item, Presentation):
                raise ValueError("encoder sessions accept Presentation values")
            if (
                not isinstance(item.encoder, int)
                or not 0 <= item.encoder < len(self.encoders)
                or not math.isfinite(float(item.t_start))
                or not math.isfinite(float(item.t_end))
                or not float(item.t_start) < float(item.t_end)
                or not math.isfinite(float(item.value))
                or not 0.0 <= float(item.value) <= 1.0
            ):
                raise ValueError(
                    f"presentation {index} requires a valid encoder, finite "
                    "start < end, and normalized value in [0, 1]"
                )
            key = (float(item.t_start), int(item.encoder))
            if key[0] < self._watermark:
                raise ValueError(
                    "presentation starts cannot precede the committed watermark"
                )
            if staged_last_key is not None and key < staged_last_key:
                raise ValueError(
                    "presentations must be submitted in (t_start, encoder) order"
                )
            prior_start = staged_starts.get(item.encoder)
            if prior_start is not None and not key[0] > prior_start:
                raise ValueError(
                    "presentation starts must increase strictly per encoder"
                )
            staged_last_key = key
            staged_starts[item.encoder] = key[0]
        self._pending.extend(normalized)
        self._last_key = staged_last_key
        self._last_starts = staged_starts

    def _finalized_prefix(self, through: float) -> tuple[Presentation, ...]:
        next_starts: dict[int, float] = {}
        effective_ends = [0.0] * len(self._pending)
        for index in range(len(self._pending) - 1, -1, -1):
            item = self._pending[index]
            effective_ends[index] = min(
                float(item.t_end), next_starts.get(item.encoder, math.inf)
            )
            next_starts[item.encoder] = float(item.t_start)
        count = 0
        for effective_end in effective_ends:
            if effective_end > through:
                break
            count += 1
        return tuple(
            Presentation(
                float(item.t_start),
                effective_ends[index],
                int(item.encoder),
                float(item.value),
            )
            for index, item in enumerate(self._pending[:count])
        )

    def _encode(self, presentations: tuple[Presentation, ...]) -> EncodedBatch:
        if not presentations:
            return EncodedBatch((), ())
        if len(presentations) > 2**32 - 1:
            raise ValueError("presentation batch exceeds the unsigned 32-bit limit")
        drive_capacity = (
            2 * len(presentations)
            if self._drive_capacity is None
            else self._drive_capacity
        )
        items = (self._types._CPresentation * len(presentations))(
            *(
                self._types._CPresentation(item.t_start, item.t_end, item.encoder, item.value)
                for item in presentations
            )
        )
        spikes = (self._types._CEncodedSpike * max(self._spike_capacity, 1))()
        drives = (self._types._CEncodedDrive * max(drive_capacity, 1))()
        spike_count = ctypes.c_uint64()
        drive_count = ctypes.c_uint64()
        status = self._core._lib.lc_encode_presentations(
            self._specs,
            len(self._specs),
            self._states,
            len(self._states),
            items,
            len(items),
            spikes,
            self._spike_capacity,
            ctypes.byref(spike_count),
            drives,
            drive_capacity,
            ctypes.byref(drive_count),
        )
        if status != _LC_OK:
            self._poisoned = True
            self._core._raise(status)
        return EncodedBatch(
            tuple(
                EncodedSpike(
                    spikes[index].t,
                    spikes[index].encoder,
                    spikes[index].value,
                )
                for index in range(spike_count.value)
            ),
            tuple(
                EncodedDrive(
                    drives[index].t,
                    drives[index].encoder,
                    drives[index].value,
                )
                for index in range(drive_count.value)
            ),
        )

    def advance(self, through: float) -> EncodedBatch:
        """Commit input history through a finite, nondecreasing watermark."""

        self._require_active()
        requested_through = float(through)
        through = self._types.time_value(requested_through).value
        if requested_through > self._watermark and through <= self._watermark:
            raise PrecisionResolutionError("positive encoder interval does not advance time")
        if not math.isfinite(through) or through < self._watermark:
            raise ValueError("watermarks must be finite and nondecreasing")
        finalized = self._finalized_prefix(through)
        result = self._encode(finalized)
        del self._pending[: len(finalized)]
        self._watermark = through
        return result

    def finish(self) -> EncodedBatch:
        """Flush every remaining finite presentation and seal the session."""

        self._require_active()
        finalized = self._finalized_prefix(math.inf)
        result = self._encode(finalized)
        del self._pending[: len(finalized)]
        self._finished = True
        return result


class CompiledDecoderBank:
    """Immutable C-owned decoder specs reusable across independent runs."""

    def __init__(
        self,
        core: CoreEvaluator,
        handle: ctypes.c_void_p,
        bindings: tuple[DecoderBinding, ...],
    ):
        self._types = core._types
        self._core = core
        self._handle = handle
        self.bindings = bindings

    @property
    def closed(self) -> bool:
        """Return whether compiled decoder resources were released."""

        return not bool(self._handle and self._handle.value)

    def close(self) -> None:
        """Release immutable decoder resources once."""

        if not self.closed:
            self._core._lib.lc_decoder_bank_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "CompiledDecoderBank":
        if self.closed:
            raise RuntimeError("compiled decoder bank is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def create_run(
        self,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        windows: Sequence[DecodeWindow] | None = None,
        schedule: Sequence[DecoderWindowBinding] | None = None,
        queries: Sequence[DecoderQueryBinding] = (),
        event_capacity: int = 4096,
    ) -> "StreamingDecoderRun":
        """Create an independent decoder run with its own schedule."""

        if self.closed:
            raise RuntimeError("compiled decoder bank is closed")
        if not isinstance(event_capacity, int) or not 0 <= event_capacity <= 2**64 - 1:
            raise ValueError("decoder event capacity must be an unsigned 64-bit integer")
        if schedule is not None:
            if windows is not None or t_start is not None or t_end is not None:
                raise ValueError(
                    "pass schedule, windows, or t_start/t_end—not a combination"
                )
            decode_schedule = tuple(schedule)
            if not decode_schedule or not all(
                isinstance(binding, DecoderWindowBinding)
                for binding in decode_schedule
            ):
                raise ValueError(
                    "schedule must contain at least one DecoderWindowBinding"
                )
            decode_windows = None
        elif windows is None:
            if t_start is None or t_end is None:
                raise ValueError("a decoder window requires both t_start and t_end")
            decode_windows = (DecodeWindow(float(t_start), float(t_end)),)
            decode_schedule = None
        else:
            if t_start is not None or t_end is not None:
                raise ValueError("pass either windows or t_start/t_end, not both")
            decode_windows = tuple(windows)
            if not decode_windows or not all(
                isinstance(window, DecodeWindow) for window in decode_windows
            ):
                raise ValueError("windows must contain at least one DecodeWindow")
            decode_schedule = None
        handle = ctypes.c_void_p()
        status = self._core._lib.lc_decoder_run_create(
            self._handle, ctypes.byref(handle)
        )
        if status != _LC_OK:
            self._core._raise(status)
        result = StreamingDecoderRun(self, handle, event_capacity)
        try:
            status = self._core._lib.lc_decoder_run_reserve_events(
                result._handle, event_capacity
            )
            if status != _LC_OK:
                self._core._raise(status)
            if decode_schedule is None:
                assert decode_windows is not None
                result.reset_windows(decode_windows)
            else:
                result.reset_schedule(decode_schedule)
            result.set_queries(queries)
        except Exception:
            result.close()
            raise
        return result


class StreamingDecoderRun:
    """Mutable streaming decoder state for an ordered observation schedule."""

    def __init__(
        self,
        compiled: CompiledDecoderBank,
        handle: ctypes.c_void_p,
        event_capacity: int,
    ):
        self._types = compiled._core._types
        self._compiled = compiled
        self._handle = handle
        self._event_capacity = event_capacity
        self._windows: tuple[DecodeWindow, ...] = ()
        self._schedule: tuple[DecoderWindowBinding, ...] = ()
        self._queries: tuple[DecoderQueryBinding, ...] = ()

    @property
    def closed(self) -> bool:
        """Return whether mutable decoder state was released."""

        return not bool(self._handle and self._handle.value)

    def close(self) -> None:
        """Release mutable decoder state once."""

        if not self.closed:
            self._compiled._core._lib.lc_decoder_run_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "StreamingDecoderRun":
        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset(self, *, t_start: float, t_end: float) -> None:
        """Reset every decoder onto one shared window."""

        self.reset_windows((DecodeWindow(float(t_start), float(t_end)),))

    def reset_windows(self, windows: Sequence[DecodeWindow]) -> None:
        """Reset every decoder across a common window sequence."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        normalized = tuple(windows)
        if not normalized or not all(
            isinstance(window, DecodeWindow) for window in normalized
        ):
            raise ValueError("windows must contain at least one DecodeWindow")
        c_windows = (self._types._CDecodeWindow * len(normalized))(
            *(
                self._types._CDecodeWindow(float(window.t_start), float(window.t_end))
                for window in normalized
            )
        )
        status = self._compiled._core._lib.lc_decoder_run_reset_windows(
            self._handle, c_windows, len(c_windows)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        self._windows = normalized
        self._schedule = ()
        self._queries = ()

    def reset_schedule(self, schedule: Sequence[DecoderWindowBinding]) -> None:
        """Reset with explicit sparse decoder/window assignments."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        normalized = tuple(schedule)
        if not normalized or not all(
            isinstance(binding, DecoderWindowBinding) for binding in normalized
        ):
            raise ValueError(
                "schedule must contain at least one DecoderWindowBinding"
            )
        if any(
            not isinstance(binding.decoder, int)
            or not 0 <= binding.decoder <= 2**32 - 1
            or not isinstance(binding.window, int)
            or not 0 <= binding.window <= 2**32 - 1
            for binding in normalized
        ):
            raise ValueError(
                "decoder and window identifiers must be unsigned 32-bit integers"
            )
        c_schedule = (self._types._CDecoderWindowBinding * len(normalized))(
            *(
                self._types._CDecoderWindowBinding(
                    int(binding.decoder),
                    int(binding.window),
                    float(binding.t_start),
                    float(binding.t_end),
                )
                for binding in normalized
            )
        )
        status = self._compiled._core._lib.lc_decoder_run_reset_schedule(
            self._handle, c_schedule, len(c_schedule)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        self._schedule = normalized
        self._windows = ()
        self._queries = ()

    def set_queries(self, queries: Sequence[DecoderQueryBinding]) -> None:
        """Set exact-time snapshots for scheduled ON_QUERY decoders."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        normalized = tuple(queries)
        if not all(isinstance(query, DecoderQueryBinding) for query in normalized):
            raise ValueError("queries must contain DecoderQueryBinding values")
        if any(
            not isinstance(query.decoder, int)
            or not 0 <= query.decoder <= 2**32 - 1
            or not isinstance(query.window, int)
            or not 0 <= query.window <= 2**32 - 1
            for query in normalized
        ):
            raise ValueError(
                "decoder and window identifiers must be unsigned 32-bit integers"
            )
        c_queries = (self._types._CDecoderQueryBinding * len(normalized))(
            *(
                self._types._CDecoderQueryBinding(
                    int(query.decoder), int(query.window), float(query.t)
                )
                for query in normalized
            )
        )
        status = self._compiled._core._lib.lc_decoder_run_set_queries(
            self._handle, c_queries, len(c_queries)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        self._queries = normalized

    def advance(self, observed_through: float) -> None:
        """Emit every query and close every window due by this time."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        status = self._compiled._core._lib.lc_decoder_run_advance(
            self._handle, float(observed_through)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)

    def consume(self, spikes: Sequence[Spike]) -> None:
        """Consume chronological output spikes."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        for item in spikes:
            spike = self._types._COutputSpike(float(item.t), int(item.node))
            status = self._compiled._core._lib.lc_decoder_run_consume(
                self._handle, ctypes.byref(spike)
            )
            if status != _LC_OK:
                self._compiled._core._raise(status)

    def finalize(self) -> tuple[DecodeValue, ...]:
        """Close remaining windows and return final values."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        result_count = ctypes.c_uint64()
        status = self._compiled._core._lib.lc_decoder_run_result_count(
            self._handle, ctypes.byref(result_count)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        results = (self._types._CDecodeResult * result_count.value)()
        status = self._compiled._core._lib.lc_decoder_run_finalize(
            self._handle, results, result_count.value
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        return tuple(
            DecodeValue(
                result.decoder,
                bool(result.valid),
                result.count,
                result.value if result.valid else None,
                result.first_spike if result.count else None,
                result.window,
                result.window_start,
                result.window_end,
            )
            for result in results
        )

    def events(self) -> tuple[DecodeEvent, ...]:
        """Copy retained decoded events in chronological emission order."""

        if self.closed:
            raise RuntimeError("streaming decoder run is closed")
        events = (self._types._CDecodedEvent * max(self._event_capacity, 1))()
        count = ctypes.c_uint64()
        status = self._compiled._core._lib.lc_decoder_run_copy_events(
            self._handle, events, self._event_capacity, ctypes.byref(count)
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        return tuple(
            DecodeEvent(
                decoder=event.decoder,
                window=event.window,
                kind=DecodeEventKind(event.kind),
                valid=bool(event.valid),
                emitted_at=event.emitted_at,
                source_spike_time=(
                    event.source_spike_time if event.has_source else None
                ),
                window_start=event.window_start,
                window_end=event.window_end,
                observed_through=event.observed_through,
                count=event.count,
                value=event.value if event.valid else None,
                first_spike=event.first_spike if event.has_first_spike else None,
            )
            for event in events[: count.value]
        )


class CompiledGraph:
    """Immutable C-owned graph blueprint reusable across independent runs."""

    def __init__(
        self,
        core: CoreEvaluator,
        handle: ctypes.c_void_p,
        models: tuple[ExecutableModel, ...] | None,
        state_layouts: tuple[tuple[int, int], ...],
        *,
        runtime_layouts: tuple[_RuntimeNodeLayout, ...] | None = None,
        edge_count: int = 0,
        plastic_edge_count: int = 0,
    ):
        self._types = core._types
        self._core = core
        self._handle = handle
        self.models = () if models is None else models
        self.state_layouts = state_layouts
        self.runtime_layouts = (
            CoreEvaluator._runtime_layouts_from_models(self.models)
            if runtime_layouts is None
            else runtime_layouts
        )
        self.node_count = len(self.runtime_layouts)
        if self.node_count != len(state_layouts):
            raise ValueError("runtime node and state layouts must have the same length")
        self.state_count = sum(count for _, count in state_layouts)
        self.edge_count = edge_count
        self.plastic_edge_count = plastic_edge_count

    @property
    def precision(self) -> PrecisionProfile:
        """Return the validated precision used by this graph's native runtime."""

        return self._core.precision

    @property
    def closed(self) -> bool:
        """Return whether the compiled graph was released."""

        return not bool(self._handle and self._handle.value)

    def to_bytes(self) -> bytes:
        """Serialize this compiled C graph as a portable deployment image."""

        if self.closed:
            raise RuntimeError("compiled graph is closed")
        size = ctypes.c_uint64()
        status = self._core._lib.lc_compiled_graph_image_size(
            self._handle, ctypes.byref(size)
        )
        if status != _LC_OK:
            self._core._raise(status)
        storage = (ctypes.c_uint8 * size.value)()
        written = ctypes.c_uint64()
        status = self._core._lib.lc_compiled_graph_serialize(
            self._handle,
            storage,
            size.value,
            ctypes.byref(written),
        )
        if status != _LC_OK:
            self._core._raise(status)
        if written.value != size.value:
            raise CoreError("compiled graph serializer returned an invalid length")
        return bytes(storage)

    def save_image(self, path: str | os.PathLike[str]) -> Path:
        """Write a compiler-free graph image for native or embedded runtime use."""

        destination = Path(path)
        destination.write_bytes(self.to_bytes())
        return destination

    def close(self) -> None:
        """Release immutable compiled graph resources once."""

        if not self.closed:
            self._core._lib.lc_mixed_graph_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "CompiledGraph":
        if self.closed:
            raise RuntimeError("compiled graph is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def create_run(
        self,
        initial_states: Sequence[float | Sequence[float]],
    ) -> "CompiledRun":
        """Create independent mutable state for one compiled graph."""

        if self.closed:
            raise RuntimeError("compiled graph is closed")
        state_array, t_last_array = self._core._pack_mixed_initial(
            self.runtime_layouts, initial_states
        )
        handle = ctypes.c_void_p()
        status = self._core._lib.lc_mixed_run_create(
            self._handle,
            state_array,
            len(state_array),
            t_last_array,
            len(t_last_array),
            ctypes.byref(handle),
        )
        if status != _LC_OK:
            self._core._raise(status)
        return CompiledRun(self, handle)

    def create_incremental_run(
        self,
        initial_states: Sequence[float | Sequence[float]],
        *,
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        decoder_run: StreamingDecoderRun | None = None,
        return_final_state: bool = True,
    ) -> "IncrementalCompiledRun":
        """Open a resumable raw-event execution with a fixed final horizon."""

        if self.closed:
            raise RuntimeError("compiled graph is closed")
        self._core._check_decoder_run(decoder_run)
        if not isinstance(return_final_state, bool):
            raise ValueError("return_final_state must be boolean")
        if (
            not math.isfinite(float(t_end))
            or float(t_end) < 0.0
            or not isinstance(queue_capacity, int)
            or not 0 < queue_capacity <= 2**64 - 1
            or not isinstance(output_capacity, int)
            or not 0 <= output_capacity <= 2**64 - 1
            or not isinstance(same_time_cascade_limit, int)
            or not 0 < same_time_cascade_limit <= 2**32 - 1
            or not isinstance(stochastic_seed, int)
            or isinstance(stochastic_seed, bool)
            or not 0 <= stochastic_seed <= 2**64 - 1
        ):
            raise ValueError("incremental run horizon or capacities are invalid")
        config = self._types._CRunConfig(
            float(t_end),
            queue_capacity,
            output_capacity,
            same_time_cascade_limit,
            stochastic_seed,
        )
        state_array, t_last_array = self._core._pack_mixed_initial(
            self.runtime_layouts, initial_states
        )
        handle = ctypes.c_void_p()
        status = self._core._lib.lc_mixed_run_create(
            self._handle,
            state_array,
            len(state_array),
            t_last_array,
            len(t_last_array),
            ctypes.byref(handle),
        )
        if status != _LC_OK:
            self._core._raise(status)
        error = self._types._CNetworkError()
        status = self._core._lib.lc_mixed_run_begin_incremental(
            handle, ctypes.byref(config), ctypes.byref(error)
        )
        if status != _LC_OK:
            self._core._lib.lc_mixed_run_destroy(handle)
            self._core._raise_mixed_status(status, error)
        return IncrementalCompiledRun(
            self,
            handle,
            config,
            decoder_run=decoder_run,
            return_final_state=return_final_state,
        )

    def run(
        self,
        initial_states: Sequence[float | Sequence[float]],
        *,
        inputs: Sequence[MixedInputSpike] = (),
        drive_updates: Sequence[MixedDriveUpdate] = (),
        modulations: Sequence[ModulationEvent] = (),
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        decoder_run: StreamingDecoderRun | None = None,
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
        return_final_state: bool = True,
    ) -> MixedRunResult:
        """Create a temporary session while retaining this prepared graph."""

        with self.create_run(initial_states) as run:
            return run.execute(
                inputs=inputs,
                drive_updates=drive_updates,
                modulations=modulations,
                t_end=t_end,
                queue_capacity=queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=same_time_cascade_limit,
                stochastic_seed=stochastic_seed,
                decoder_run=decoder_run,
                recording=recording,
                inspections=inspections,
                return_final_state=return_final_state,
            )


class IncrementalCompiledRun:
    """Resumable C network execution over open, half-interval boundaries."""

    def __init__(
        self,
        compiled: CompiledGraph,
        handle: ctypes.c_void_p,
        config: _CRunConfig,
        *,
        decoder_run: StreamingDecoderRun | None,
        return_final_state: bool,
    ):
        self._types = compiled._core._types
        self._compiled = compiled
        self._handle = handle
        self._config = config
        self._decoder_run = decoder_run
        self._return_final_state = return_final_state
        self._frontier = 0.0
        self._finished = False
        self._failed = False
        self._decoder_event_cursor = 0
        self._cumulative_stats = RunStats(*(0 for _ in range(11)))

    @property
    def closed(self) -> bool:
        """Return whether the incremental C run is closed."""

        return not bool(self._handle and self._handle.value)

    @property
    def frontier(self) -> float:
        """Return the settled open event frontier."""

        return self._frontier

    @property
    def t_end(self) -> float:
        """Return the fixed final horizon."""

        return float(self._config.t_end)

    @property
    def finished(self) -> bool:
        """Return whether the final horizon has been sealed."""

        return self._finished

    @property
    def cumulative_stats(self) -> RunStats:
        """Return counters accumulated across all run segments."""

        return self._cumulative_stats

    def close(self) -> None:
        """Release the incremental C run once."""

        if not self.closed:
            self._compiled._core._lib.lc_mixed_run_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "IncrementalCompiledRun":
        if self.closed:
            raise RuntimeError("incremental compiled run is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _require_active(self) -> None:
        if self.closed:
            raise RuntimeError("incremental compiled run is closed")
        if self._failed:
            raise RuntimeError("incremental compiled run failed and cannot resume")
        if self._finished:
            raise RuntimeError("incremental compiled run is finished")

    @staticmethod
    def _accumulate_stats(total: RunStats, part: RunStats) -> RunStats:
        return RunStats(
            events_popped=total.events_popped + part.events_popped,
            stale_predictions=total.stale_predictions + part.stale_predictions,
            peak_queue_occupancy=max(
                total.peak_queue_occupancy, part.peak_queue_occupancy
            ),
            deliveries_scheduled=(
                total.deliveries_scheduled + part.deliveries_scheduled
            ),
            deliveries_processed=(
                total.deliveries_processed + part.deliveries_processed
            ),
            input_spikes_processed=(
                total.input_spikes_processed + part.input_spikes_processed
            ),
            drive_updates_processed=(
                total.drive_updates_processed + part.drive_updates_processed
            ),
            autonomous_spikes_confirmed=(
                total.autonomous_spikes_confirmed
                + part.autonomous_spikes_confirmed
            ),
            output_spikes=total.output_spikes + part.output_spikes,
            refractory_releases_processed=(
                total.refractory_releases_processed
                + part.refractory_releases_processed
            ),
            max_same_time_cascade_depth=max(
                total.max_same_time_cascade_depth,
                part.max_same_time_cascade_depth,
            ),
        )

    def advance_until(
        self,
        until: float,
        *,
        inputs: Sequence[MixedInputSpike] = (),
        drive_updates: Sequence[MixedDriveUpdate] = (),
        modulations: Sequence[ModulationEvent] = (),
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> MixedRunResult:
        """Process events strictly before ``until`` and leave it open."""

        return self._advance(
            float(until),
            seal=False,
            inputs=inputs,
            drive_updates=drive_updates,
            modulations=modulations,
            recording=recording,
            inspections=inspections,
        )

    def reset_episode(
        self,
        initial_states: Sequence[float | Sequence[float]],
    ) -> None:
        """Reset dynamic and learning-trace state while preserving weights."""

        self._require_active()
        state_array, _ = self._compiled._core._pack_mixed_initial(
            self._compiled.runtime_layouts, initial_states
        )
        error = self._types._CNetworkError()
        status = self._compiled._core._lib.lc_mixed_run_reset_episode(
            self._handle,
            state_array,
            len(state_array),
            ctypes.byref(error),
        )
        if status != _LC_OK:
            self._failed = True
            self._compiled._core._raise_mixed_status(status, error)

    def finish(
        self,
        *,
        inputs: Sequence[MixedInputSpike] = (),
        drive_updates: Sequence[MixedDriveUpdate] = (),
        modulations: Sequence[ModulationEvent] = (),
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> MixedRunResult:
        """Process the final boundary, emit final state, and seal the run."""

        return self._advance(
            float(self._config.t_end),
            seal=True,
            inputs=inputs,
            drive_updates=drive_updates,
            modulations=modulations,
            recording=recording,
            inspections=inspections,
        )

    def _advance(
        self,
        until: float,
        *,
        seal: bool,
        inputs: Sequence[MixedInputSpike],
        drive_updates: Sequence[MixedDriveUpdate],
        modulations: Sequence[ModulationEvent],
        recording: RecordingConfig | None,
        inspections: Sequence[StateInspectionRequest],
    ) -> MixedRunResult:
        self._require_active()
        self._compiled._core._check_decoder_run(self._decoder_run)
        requested_until = float(until)
        until = self._types.time_value(requested_until).value
        if requested_until > self._frontier and until <= self._frontier:
            raise PrecisionResolutionError("positive run interval does not advance time")
        final_horizon = float(self._config.t_end)
        if (
            not math.isfinite(until)
            or until < self._frontier
            or until > final_horizon
            or (seal and until != final_horizon)
        ):
            raise ValueError("incremental frontier must be nondecreasing and in range")
        input_items = tuple(inputs)
        drive_items = tuple(drive_updates)
        modulation_items = tuple(modulations)
        for item in input_items:
            if not isinstance(item, MixedInputSpike) or not (
                self._frontier <= self._types.time_value(float(item.t)).value <= until
                if seal
                else self._frontier <= self._types.time_value(float(item.t)).value < until
            ):
                raise ValueError("incremental input lies outside this open interval")
        for item in drive_items:
            if not isinstance(item, MixedDriveUpdate) or not (
                self._frontier <= self._types.time_value(float(item.t)).value <= until
                if seal
                else self._frontier <= self._types.time_value(float(item.t)).value < until
            ):
                raise ValueError("incremental drive update lies outside this interval")
        for item in modulation_items:
            if not isinstance(item, ModulationEvent) or not (
                self._frontier <= self._types.time_value(float(item.t)).value <= until
                if seal
                else self._frontier <= self._types.time_value(float(item.t)).value < until
            ):
                raise ValueError("incremental modulation lies outside this interval")

        normalized_inspections = tuple(inspections)
        if len(normalized_inspections) > 0xFFFFFFFF or any(
            not isinstance(request, StateInspectionRequest)
            for request in normalized_inspections
        ):
            raise ValueError("inspections must contain bounded inspection requests")
        ordered_inspections = []
        for original_index, request in enumerate(normalized_inspections):
            t = self._types.time_value(float(request.t)).value
            if not math.isfinite(t) or not (
                self._frontier <= t <= until
                if seal
                else self._frontier <= t < until
            ):
                raise ValueError("inspection lies outside this incremental interval")
            if (
                not isinstance(request.node, int)
                or isinstance(request.node, bool)
                or not 0 <= request.node < self._compiled.node_count
            ):
                raise ValueError("state inspection references an invalid node")
            count = self._compiled.state_layouts[request.node][1]
            selected = (
                tuple(range(count))
                if request.state_indices is None
                else tuple(request.state_indices)
            )
            if len(selected) != len(set(selected)) or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < count
                for index in selected
            ):
                raise ValueError("state inspection indices are invalid")
            ordered_inspections.append((t, original_index, request.node, selected))
        ordered_inspections.sort(key=lambda item: (item[0], item[1]))
        inspection_request_array = (
            self._types._CStateInspectionRequest * len(ordered_inspections)
        )(*(self._types._CStateInspectionRequest(t, node) for t, _, node, _ in ordered_inspections))
        inspection_result_array = (
            self._types._CStateInspectionResult * max(len(ordered_inspections), 1)
        )()
        inspection_count = ctypes.c_uint64()
        inspection_config = (
            self._types._CStateInspectionConfig(
                inspection_request_array,
                len(ordered_inspections),
                inspection_result_array,
                len(ordered_inspections),
                ctypes.pointer(inspection_count),
            )
            if ordered_inspections
            else None
        )

        core = self._compiled._core
        modulation_array = core._pack_modulations(modulation_items)
        input_array = core._pack_mixed_inputs(input_items)
        drive_array = core._pack_mixed_drives(
            self._compiled.runtime_layouts, drive_items
        )
        output_capacity = int(self._config.output_capacity)
        output_array = (self._types._COutputSpike * max(output_capacity, 1))()
        output_count = ctypes.c_uint64()
        stats = self._types._CRunStats()
        error = self._types._CNetworkError()

        trace_config = None
        trace_array = None
        trace_callback = None
        trace_count = ctypes.c_uint64()
        callback_errors: list[BaseException] = []
        if recording is not None:
            if not isinstance(recording.capture_state, bool):
                raise ValueError("recording capture_state must be boolean")
            state_indices = recording.state_indices
            if state_indices is not None:
                state_indices = tuple(state_indices)
                if (
                    not recording.capture_state
                    or len(state_indices) != len(set(state_indices))
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        for index in state_indices
                    )
                ):
                    raise ValueError("recording state indices are invalid")
            if (
                not isinstance(recording.capacity, int)
                or isinstance(recording.capacity, bool)
                or not 0 <= recording.capacity <= 2**64 - 1
            ):
                raise ValueError("recording capacity must be unsigned 64-bit")
            kinds = set(TraceKind) if recording.kinds is None else set(recording.kinds)
            if any(not isinstance(kind, TraceKind) for kind in kinds):
                raise ValueError("recording kinds must contain TraceKind values")
            kind_mask = sum(1 << int(kind) for kind in kinds)
            node_mask_array = None
            node_mask_count = 0
            if recording.nodes is not None:
                nodes = tuple(recording.nodes)
                if len(nodes) != len(set(nodes)) or any(
                    not isinstance(node, int)
                    or isinstance(node, bool)
                    or not 0 <= node < self._compiled.node_count
                    for node in nodes
                ):
                    raise ValueError("recording nodes are invalid")
                node_mask_array = (ctypes.c_uint8 * self._compiled.node_count)()
                for node in nodes:
                    node_mask_array[node] = 1
                node_mask_count = self._compiled.node_count
            selected_nodes = (
                range(self._compiled.node_count)
                if recording.nodes is None
                else recording.nodes
            )
            if state_indices is not None and any(
                index >= self._compiled.state_layouts[node][1]
                for node in selected_nodes
                for index in state_indices
            ):
                raise ValueError("recording state index is absent from a selected node")
            null_callback = self._types._CTraceConsumer()
            if kind_mask != 0 and recording.consumer is None:
                if recording.capacity == 0:
                    raise ValueError("buffered recording requires positive capacity")
                trace_array = (self._types._CTraceRecord * recording.capacity)()
                records_pointer = trace_array
                capacity = recording.capacity
                consumer_pointer = null_callback
            elif kind_mask != 0:
                if not callable(recording.consumer):
                    raise ValueError("recording consumer must be callable")

                def consume(
                    record: ctypes.POINTER(_CTraceRecord),
                    _context: ctypes.c_void_p,
                ) -> int:
                    """Project one C trace record into the Python consumer."""

                    try:
                        assert recording.consumer is not None
                        recording.consumer(core._trace_record(record.contents, state_indices))
                    except BaseException as exc:
                        callback_errors.append(exc)
                        return _LC_INVALID_ARGUMENT
                    return _LC_OK

                trace_callback = self._types._CTraceConsumer(consume)
                records_pointer = None
                capacity = 0
                consumer_pointer = trace_callback
            else:
                records_pointer = None
                capacity = 0
                consumer_pointer = null_callback
            trace_config = self._types._CTraceConfig(
                kind_mask,
                node_mask_array,
                node_mask_count,
                int(recording.capture_state),
                records_pointer,
                capacity,
                ctypes.pointer(trace_count),
                consumer_pointer,
                None,
                0,
            )

        status = core._lib.lc_mixed_run_schedule_modulations(
            self._handle, modulation_array, len(modulation_array)
        )
        if status != _LC_OK:
            self._failed = True
            core._raise(status)
        status = core._lib.lc_mixed_run_advance_incremental(
            self._handle,
            input_array,
            len(input_array),
            drive_array,
            len(drive_array),
            until,
            int(seal),
            output_array,
            ctypes.byref(output_count),
            ctypes.byref(stats),
            ctypes.byref(error),
            None if self._decoder_run is None else self._decoder_run._handle,
            None if trace_config is None else ctypes.byref(trace_config),
            None if inspection_config is None else ctypes.byref(inspection_config),
        )
        if callback_errors:
            self._failed = True
            raise callback_errors[0]
        if status != _LC_OK:
            self._failed = True
            core._raise_mixed_status(status, error)

        state_array = None
        t_last_array = None
        if self._return_final_state:
            state_array = (self._types.real_array(self._compiled.state_count))()
            t_last_array = (self._types.time_array(self._compiled.node_count))()
            status = core._lib.lc_mixed_run_copy_state(
                self._handle,
                state_array,
                len(state_array),
                t_last_array,
                len(t_last_array),
            )
            if status != _LC_OK:
                self._failed = True
                core._raise(status)

        part_stats = core._mixed_stats(stats)
        self._cumulative_stats = self._accumulate_stats(
            self._cumulative_stats, part_stats
        )
        self._frontier = until
        decoded = ()
        decoded_events = ()
        if self._decoder_run is not None:
            all_events = self._decoder_run.events()
            decoded_events = all_events[self._decoder_event_cursor :]
            self._decoder_event_cursor = len(all_events)
            if seal:
                decoded = self._decoder_run.finalize()
        trace_records = (
            ()
            if trace_array is None
            else tuple(
                core._trace_record(trace_array[index], recording.state_indices)
                for index in range(trace_count.value)
            )
        )
        inspection_results: list[StateInspection | None] = [None] * len(
            ordered_inspections
        )
        if inspection_count.value != len(ordered_inspections):
            self._failed = True
            raise CoreError("state inspection result count mismatch")
        for ordered_index, (_, original_index, node, selected) in enumerate(
            ordered_inspections
        ):
            item = inspection_result_array[ordered_index]
            inspection_results[original_index] = StateInspection(
                t=float(item.t),
                node=int(item.node),
                generation=int(item.generation),
                clamped=bool(item.clamped),
                state_indices=selected,
                values=tuple(float(item.values[index]) for index in selected),
            )
        self._finished = seal
        weights, plasticity, learning_observers = core._copy_learning_state(
            self._handle,
            self._compiled.edge_count,
            self._compiled.plastic_edge_count,
            self._compiled.node_count,
        )
        return MixedRunResult(
            states=(
                ()
                if state_array is None or t_last_array is None
                else tuple(
                    AugmentedState(
                        tuple(state_array[offset + local] for local in range(count)),
                        t_last_array[index],
                    )
                    for index, (offset, count) in enumerate(self._compiled.state_layouts)
                )
            ),
            spikes=tuple(
                Spike(output_array[index].t, output_array[index].node)
                for index in range(output_count.value)
            ),
            stats=part_stats,
            kernel_seconds=float(stats.kernel_seconds),
            decoded=decoded,
            decoded_events=decoded_events,
            trace=trace_records,
            inspections=tuple(item for item in inspection_results if item is not None),
            weights=weights,
            plasticity=plasticity,
            learning_observers=learning_observers,
        )


class CompiledRun:
    """Mutable state and scheduler storage for one compiled-graph execution."""

    def __init__(self, compiled: CompiledGraph, handle: ctypes.c_void_p):
        self._types = compiled._core._types
        self._compiled = compiled
        self._handle = handle
        self._ready = True

    @property
    def closed(self) -> bool:
        """Return whether mutable run state was released."""

        return not bool(self._handle and self._handle.value)

    def close(self) -> None:
        """Release mutable run state once."""

        if not self.closed:
            self._compiled._core._lib.lc_mixed_run_destroy(self._handle)
            self._handle = ctypes.c_void_p()
            self._ready = False

    def __enter__(self) -> "CompiledRun":
        if self.closed:
            raise RuntimeError("compiled run is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset(
        self,
        initial_states: Sequence[float | Sequence[float]],
    ) -> None:
        """Restore initial state and the compiled graph's default parameters."""

        if self.closed:
            raise RuntimeError("compiled run is closed")
        state_array, t_last_array = self._compiled._core._pack_mixed_initial(
            self._compiled.runtime_layouts, initial_states
        )
        status = self._compiled._core._lib.lc_mixed_run_reset(
            self._handle,
            state_array,
            len(state_array),
            t_last_array,
            len(t_last_array),
        )
        if status != _LC_OK:
            self._compiled._core._raise(status)
        self._ready = True

    def execute(
        self,
        *,
        inputs: Sequence[MixedInputSpike] = (),
        drive_updates: Sequence[MixedDriveUpdate] = (),
        modulations: Sequence[ModulationEvent] = (),
        t_end: float,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        decoder_run: StreamingDecoderRun | None = None,
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
        return_final_state: bool = True,
    ) -> MixedRunResult:
        """Execute one complete run and return requested observations."""

        if self.closed:
            raise RuntimeError("compiled run is closed")
        if not self._ready:
            raise RuntimeError("compiled run must be reset before another execution")
        self._compiled._core._check_decoder_run(decoder_run)
        if not isinstance(return_final_state, bool):
            raise ValueError("return_final_state must be boolean")
        if (
            queue_capacity <= 0
            or output_capacity < 0
            or same_time_cascade_limit <= 0
            or not isinstance(stochastic_seed, int)
            or isinstance(stochastic_seed, bool)
            or not 0 <= stochastic_seed <= 2**64 - 1
        ):
            raise ValueError("compiled runner capacities and cascade limit are invalid")
        normalized_inspections = tuple(inspections)
        if len(normalized_inspections) > 0xFFFFFFFF:
            raise ValueError("state inspection request count must fit uint32")
        if any(
            not isinstance(request, StateInspectionRequest)
            for request in normalized_inspections
        ):
            raise ValueError(
                "inspections must contain StateInspectionRequest values"
            )
        ordered_inspections = []
        for original_index, request in enumerate(normalized_inspections):
            t = float(request.t)
            if not math.isfinite(t) or t < 0.0 or t > float(t_end):
                raise ValueError(
                    "state inspection times must be finite and inside the run"
                )
            if (
                not isinstance(request.node, int)
                or isinstance(request.node, bool)
                or not 0 <= request.node < self._compiled.node_count
            ):
                raise ValueError("state inspection references an invalid node")
            count = self._compiled.state_layouts[request.node][1]
            state_indices = request.state_indices
            if state_indices is None:
                selected = tuple(range(count))
            else:
                selected = tuple(state_indices)
                if (
                    len(selected) != len(set(selected))
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or not 0 <= index < count
                        for index in selected
                    )
                ):
                    raise ValueError(
                        "state inspection indices must be unique valid local indices"
                    )
            ordered_inspections.append(
                (t, original_index, request.node, selected)
            )
        ordered_inspections.sort(key=lambda item: (item[0], item[1]))
        inspection_request_array = (
            self._types._CStateInspectionRequest * len(ordered_inspections)
        )(
            *(
                self._types._CStateInspectionRequest(t, node)
                for t, _, node, _ in ordered_inspections
            )
        )
        inspection_result_array = (
            self._types._CStateInspectionResult * max(len(ordered_inspections), 1)
        )()
        inspection_count = ctypes.c_uint64()
        inspection_config = (
            self._types._CStateInspectionConfig(
                inspection_request_array,
                len(ordered_inspections),
                inspection_result_array,
                len(ordered_inspections),
                ctypes.pointer(inspection_count),
            )
            if ordered_inspections
            else None
        )
        core = self._compiled._core
        modulation_array = core._pack_modulations(modulations)
        input_array = core._pack_mixed_inputs(inputs)
        drive_array = core._pack_mixed_drives(
            self._compiled.runtime_layouts, drive_updates
        )
        output_array = (self._types._COutputSpike * max(output_capacity, 1))()
        output_count = ctypes.c_uint64()
        stats = self._types._CRunStats()
        error = self._types._CNetworkError()
        config = self._types._CRunConfig(
            float(t_end),
            queue_capacity,
            output_capacity,
            same_time_cascade_limit,
            stochastic_seed,
        )
        arguments = (
            self._handle,
            input_array,
            len(input_array),
            drive_array,
            len(drive_array),
            ctypes.byref(config),
            output_array,
            ctypes.byref(output_count),
            ctypes.byref(stats),
            ctypes.byref(error),
        )
        trace_config = None
        trace_array = None
        trace_callback = None
        trace_count = ctypes.c_uint64()
        callback_errors: list[BaseException] = []
        if recording is not None:
            if not isinstance(recording.capture_state, bool):
                raise ValueError("recording capture_state must be boolean")
            state_indices = recording.state_indices
            if state_indices is not None:
                state_indices = tuple(state_indices)
                if (
                    not recording.capture_state
                    or len(state_indices) != len(set(state_indices))
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        for index in state_indices
                    )
                ):
                    raise ValueError(
                        "recording state_indices require capture_state and unique "
                        "nonnegative integers"
                    )
            if (
                not isinstance(recording.capacity, int)
                or isinstance(recording.capacity, bool)
                or not 0 <= recording.capacity <= 0xFFFFFFFFFFFFFFFF
            ):
                raise ValueError("recording capacity must be an unsigned 64-bit integer")
            kinds = set(TraceKind) if recording.kinds is None else set(recording.kinds)
            if any(not isinstance(kind, TraceKind) for kind in kinds):
                raise ValueError("recording kinds must contain TraceKind values")
            kind_mask = sum(1 << int(kind) for kind in kinds)
            node_mask_array = None
            node_mask_count = 0
            if recording.nodes is not None:
                nodes = tuple(recording.nodes)
                if len(nodes) != len(set(nodes)) or any(
                    not isinstance(node, int)
                    or isinstance(node, bool)
                    or node < 0
                    or node >= self._compiled.node_count
                    for node in nodes
                ):
                    raise ValueError("recording nodes must be unique valid node indices")
                node_mask_array = (ctypes.c_uint8 * self._compiled.node_count)()
                for node in nodes:
                    node_mask_array[node] = 1
                node_mask_count = self._compiled.node_count
            selected_nodes = (
                range(self._compiled.node_count)
                if recording.nodes is None
                else recording.nodes
            )
            if state_indices is not None and any(
                index >= self._compiled.state_layouts[node][1]
                for node in selected_nodes
                for index in state_indices
            ):
                raise ValueError(
                    "recording state_indices must exist on every selected node"
                )
            null_callback = self._types._CTraceConsumer()
            if kind_mask != 0 and recording.consumer is None:
                if recording.capacity == 0:
                    raise ValueError("buffered recording requires positive capacity")
                trace_array = (self._types._CTraceRecord * recording.capacity)()
                records_pointer = trace_array
                capacity = recording.capacity
                consumer_pointer = null_callback
            elif kind_mask != 0:
                if not callable(recording.consumer):
                    raise ValueError("recording consumer must be callable")

                def consume(
                    record: ctypes.POINTER(_CTraceRecord), _context: ctypes.c_void_p
                ) -> int:
                    """Project one C trace record into the Python consumer."""

                    try:
                        assert recording.consumer is not None
                        recording.consumer(
                            core._trace_record(record.contents, state_indices)
                        )
                    except BaseException as exc:  # Re-raise after C returns.
                        callback_errors.append(exc)
                        return _LC_INVALID_ARGUMENT
                    return _LC_OK

                trace_callback = self._types._CTraceConsumer(consume)
                records_pointer = None
                capacity = 0
                consumer_pointer = trace_callback
            else:
                records_pointer = None
                capacity = 0
                consumer_pointer = null_callback
            trace_config = self._types._CTraceConfig(
                kind_mask,
                node_mask_array,
                node_mask_count,
                int(recording.capture_state),
                records_pointer,
                capacity,
                ctypes.pointer(trace_count),
                consumer_pointer,
                None,
                0,
            )
        status = core._lib.lc_mixed_run_schedule_modulations(
            self._handle, modulation_array, len(modulation_array)
        )
        if status != _LC_OK:
            self._ready = False
            core._raise(status)
        if inspection_config is not None:
            status = core._lib.lc_mixed_run_execute_observed(
                *arguments,
                None if decoder_run is None else decoder_run._handle,
                None if trace_config is None else ctypes.byref(trace_config),
                ctypes.byref(inspection_config),
            )
        elif trace_config is None:
            if decoder_run is None:
                status = core._lib.lc_mixed_run_execute(*arguments)
            else:
                status = core._lib.lc_mixed_run_execute_with_decoders(
                    *arguments, decoder_run._handle
                )
        elif decoder_run is None:
            status = core._lib.lc_mixed_run_execute_recorded(
                *arguments, ctypes.byref(trace_config)
            )
        else:
            status = core._lib.lc_mixed_run_execute_with_decoders_recorded(
                *arguments, decoder_run._handle, ctypes.byref(trace_config)
            )
        self._ready = False
        if callback_errors:
            raise callback_errors[0]
        if status != _LC_OK:
            core._raise_mixed_status(status, error)
        state_array = None
        t_last_array = None
        if return_final_state:
            state_array = (self._types.real_array(self._compiled.state_count))()
            t_last_array = (self._types.time_array(self._compiled.node_count))()
            status = core._lib.lc_mixed_run_copy_state(
                self._handle,
                state_array,
                len(state_array),
                t_last_array,
                len(t_last_array),
            )
            if status != _LC_OK:
                core._raise(status)
        decoded = () if decoder_run is None else decoder_run.finalize()
        decoded_events = () if decoder_run is None else decoder_run.events()
        trace_records = (
            ()
            if trace_array is None
            else tuple(
                core._trace_record(trace_array[index], recording.state_indices)
                for index in range(trace_count.value)
            )
        )
        inspection_results: list[StateInspection | None] = [
            None
        ] * len(ordered_inspections)
        if inspection_count.value != len(ordered_inspections):
            raise CoreError(
                "state inspection result count does not match the request count"
            )
        for ordered_index, (_, original_index, node, selected) in enumerate(
            ordered_inspections
        ):
            item = inspection_result_array[ordered_index]
            expected_count = self._compiled.state_layouts[node][1]
            if item.node != node or item.state_count != expected_count:
                raise CoreError("state inspection result layout mismatch")
            inspection_results[original_index] = StateInspection(
                t=float(item.t),
                node=int(item.node),
                generation=int(item.generation),
                clamped=bool(item.clamped),
                state_indices=selected,
                values=tuple(float(item.values[index]) for index in selected),
            )
        if any(item is None for item in inspection_results):
            raise CoreError("state inspection result mapping is incomplete")
        weights, plasticity, learning_observers = core._copy_learning_state(
            self._handle,
            self._compiled.edge_count,
            self._compiled.plastic_edge_count,
            self._compiled.node_count,
        )
        return MixedRunResult(
            states=(
                ()
                if state_array is None or t_last_array is None
                else tuple(
                    AugmentedState(
                        tuple(state_array[offset + local] for local in range(count)),
                        t_last_array[index],
                    )
                    for index, (offset, count) in enumerate(self._compiled.state_layouts)
                )
            ),
            spikes=tuple(
                Spike(output_array[index].t, output_array[index].node)
                for index in range(output_count.value)
            ),
            stats=core._mixed_stats(stats),
            kernel_seconds=float(stats.kernel_seconds),
            decoded=decoded,
            decoded_events=decoded_events,
            trace=trace_records,
            inspections=tuple(
                item for item in inspection_results if item is not None
            ),
            weights=weights,
            plasticity=plasticity,
            learning_observers=learning_observers,
        )
