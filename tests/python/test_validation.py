from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna.audit import TraceAuditError, audit_causal_trace
from lacuna.ffi import CoreEvaluator, RecordingConfig, TraceKind, TracePhase
from lacuna.validation import (
    chain_case,
    driven_population_case,
    evaluate_case,
    fanout_fanin_case,
)


@pytest.fixture(scope="module")
def causal_trace_samples(core: CoreEvaluator):
    samples = {}
    for key, case in (
        ("chain", chain_case(3)),
        ("driven", driven_population_case(1)),
    ):
        resolved = case.graph.resolve()
        with resolved.compile(core) as runner:
            result = runner.run(
                spike_inputs=case.spike_inputs,
                drive_inputs=case.drive_inputs,
                t_end=case.t_end,
                recording=RecordingConfig(capacity=256),
            ).core
        samples[key] = (case, resolved, result)
    return samples


def _audit(sample, result=None):
    case, resolved, source = sample
    return audit_causal_trace(
        source if result is None else result,
        models=resolved.models,
        edges=resolved.edges,
        t_end=case.t_end,
        expected_input_count=len(case.spike_inputs),
        expected_drive_count=len(case.drive_inputs),
    )


def _record_index(result, kind: TraceKind) -> int:
    return next(index for index, record in enumerate(result.trace) if record.kind is kind)


def _change_record(result, index: int, **changes):
    records = list(result.trace)
    records[index] = replace(records[index], **changes)
    return replace(result, trace=tuple(records))


def _remove_record(result, index: int, *, next_changes=None):
    records = list(result.trace)
    del records[index]
    if next_changes is not None:
        records[index] = replace(records[index], **next_changes)
    records = [replace(record, sequence=sequence) for sequence, record in enumerate(records)]
    return replace(result, trace=tuple(records))


def test_generated_chain_network_passes_campaign_invariants(core: CoreEvaluator) -> None:
    report = evaluate_case(core, chain_case(16), timed_repetitions=1)
    assert report.output_spikes == 16
    assert report.deliveries == 15
    assert report.deterministic_replays == 3


def test_generated_fanout_network_exercises_queue_boundary(core: CoreEvaluator) -> None:
    report = evaluate_case(core, fanout_fanin_case(12), timed_repetitions=1)
    assert report.output_spikes == 14
    assert report.peak_queue_occupancy > 1
    assert report.queue_headroom_at_test_capacity >= 0


def test_complete_causal_trace_passes_independent_audit(causal_trace_samples) -> None:
    report = _audit(causal_trace_samples["chain"])
    assert report.record_count == 18
    assert report.delivery_records == 2
    assert report.spike_records == 3
    assert report.final_state_records == 3
    assert len(report.checks) == 11


def test_auditor_rejects_each_adversarial_invariant_violation(
    causal_trace_samples,
) -> None:
    chain = causal_trace_samples["chain"]
    driven = causal_trace_samples["driven"]
    chain_result = chain[2]
    driven_result = driven[2]

    reset_index = _record_index(chain_result, TraceKind.RESET)
    release_index = _record_index(driven_result, TraceKind.REFRACTORY_RELEASE)
    deposit_index = _record_index(chain_result, TraceKind.DEPOSIT_APPLY)
    delivery_index = _record_index(chain_result, TraceKind.DELIVERY)
    confirmed_index = _record_index(driven_result, TraceKind.PREDICTION_CONFIRMED)
    final_index = _record_index(chain_result, TraceKind.FINAL_STATE)

    changed_states = list(chain_result.states)
    changed_states[0] = replace(
        changed_states[0], values=(changed_states[0].values[0] + 1.0,)
    )

    corruptions = (
        (
            "trace_sequence_and_time",
            chain,
            _change_record(chain_result, 0, sequence=9),
        ),
        (
            "trace_phase_and_payload_contract",
            chain,
            _change_record(chain_result, 0, phase=TracePhase.BOUNDARY),
        ),
        (
            "full_state_snapshot_contract",
            chain,
            _change_record(chain_result, final_index, before=(), after=()),
        ),
        (
            "same_time_state_continuity",
            chain,
            _change_record(chain_result, reset_index, before=(-44.0,)),
        ),
        (
            "spike_reset_pairing",
            chain,
            _remove_record(
                chain_result,
                reset_index,
                next_changes={"before": (-45.0,), "after": (-45.0,)},
            ),
        ),
        (
            "refractory_state_machine",
            driven,
            _change_record(
                driven_result,
                release_index,
                generation=driven_result.trace[release_index].generation + 1,
            ),
        ),
        (
            "same_time_deposit_causality",
            chain,
            _remove_record(chain_result, deposit_index),
        ),
        (
            "delivery_conservation",
            chain,
            _change_record(
                chain_result,
                delivery_index,
                value=chain_result.trace[delivery_index].value + 1.0,
            ),
        ),
        (
            "prediction_generation_consistency",
            driven,
            _change_record(
                driven_result,
                confirmed_index,
                generation=driven_result.trace[confirmed_index].generation + 1,
            ),
        ),
        (
            "runtime_accounting_from_trace",
            chain,
            replace(
                chain_result,
                stats=replace(
                    chain_result.stats,
                    events_popped=chain_result.stats.events_popped + 1,
                ),
            ),
        ),
        (
            "final_state_agreement",
            chain,
            replace(chain_result, states=tuple(changed_states)),
        ),
    )

    for expected_invariant, sample, corrupted in corruptions:
        with pytest.raises(TraceAuditError) as caught:
            _audit(sample, corrupted)
        assert caught.value.invariant == expected_invariant
