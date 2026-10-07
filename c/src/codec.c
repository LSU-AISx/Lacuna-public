#include "lacuna.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Encode scalar presentations and decode output spikes without Python arithmetic. */

static int lc_codec_allocation_size_overflows(
    uint64_t count,
    size_t element_size
) {
    return element_size != 0U && count > (uint64_t)(SIZE_MAX / element_size);
}

static uint64_t lc_splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

/* Draw from a deterministic stream identified by the encoder and draw index. */
static lc_status lc_exponential_draw(
    lc_encoder_state *state,
    uint64_t stream,
    lc_real_t *draw
) {
    uint64_t bits = lc_splitmix64(
        state->seed ^ (stream * UINT64_C(0xd1342543de82ef95)) ^ state->draw_index
    );
    lc_real_t uniform;
    if (state->draw_index == UINT64_MAX || draw == NULL) {
        return LC_NUMERIC_ERROR;
    }
    state->draw_index += 1U;
#if LACUNA_REAL_BITS == 16
    /* Binary16 half-bin points stay strictly inside (0, 1). */
    uniform = ((lc_real_t)(bits >> 54U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1.0p-10);
#elif LACUNA_REAL_BITS == 32
    /* Half-bin points cannot round to either endpoint in binary32. */
    uniform = ((lc_real_t)(bits >> 41U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1.0p-23);
#else
    uniform = ((lc_real_t)(bits >> 11U) + LC_REAL_C(0.5)) * (LC_REAL_C(1.0) / LC_REAL_C(9007199254740992.0));
#endif
    *draw = -lc_real_log(uniform);
#if LACUNA_REAL_BITS == 16
    if (!lc_isfinite(*draw) || *draw <= LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
#endif
    return LC_OK;
}

/* Append one encoded spike while preserving chronological production order. */
static lc_status lc_append_spike(
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count,
    lc_time_t t,
    uint32_t encoder,
    lc_real_t value
) {
    if (*count >= capacity) {
        return LC_OUTPUT_OVERFLOW;
    }
    outputs[*count].t = t;
    outputs[*count].encoder = encoder;
    outputs[*count].value = value;
    *count += 1U;
    return LC_OK;
}

/* Append one held-current update to the caller-owned output buffer. */
static lc_status lc_append_drive(
    lc_encoded_drive *outputs,
    uint64_t capacity,
    uint64_t *count,
    lc_time_t t,
    uint32_t encoder,
    lc_real_t value
) {
    if (*count >= capacity) {
        return LC_OUTPUT_OVERFLOW;
    }
    outputs[*count].t = t;
    outputs[*count].encoder = encoder;
    outputs[*count].value = value;
    *count += 1U;
    return LC_OK;
}

/* Validate parameters before an encoder can mutate its persistent state. */
static lc_status lc_validate_encoder(const lc_encoder_spec *spec) {
    if (spec == NULL || !lc_isfinite(spec->amplitude)) {
        return LC_INVALID_ARGUMENT;
    }
    switch (spec->kind) {
        case LC_ENCODER_NATIVE_EVENT:
            return LC_OK;
        case LC_ENCODER_REGULAR_RATE:
        case LC_ENCODER_POISSON_RATE:
        case LC_ENCODER_BURST:
            if (!lc_isfinite(spec->rate_min) || !lc_isfinite(spec->rate_max) ||
                spec->rate_min < LC_REAL_C(0.0) || spec->rate_max < spec->rate_min) {
                return LC_INVALID_ARGUMENT;
            }
            if (spec->kind == LC_ENCODER_BURST &&
                (!lc_isfinite(spec->duration) || spec->duration <= LC_TIME_C(0.0))) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        case LC_ENCODER_TTFS:
            if (!lc_isfinite(spec->latency_min) ||
                !lc_isfinite(spec->latency_max) || spec->latency_min < LC_TIME_C(0.0) ||
                spec->latency_max < spec->latency_min ||
                !lc_isfinite(spec->silence_threshold) ||
                spec->silence_threshold < LC_REAL_C(0.0) || spec->silence_threshold > LC_REAL_C(1.0)) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        case LC_ENCODER_LATENCY_BURST:
            if (!lc_isfinite(spec->latency_min) ||
                !lc_isfinite(spec->latency_max) || spec->latency_min < LC_TIME_C(0.0) ||
                spec->latency_max < spec->latency_min ||
                !lc_isfinite(spec->rate_max) || spec->rate_max <= LC_REAL_C(0.0) ||
                !lc_isfinite(spec->duration) || spec->duration <= LC_TIME_C(0.0) ||
                !lc_isfinite(spec->silence_threshold) ||
                spec->silence_threshold < LC_REAL_C(0.0) || spec->silence_threshold > LC_REAL_C(1.0)) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        case LC_ENCODER_HELD_CURRENT:
            if (!lc_isfinite(spec->gain) || !lc_isfinite(spec->offset) ||
                !lc_isfinite(spec->baseline)) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        default:
            return LC_INVALID_ARGUMENT;
    }
}

static lc_real_t lc_linear_rate(const lc_encoder_spec *spec, lc_real_t value) {
    return spec->rate_min + value * (spec->rate_max - spec->rate_min);
}

static lc_time_t lc_inverse_latency(const lc_encoder_spec *spec, lc_real_t value) {
    return spec->latency_max - (lc_time_t)value *
        (spec->latency_max - spec->latency_min);
}

/* A positive encoded interval must advance the target clock. */
static int lc_codec_time_sum_valid(lc_time_t base, lc_time_t interval, lc_time_t t) {
    return lc_isfinite(interval) && lc_isfinite(t) && interval >= LC_TIME_C(0.0) &&
        t >= base && (interval == LC_TIME_C(0.0) || t > base);
}

/* Clock differences enter model arithmetic only through a checked conversion. */
static int lc_codec_real_interval(lc_time_t end, lc_time_t start, lc_real_t *value) {
    lc_time_t interval = end - start;
    if (!lc_isfinite(interval) || interval < LC_TIME_C(0.0)) return 0;
    *value = (lc_real_t)interval;
    return lc_isfinite(*value) &&
        (interval == LC_TIME_C(0.0) || *value > LC_REAL_C(0.0));
}

static int lc_codec_hazard(
    lc_real_t rate, lc_time_t end, lc_time_t start, lc_real_t *hazard
) {
    lc_real_t interval;
    if (rate == LC_REAL_C(0.0)) {
        *hazard = LC_REAL_C(0.0);
        return 1;
    }
    if (!lc_isfinite(rate) || rate < LC_REAL_C(0.0) ||
        !lc_codec_real_interval(end, start, &interval)) return 0;
    *hazard = rate * interval;
    return lc_isfinite(*hazard) &&
        (interval == LC_REAL_C(0.0) || *hazard > LC_REAL_C(0.0));
}

/* Phase/rate is an interval, so division uses the declared clock precision. */
static int lc_codec_phase_offset(lc_real_t phase, lc_real_t rate, lc_time_t *offset) {
    *offset = (lc_time_t)phase / (lc_time_t)rate;
    return lc_isfinite(*offset) && *offset >= LC_TIME_C(0.0) &&
        (phase == LC_REAL_C(0.0) || *offset > LC_TIME_C(0.0));
}

static int lc_codec_temporal_contribution(
    lc_time_t t, lc_time_t start, lc_real_t tau, lc_real_t *value
) {
    lc_real_t elapsed;
    if (!lc_codec_real_interval(t, start, &elapsed)) return 0;
    *value = lc_real_exp(-elapsed / tau);
    return lc_isfinite(*value);
}

/* Convert accumulated regular-rate phase into spikes within one presentation. */
static lc_status lc_encode_regular(
    const lc_encoder_spec *spec,
    lc_encoder_state *state,
    const lc_presentation *presentation,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_real_t rate = lc_linear_rate(spec, presentation->value);
    lc_real_t hazard;
    lc_real_t remaining = state->phase_remaining;
    lc_time_t previous = presentation->t_start;
    int emitted = 0;
    lc_status status;

    if (!lc_codec_hazard(rate, presentation->t_end, presentation->t_start, &hazard)) {
        return LC_NUMERIC_ERROR;
    }
    if (rate == LC_REAL_C(0.0)) {
        return LC_OK;
    }
    while (remaining <= hazard) {
        lc_time_t offset;
        lc_time_t t;
        if (!lc_codec_phase_offset(remaining, rate, &offset)) return LC_NUMERIC_ERROR;
        t = presentation->t_start + offset;
        if (!lc_codec_time_sum_valid(presentation->t_start, offset, t) ||
            (emitted && t <= previous)) {
            return LC_NUMERIC_ERROR;
        }
        if (!(t < presentation->t_end)) {
            break;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, presentation->encoder, spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        previous = t;
        emitted = 1;
        if (!(remaining + LC_REAL_C(1.0) > remaining)) return LC_NUMERIC_ERROR;
        remaining += LC_REAL_C(1.0);
    }
    remaining -= hazard;
#if LACUNA_REAL_BITS == 16
    if (remaining < LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
#endif
    if (remaining < LC_REAL_C(0.0) && remaining > -LC_REAL_C(32.0) * LC_REAL_EPSILON) {
        remaining = LC_REAL_C(0.0);
    }
    state->phase_remaining = remaining;
    return LC_OK;
}

/* Consume exponential hazard draws over one constant-rate presentation. */
static lc_status lc_encode_poisson(
    const lc_encoder_spec *spec,
    lc_encoder_state *state,
    const lc_presentation *presentation,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_real_t rate = lc_linear_rate(spec, presentation->value);
    lc_real_t hazard;
    lc_real_t consumed = LC_REAL_C(0.0);
    lc_real_t remaining;
    lc_time_t previous = presentation->t_start;
    int emitted = 0;
    lc_status status;

    if (!lc_codec_hazard(rate, presentation->t_end, presentation->t_start, &hazard)) {
        return LC_NUMERIC_ERROR;
    }
    if (!state->poisson_ready) {
        status = lc_exponential_draw(
            state, spec->stream, &state->poisson_remaining
        );
        if (status != LC_OK) {
            return status;
        }
        state->poisson_ready = 1U;
    }
    if (rate == LC_REAL_C(0.0)) {
        return LC_OK;
    }
    remaining = state->poisson_remaining;
    while (remaining <= hazard - consumed) {
        lc_real_t crossing = consumed + remaining;
        lc_time_t offset;
        lc_time_t t;
        if (!lc_codec_phase_offset(crossing, rate, &offset)) return LC_NUMERIC_ERROR;
        t = presentation->t_start + offset;
        if ((remaining > LC_REAL_C(0.0) && crossing <= consumed) ||
            !lc_codec_time_sum_valid(presentation->t_start, offset, t) ||
            (emitted && remaining > LC_REAL_C(0.0) && t <= previous)) {
            return LC_NUMERIC_ERROR;
        }
        if (!(t < presentation->t_end)) {
            remaining = LC_REAL_C(0.0);
            consumed = hazard;
            break;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, presentation->encoder, spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        previous = t;
        emitted = 1;
        consumed = crossing;
        status = lc_exponential_draw(state, spec->stream, &remaining);
        if (status != LC_OK) {
            return status;
        }
    }
    remaining -= hazard - consumed;
#if LACUNA_REAL_BITS == 16
    if (remaining < LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
#endif
    if (remaining < LC_REAL_C(0.0) && remaining > -LC_REAL_C(32.0) * LC_REAL_EPSILON) {
        remaining = LC_REAL_C(0.0);
    }
    state->poisson_remaining = remaining;
    return LC_OK;
}

/* Emit a bounded periodic burst from a resolved onset and rate. */
static lc_status lc_encode_periodic_burst(
    const lc_encoder_spec *spec,
    const lc_presentation *presentation,
    lc_time_t onset,
    lc_real_t rate,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_time_t stop = onset + spec->duration;
    uint64_t index = 0U;
    lc_status status;
    if (!lc_isfinite(onset) || !lc_codec_time_sum_valid(onset, spec->duration, stop)) {
        return LC_NUMERIC_ERROR;
    }
    if (stop > presentation->t_end) {
        stop = presentation->t_end;
    }
    if (rate <= LC_REAL_C(0.0) || !(onset < stop)) {
        return LC_OK;
    }
    for (;;) {
        lc_time_t t = onset + (lc_time_t)index / (lc_time_t)rate;
        if (!(t < stop)) {
            break;
        }
        if (!lc_isfinite(t) || (index > 0U &&
            t <= onset + (lc_time_t)(index - 1U) / (lc_time_t)rate)) {
            return LC_NUMERIC_ERROR;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, presentation->encoder, spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        if (index == UINT64_MAX) {
            return LC_NUMERIC_ERROR;
        }
        index += 1U;
    }
    return LC_OK;
}

typedef struct lc_live_encoder_state {
    lc_encoder_state continuity;
    lc_presentation presentation;
    lc_time_t cursor;
    lc_time_t last_start;
    uint64_t burst_index;
    uint32_t active;
    uint32_t single_emitted;
    uint32_t has_start;
} lc_live_encoder_state;

struct lc_encoder_run {
    lc_encoder_spec *specs;
    lc_live_encoder_state *states;
    uint32_t spec_count;
    lc_time_t frontier;
    uint32_t finished;
};

typedef enum lc_live_boundary_mode {
    LC_LIVE_REPLACEMENT = 0,
    LC_LIVE_OPEN = 1,
    LC_LIVE_SEALED = 2
} lc_live_boundary_mode;

/* Apply the sealed-boundary rule used by all live encoder variants. */
static int lc_live_at_or_before(
    lc_time_t t,
    lc_time_t boundary,
    uint32_t include_boundary
) {
    return t < boundary || (include_boundary && t == boundary);
}

/* Live encoders preserve phase across incremental simulation boundaries. */
static lc_status lc_live_encode_regular(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t end,
    uint32_t include_end,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_real_t rate = lc_linear_rate(spec, state->presentation.value);
    lc_real_t hazard;
    lc_real_t remaining = state->continuity.phase_remaining;
    lc_time_t previous = state->cursor;
    int emitted = 0;
    lc_status status;

    if (!lc_codec_hazard(rate, end, state->cursor, &hazard)) {
        return LC_NUMERIC_ERROR;
    }
    if (rate == LC_REAL_C(0.0)) {
        return LC_OK;
    }
    while (remaining <= hazard) {
        lc_time_t offset;
        lc_time_t t;
        if (!lc_codec_phase_offset(remaining, rate, &offset)) return LC_NUMERIC_ERROR;
        t = state->cursor + offset;
        if (!lc_codec_time_sum_valid(state->cursor, offset, t) ||
            (emitted && t <= previous)) {
            return LC_NUMERIC_ERROR;
        }
        if (!lc_live_at_or_before(t, end, include_end)) {
            break;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, state->presentation.encoder,
            spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        previous = t;
        emitted = 1;
        if (!(remaining + LC_REAL_C(1.0) > remaining)) return LC_NUMERIC_ERROR;
        remaining += LC_REAL_C(1.0);
    }
    remaining -= hazard;
#if LACUNA_REAL_BITS == 16
    if (remaining < LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
#endif
    if (remaining < LC_REAL_C(0.0) && remaining > -LC_REAL_C(32.0) * LC_REAL_EPSILON) {
        remaining = LC_REAL_C(0.0);
    }
    state->continuity.phase_remaining = remaining;
    return LC_OK;
}

/* Advance one Poisson stream without discarding its unused hazard draw. */
static lc_status lc_live_encode_poisson(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t end,
    uint32_t include_end,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_real_t rate = lc_linear_rate(spec, state->presentation.value);
    lc_real_t hazard;
    lc_real_t consumed = LC_REAL_C(0.0);
    lc_real_t remaining;
    lc_time_t previous = state->cursor;
    int emitted = 0;
    lc_status status;

    if (!lc_codec_hazard(rate, end, state->cursor, &hazard)) {
        return LC_NUMERIC_ERROR;
    }
    if (!state->continuity.poisson_ready) {
        status = lc_exponential_draw(
            &state->continuity, spec->stream,
            &state->continuity.poisson_remaining
        );
        if (status != LC_OK) {
            return status;
        }
        state->continuity.poisson_ready = 1U;
    }
    if (rate == LC_REAL_C(0.0)) {
        return LC_OK;
    }
    remaining = state->continuity.poisson_remaining;
    while (remaining <= hazard - consumed) {
        lc_real_t crossing = consumed + remaining;
        lc_time_t offset;
        lc_time_t t;
        if (!lc_codec_phase_offset(crossing, rate, &offset)) return LC_NUMERIC_ERROR;
        t = state->cursor + offset;
        if ((remaining > LC_REAL_C(0.0) && crossing <= consumed) ||
            !lc_codec_time_sum_valid(state->cursor, offset, t) ||
            (emitted && remaining > LC_REAL_C(0.0) && t <= previous)) {
            return LC_NUMERIC_ERROR;
        }
        if (!lc_live_at_or_before(t, end, include_end)) {
            remaining = LC_REAL_C(0.0);
            consumed = hazard;
            break;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, state->presentation.encoder,
            spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        previous = t;
        emitted = 1;
        consumed = crossing;
        status = lc_exponential_draw(&state->continuity, spec->stream, &remaining);
        if (status != LC_OK) {
            return status;
        }
    }
    remaining -= hazard - consumed;
#if LACUNA_REAL_BITS == 16
    if (remaining < LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
#endif
    if (remaining < LC_REAL_C(0.0) && remaining > -LC_REAL_C(32.0) * LC_REAL_EPSILON) {
        remaining = LC_REAL_C(0.0);
    }
    state->continuity.poisson_remaining = remaining;
    return LC_OK;
}

/* Emit a pending single spike when it enters the current open interval. */
static lc_status lc_live_encode_single(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t end,
    uint32_t include_end,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_time_t onset;
    if (state->single_emitted ||
        state->presentation.value <= spec->silence_threshold) {
        return LC_OK;
    }
    onset = state->presentation.t_start +
            lc_inverse_latency(spec, state->presentation.value);
    if (!lc_codec_time_sum_valid(
            state->presentation.t_start,
            lc_inverse_latency(spec, state->presentation.value), onset)) {
        return LC_NUMERIC_ERROR;
    }
    if (onset < state->cursor || !(onset < state->presentation.t_end) ||
        !lc_live_at_or_before(onset, end, include_end)) {
        return LC_OK;
    }
    if (lc_append_spike(
            outputs, capacity, count, onset, state->presentation.encoder,
            spec->amplitude
        ) != LC_OK) {
        return LC_OUTPUT_OVERFLOW;
    }
    state->single_emitted = 1U;
    return LC_OK;
}

/* Emit the pending portion of an active burst and retain its next phase. */
static lc_status lc_live_encode_burst(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t end,
    uint32_t include_end,
    lc_encoded_spike *outputs,
    uint64_t capacity,
    uint64_t *count
) {
    lc_time_t onset = state->presentation.t_start;
    lc_time_t stop;
    lc_real_t rate;

    if (spec->kind == LC_ENCODER_LATENCY_BURST) {
        if (state->presentation.value <= spec->silence_threshold) {
            return LC_OK;
        }
        onset += lc_inverse_latency(spec, state->presentation.value);
        rate = spec->rate_max;
    } else {
        rate = lc_linear_rate(spec, state->presentation.value);
    }
    stop = onset + spec->duration;
    if (!lc_isfinite(onset) || !lc_codec_time_sum_valid(onset, spec->duration, stop) ||
        (spec->kind == LC_ENCODER_LATENCY_BURST && !lc_codec_time_sum_valid(
            state->presentation.t_start,
            lc_inverse_latency(spec, state->presentation.value), onset))) {
        return LC_NUMERIC_ERROR;
    }
    if (stop > state->presentation.t_end) {
        stop = state->presentation.t_end;
    }
    if (rate <= LC_REAL_C(0.0) || !(onset < stop)) {
        return LC_OK;
    }
    for (;;) {
        lc_time_t t = onset + (lc_time_t)state->burst_index / (lc_time_t)rate;
        lc_status status;
        if (!(t < stop) || !lc_live_at_or_before(t, end, include_end)) {
            break;
        }
        if (!lc_isfinite(t) || (state->burst_index > 0U &&
            t <= onset + (lc_time_t)(state->burst_index - 1U) / (lc_time_t)rate)) {
            return LC_NUMERIC_ERROR;
        }
        if (t < state->cursor) {
            if (state->burst_index == UINT64_MAX) {
                return LC_NUMERIC_ERROR;
            }
            state->burst_index += 1U;
            continue;
        }
        status = lc_append_spike(
            outputs, capacity, count, t, state->presentation.encoder,
            spec->amplitude
        );
        if (status != LC_OK) {
            return status;
        }
        if (state->burst_index == UINT64_MAX) {
            return LC_NUMERIC_ERROR;
        }
        state->burst_index += 1U;
    }
    return LC_OK;
}

/* Dispatch one active presentation to its stateful encoder implementation. */
static lc_status lc_live_process_active(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t boundary,
    lc_live_boundary_mode mode,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
    lc_time_t end;
    uint32_t include_spike_end;
    uint32_t ended;
    lc_status status = LC_OK;

    if (!state->active) {
        return LC_OK;
    }
    if (boundary < state->cursor) {
        return LC_TIME_REVERSED;
    }
    end = boundary < state->presentation.t_end
              ? boundary
              : state->presentation.t_end;
    include_spike_end = mode == LC_LIVE_SEALED && end == boundary &&
                        boundary < state->presentation.t_end;
    switch (spec->kind) {
        case LC_ENCODER_REGULAR_RATE:
            status = lc_live_encode_regular(
                spec, state, end, include_spike_end, spikes, spike_capacity,
                spike_count
            );
            break;
        case LC_ENCODER_POISSON_RATE:
            status = lc_live_encode_poisson(
                spec, state, end, include_spike_end, spikes, spike_capacity,
                spike_count
            );
            break;
        case LC_ENCODER_TTFS:
            status = lc_live_encode_single(
                spec, state, end, include_spike_end, spikes, spike_capacity,
                spike_count
            );
            break;
        case LC_ENCODER_BURST:
        case LC_ENCODER_LATENCY_BURST:
            status = lc_live_encode_burst(
                spec, state, end, include_spike_end, spikes, spike_capacity,
                spike_count
            );
            break;
        case LC_ENCODER_HELD_CURRENT:
            break;
        default:
            return LC_INVALID_ARGUMENT;
    }
    if (status != LC_OK) {
        return status;
    }
    state->cursor = end;
    ended = state->presentation.t_end < boundary ||
            (state->presentation.t_end == boundary && mode != LC_LIVE_OPEN);
    if (ended) {
        if (spec->kind == LC_ENCODER_HELD_CURRENT) {
            status = lc_append_drive(
                drives, drive_capacity, drive_count, state->presentation.t_end,
                state->presentation.encoder, spec->baseline
            );
            if (status != LC_OK) {
                return status;
            }
        }
        state->active = 0U;
        state->continuity.last_end = state->presentation.t_end;
    }
    return LC_OK;
}

/* End an active presentation and restore held current when required. */
static lc_status lc_live_cancel_active(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    lc_time_t t,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
    lc_status status = LC_OK;
    if (!state->active) {
        return LC_OK;
    }
    if (spec->kind == LC_ENCODER_HELD_CURRENT) {
        status = lc_append_drive(
            drives, drive_capacity, drive_count, t,
            state->presentation.encoder, spec->baseline
        );
        if (status != LC_OK) {
            return status;
        }
    }
    state->active = 0U;
    state->continuity.last_end = t;
    return LC_OK;
}

/* Replace an encoder presentation after first advancing its previous state. */
static lc_status lc_live_start(
    const lc_encoder_spec *spec,
    lc_live_encoder_state *state,
    const lc_presentation *presentation,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
    lc_real_t value;
    lc_status status;

    state->presentation = *presentation;
    state->cursor = presentation->t_start;
    state->last_start = presentation->t_start;
    state->burst_index = 0U;
    state->active = 1U;
    state->single_emitted = 0U;
    state->has_start = 1U;
    state->continuity.initialized = 1U;
    if (spec->kind != LC_ENCODER_HELD_CURRENT) {
        return LC_OK;
    }
    value = spec->offset + spec->gain * presentation->value;
    if (!lc_isfinite(value)) {
        return LC_NUMERIC_ERROR;
    }
    status = lc_append_drive(
        drives, drive_capacity, drive_count, presentation->t_start,
        presentation->encoder, value
    );
    return status;
}

/* Allocate independent mutable state for a compiled set of encoder specs. */
lc_status lc_encoder_run_create(
    const lc_encoder_spec *specs,
    uint32_t spec_count,
    uint64_t seed,
    lc_time_t initial_frontier,
    lc_encoder_run **run
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (run != NULL) *run = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_encoder_run *created;
    uint32_t index;
    if (run == NULL || !lc_isfinite(initial_frontier) ||
        (spec_count > 0U && specs == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *run = NULL;
    for (index = 0U; index < spec_count; ++index) {
        lc_status status = lc_validate_encoder(&specs[index]);
        if (status != LC_OK) {
            return status;
        }
    }
    created = (lc_encoder_run *)calloc(1U, sizeof(lc_encoder_run));
    if (created == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    if (spec_count > 0U) {
        created->specs = (lc_encoder_spec *)malloc(
            (size_t)spec_count * sizeof(lc_encoder_spec)
        );
        created->states = (lc_live_encoder_state *)calloc(
            (size_t)spec_count, sizeof(lc_live_encoder_state)
        );
        if (created->specs == NULL || created->states == NULL) {
            lc_encoder_run_destroy(created);
            return LC_ALLOCATION_FAILED;
        }
        memcpy(
            created->specs, specs,
            (size_t)spec_count * sizeof(lc_encoder_spec)
        );
        for (index = 0U; index < spec_count; ++index) {
            lc_status status = lc_encoder_state_reset(
                &created->states[index].continuity, 1U, seed
            );
            if (status != LC_OK) {
                lc_encoder_run_destroy(created);
                return status;
            }
        }
    }
    created->spec_count = spec_count;
    created->frontier = initial_frontier;
    *run = created;
    return LC_OK;
}

/* Advance live encoders transactionally to an open or sealed frontier. */
lc_status lc_encoder_run_advance(
    lc_encoder_run *run,
    const lc_presentation *presentations,
    uint32_t presentation_count,
    lc_time_t until,
    uint32_t seal,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_live_encoder_state *staged = NULL;
    uint32_t *heads = NULL;
    uint32_t *tails = NULL;
    uint32_t *next = NULL;
    uint32_t index;
    lc_status final_status = LC_OK;

    if (run == NULL || spike_count == NULL || drive_count == NULL ||
        run->finished || !lc_isfinite(until) || until < run->frontier ||
        seal > 1U || (presentation_count > 0U && presentations == NULL) ||
        (spike_capacity > 0U && spikes == NULL) ||
        (drive_capacity > 0U && drives == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *spike_count = 0U;
    *drive_count = 0U;
    if (run->spec_count > 0U) {
        staged = (lc_live_encoder_state *)malloc(
            (size_t)run->spec_count * sizeof(lc_live_encoder_state)
        );
        heads = (uint32_t *)malloc((size_t)run->spec_count * sizeof(uint32_t));
        tails = (uint32_t *)malloc((size_t)run->spec_count * sizeof(uint32_t));
        if (staged == NULL || heads == NULL || tails == NULL) {
            final_status = LC_ALLOCATION_FAILED;
            goto cleanup;
        }
        memcpy(
            staged, run->states,
            (size_t)run->spec_count * sizeof(lc_live_encoder_state)
        );
        for (index = 0U; index < run->spec_count; ++index) {
            heads[index] = UINT32_MAX;
            tails[index] = UINT32_MAX;
        }
    }
    if (presentation_count > 0U) {
        next = (uint32_t *)malloc((size_t)presentation_count * sizeof(uint32_t));
        if (next == NULL) {
            final_status = LC_ALLOCATION_FAILED;
            goto cleanup;
        }
    }
    for (index = 0U; index < presentation_count; ++index) {
        const lc_presentation *item = &presentations[index];
        uint32_t encoder = item->encoder;
        if (encoder >= run->spec_count ||
            run->specs[encoder].kind == LC_ENCODER_NATIVE_EVENT ||
            !lc_isfinite(item->t_start) || !lc_isfinite(item->t_end) ||
            !(item->t_start < item->t_end) || !lc_isfinite(item->value) ||
            item->value < LC_REAL_C(0.0) || item->value > LC_REAL_C(1.0) ||
            item->t_start < run->frontier ||
            (seal ? item->t_start > until : item->t_start >= until)) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (index > 0U &&
            (item->t_start < presentations[index - 1U].t_start ||
             (item->t_start == presentations[index - 1U].t_start &&
              item->encoder < presentations[index - 1U].encoder))) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
        if ((staged[encoder].has_start &&
             item->t_start <= staged[encoder].last_start) ||
            (tails[encoder] != UINT32_MAX &&
             item->t_start <= presentations[tails[encoder]].t_start)) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
        next[index] = UINT32_MAX;
        if (heads[encoder] == UINT32_MAX) {
            heads[encoder] = index;
        } else {
            next[tails[encoder]] = index;
        }
        tails[encoder] = index;
    }
    for (index = 0U; index < run->spec_count; ++index) {
        lc_live_encoder_state *state = &staged[index];
        const lc_encoder_spec *spec = &run->specs[index];
        uint32_t presentation_index = heads[index];
        while (presentation_index != UINT32_MAX) {
            const lc_presentation *item = &presentations[presentation_index];
            final_status = lc_live_process_active(
                spec, state, item->t_start, LC_LIVE_REPLACEMENT,
                spikes, spike_capacity, spike_count,
                drives, drive_capacity, drive_count
            );
            if (final_status != LC_OK) {
                goto cleanup;
            }
            final_status = lc_live_cancel_active(
                spec, state, item->t_start,
                drives, drive_capacity, drive_count
            );
            if (final_status != LC_OK) {
                goto cleanup;
            }
            final_status = lc_live_start(
                spec, state, item, drives, drive_capacity, drive_count
            );
            if (final_status != LC_OK) {
                goto cleanup;
            }
            presentation_index = next[presentation_index];
        }
        final_status = lc_live_process_active(
            spec, state, until,
            seal ? LC_LIVE_SEALED : LC_LIVE_OPEN,
            spikes, spike_capacity, spike_count,
            drives, drive_capacity, drive_count
        );
        if (final_status != LC_OK) {
            goto cleanup;
        }
    }
    if (run->spec_count > 0U) {
        memcpy(
            run->states, staged,
            (size_t)run->spec_count * sizeof(lc_live_encoder_state)
        );
    }
    run->frontier = until;
    run->finished = seal;
cleanup:
    free(next);
    free(tails);
    free(heads);
    free(staged);
    if (final_status != LC_OK) {
        *spike_count = 0U;
        *drive_count = 0U;
    }
    return final_status;
}

lc_status lc_encoder_run_reset_episode(lc_encoder_run *run) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint32_t index;
    if (run == NULL || run->finished || !lc_isfinite(run->frontier)) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < run->spec_count; ++index) {
        lc_encoder_state continuity = run->states[index].continuity;
        uint64_t draw_index = continuity.draw_index;
        uint64_t seed = continuity.seed;
        memset(&run->states[index], 0, sizeof(lc_live_encoder_state));
        if (lc_encoder_state_reset(
                &run->states[index].continuity, 1U, seed
            ) != LC_OK) {
            return LC_NUMERIC_ERROR;
        }
        run->states[index].continuity.draw_index = draw_index;
    }
    return LC_OK;
}

void lc_encoder_run_destroy(lc_encoder_run *run) {
    if (run == NULL) {
        return;
    }
    free(run->states);
    free(run->specs);
    free(run);
}

/* Reset caller-owned encoder states and derive independent random streams. */
lc_status lc_encoder_state_reset(
    lc_encoder_state *states,
    uint32_t state_count,
    uint64_t seed
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint32_t index;
    if (state_count > 0U && states == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < state_count; ++index) {
        states[index].phase_remaining = LC_REAL_C(1.0);
        states[index].poisson_remaining = LC_REAL_C(0.0);
        states[index].last_end = -(lc_time_t)INFINITY;
        states[index].draw_index = 0U;
        states[index].seed = seed;
        states[index].initialized = 0U;
        states[index].poisson_ready = 0U;
    }
    return LC_OK;
}

/* Encode a sorted batch while carrying phase across adjacent presentations. */
lc_status lc_encode_presentations(
    const lc_encoder_spec *specs,
    uint32_t spec_count,
    lc_encoder_state *states,
    uint32_t state_count,
    const lc_presentation *presentations,
    uint32_t presentation_count,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint32_t index;
    lc_time_t *next_starts = NULL;
    uint32_t *next_indices = NULL;
    lc_status final_status = LC_OK;
    if (spec_count != state_count || spike_count == NULL || drive_count == NULL ||
        (spec_count > 0U && (specs == NULL || states == NULL)) ||
        (presentation_count > 0U && presentations == NULL) ||
        (spike_capacity > 0U && spikes == NULL) ||
        (drive_capacity > 0U && drives == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *spike_count = 0U;
    *drive_count = 0U;
    for (index = 0U; index < spec_count; ++index) {
        lc_status status = lc_validate_encoder(&specs[index]);
        if (status != LC_OK) {
            return status;
        }
    }
    if (presentation_count > 0U) {
        next_starts = (lc_time_t *)malloc(
            (size_t)presentation_count * sizeof(lc_time_t)
        );
        if (next_starts == NULL) {
            return LC_ALLOCATION_FAILED;
        }
    }
    if (spec_count > 0U) {
        next_indices = (uint32_t *)malloc((size_t)spec_count * sizeof(uint32_t));
        if (next_indices == NULL) {
            free(next_starts);
            return LC_ALLOCATION_FAILED;
        }
        for (index = 0U; index < spec_count; ++index) {
            next_indices[index] = UINT32_MAX;
        }
    }
    for (index = 0U; index < presentation_count; ++index) {
        const lc_presentation *item = &presentations[index];
        if (item->encoder >= spec_count || !lc_isfinite(item->t_start) ||
            !lc_isfinite(item->t_end) || !(item->t_start < item->t_end) ||
            !lc_isfinite(item->value) || item->value < LC_REAL_C(0.0) || item->value > LC_REAL_C(1.0)) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
    }
    for (index = presentation_count; index > 0U; --index) {
        uint32_t current = index - 1U;
        uint32_t encoder = presentations[current].encoder;
        uint32_t next = next_indices[encoder];
        next_starts[current] = (lc_time_t)INFINITY;
        if (next != UINT32_MAX) {
            next_starts[current] = presentations[next].t_start;
            if (next_starts[current] <= presentations[current].t_start) {
                final_status = LC_INVALID_ARGUMENT;
                goto cleanup;
            }
        }
        next_indices[encoder] = current;
    }
    for (index = 0U; index < presentation_count; ++index) {
        lc_presentation effective = presentations[index];
        const lc_presentation *item = &effective;
        const lc_encoder_spec *spec;
        lc_encoder_state *state;
        lc_status status = LC_OK;
        lc_time_t onset;
        if (next_starts[index] < effective.t_end) {
            effective.t_end = next_starts[index];
        }
        spec = &specs[item->encoder];
        state = &states[item->encoder];
        if (item->t_start < state->last_end) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (spec->kind == LC_ENCODER_NATIVE_EVENT) {
            final_status = LC_INVALID_ARGUMENT;
            goto cleanup;
        }
        switch (spec->kind) {
            case LC_ENCODER_REGULAR_RATE:
                status = lc_encode_regular(
                    spec, state, item, spikes, spike_capacity, spike_count
                );
                break;
            case LC_ENCODER_POISSON_RATE:
                status = lc_encode_poisson(
                    spec, state, item, spikes, spike_capacity, spike_count
                );
                break;
            case LC_ENCODER_TTFS:
                if (item->value > spec->silence_threshold) {
                    onset = item->t_start + lc_inverse_latency(spec, item->value);
                    if (!lc_codec_time_sum_valid(
                            item->t_start, lc_inverse_latency(spec, item->value), onset)) {
                        final_status = LC_NUMERIC_ERROR;
                        goto cleanup;
                    }
                    if (onset < item->t_end) {
                        status = lc_append_spike(
                            spikes, spike_capacity, spike_count, onset,
                            item->encoder, spec->amplitude
                        );
                    }
                }
                break;
            case LC_ENCODER_BURST:
                status = lc_encode_periodic_burst(
                    spec, item, item->t_start, lc_linear_rate(spec, item->value),
                    spikes, spike_capacity, spike_count
                );
                break;
            case LC_ENCODER_LATENCY_BURST:
                if (item->value > spec->silence_threshold) {
                    onset = item->t_start + lc_inverse_latency(spec, item->value);
                    if (!lc_codec_time_sum_valid(
                            item->t_start, lc_inverse_latency(spec, item->value), onset)) {
                        final_status = LC_NUMERIC_ERROR;
                        goto cleanup;
                    }
                    status = lc_encode_periodic_burst(
                        spec, item, onset, spec->rate_max, spikes,
                        spike_capacity, spike_count
                    );
                }
                break;
            case LC_ENCODER_HELD_CURRENT: {
                lc_real_t value = spec->offset + spec->gain * item->value;
                if (!lc_isfinite(value)) {
                    final_status = LC_NUMERIC_ERROR;
                    goto cleanup;
                }
                status = lc_append_drive(
                    drives, drive_capacity, drive_count, item->t_start,
                    item->encoder, value
                );
                if (status == LC_OK) {
                    status = lc_append_drive(
                        drives, drive_capacity, drive_count, item->t_end,
                        item->encoder, spec->baseline
                    );
                }
                break;
            }
            default:
                final_status = LC_INVALID_ARGUMENT;
                goto cleanup;
        }
        if (status != LC_OK) {
            final_status = status;
            goto cleanup;
        }
        state->last_end = effective.t_end;
        state->initialized = 1U;
    }
cleanup:
    free(next_indices);
    free(next_starts);
    return final_status;
}

/* Decoder banks own sorted schedules while decoder runs own observations. */

static lc_status lc_validate_decoder(const lc_decoder_spec *spec) {
    if (spec == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    switch (spec->kind) {
        case LC_DECODER_RATE:
            if ((spec->emission != LC_EMIT_ON_WINDOW_CLOSE &&
                 spec->emission != LC_EMIT_ON_QUERY) ||
                spec->mode > LC_RATE_CUMULATIVE ||
                (spec->mode == LC_RATE_SLIDING &&
                 (!lc_isfinite(spec->width) || spec->width <= LC_TIME_C(0.0))) ||
                (spec->mode == LC_RATE_SLIDING &&
                 spec->emission == LC_EMIT_ON_QUERY) ||
                (spec->mode == LC_RATE_CUMULATIVE &&
                 !lc_isfinite(spec->origin))) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        case LC_DECODER_TTFS:
            return spec->normalize <= 1U &&
                           (spec->emission == LC_EMIT_ON_EVENT ||
                            spec->emission == LC_EMIT_ON_WINDOW_CLOSE ||
                            spec->emission == LC_EMIT_ON_QUERY)
                       ? LC_OK
                       : LC_INVALID_ARGUMENT;
        case LC_DECODER_TEMPORAL_WEIGHT:
            if (!lc_isfinite(spec->tau) || spec->tau <= LC_REAL_C(0.0) ||
                spec->first_only > 1U || spec->normalize > 1U ||
                spec->emission > LC_EMIT_ON_QUERY) {
                return LC_INVALID_ARGUMENT;
            }
            return LC_OK;
        default:
            return LC_INVALID_ARGUMENT;
    }
}

/* Decode a complete half-open window without allocating persistent state. */
lc_status lc_decode_spikes(
    const lc_decoder_spec *specs,
    uint32_t spec_count,
    const lc_output_spike *spikes,
    uint64_t spike_count,
    const lc_decode_window *window,
    lc_decode_result *results,
    uint32_t result_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint32_t decoder;
    if (result_count != spec_count || window == NULL ||
        (spec_count > 0U && (specs == NULL || results == NULL)) ||
        (spike_count > 0U && spikes == NULL) || !lc_isfinite(window->t_start) ||
        !lc_isfinite(window->t_end) || !(window->t_start < window->t_end)) {
        return LC_INVALID_ARGUMENT;
    }
    for (decoder = 0U; decoder < spec_count; ++decoder) {
        const lc_decoder_spec *spec = &specs[decoder];
        lc_decode_result *result = &results[decoder];
        lc_time_t start = window->t_start;
        lc_time_t end = window->t_end;
        uint64_t index;
        lc_real_t weighted = LC_REAL_C(0.0);
        lc_status status = lc_validate_decoder(spec);
        if (status != LC_OK) {
            return status;
        }
        if (spec->kind == LC_DECODER_RATE) {
            if (spec->mode == LC_RATE_SLIDING && start < end - spec->width) {
                start = end - spec->width;
            } else if (spec->mode == LC_RATE_CUMULATIVE) {
                start = spec->origin;
            }
        }
        if (!(start < end)) {
            return LC_INVALID_ARGUMENT;
        }
        result->decoder = decoder;
        result->window = 0U;
        result->valid = spec->kind == LC_DECODER_TTFS ? 0U : 1U;
        result->count = 0U;
        result->value = LC_REAL_C(0.0);
        result->first_spike = LC_TIME_C(0.0);
        result->window_start = start;
        result->window_end = end;
        for (index = 0U; index < spike_count; ++index) {
            lc_time_t t = spikes[index].t;
            if (!lc_isfinite(t)) {
                return LC_INVALID_ARGUMENT;
            }
            if (spikes[index].node != spec->node || !(t >= start && t < end)) {
                continue;
            }
            if (result->count == 0U) {
                result->first_spike = t;
            }
            result->count += 1U;
            if (spec->kind == LC_DECODER_TEMPORAL_WEIGHT) {
                lc_real_t contribution;
                if (!lc_codec_temporal_contribution(t, start, spec->tau, &contribution))
                    return LC_NUMERIC_ERROR;
                weighted += contribution;
                if (!lc_isfinite(weighted)) return LC_NUMERIC_ERROR;
                if (spec->first_only) {
                    break;
                }
            } else if (spec->kind == LC_DECODER_TTFS) {
                break;
            }
        }
        if (spec->kind == LC_DECODER_RATE) {
            lc_real_t interval;
            if (!lc_codec_real_interval(end, start, &interval) ||
                interval <= LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
            result->value = (lc_real_t)result->count / interval;
        } else if (spec->kind == LC_DECODER_TTFS) {
            if (result->count > 0U) {
                result->valid = 1U;
                if (!lc_codec_real_interval(result->first_spike, start, &result->value))
                    return LC_NUMERIC_ERROR;
                if (spec->normalize) {
                    lc_real_t interval;
                    if (!lc_codec_real_interval(end, start, &interval) ||
                        interval <= LC_REAL_C(0.0)) return LC_NUMERIC_ERROR;
                    result->value /= interval;
                }
            }
        } else {
            result->value = weighted;
            if (spec->normalize && result->count > 0U) {
                result->value /= (lc_real_t)result->count;
            }
        }
        if (!lc_isfinite(result->value)) return LC_NUMERIC_ERROR;
    }
    return LC_OK;
}

typedef struct lc_stream_decoder_state {
    lc_time_t start;
    uint64_t count;
    lc_real_t weighted;
    lc_time_t first_spike;
} lc_stream_decoder_state;

typedef struct lc_decoder_query_item {
    lc_time_t t;
    uint64_t entry;
} lc_decoder_query_item;

struct lc_compiled_decoders {
    uint64_t references;
    uint32_t spec_count;
    uint32_t node_count;
    lc_decoder_spec *specs;
    uint64_t *node_offsets;
    uint32_t *decoder_indices;
};

struct lc_decoder_run {
    lc_compiled_decoders *compiled;
    lc_decode_window *windows;
    lc_stream_decoder_state *states;
    uint8_t *window_closed;
    uint32_t *entry_decoders;
    uint32_t *window_ids;
    uint64_t entry_count;
    uint64_t *close_order;
    uint64_t close_position;
    lc_decoder_query_item *queries;
    uint64_t query_count;
    uint64_t query_position;
    uint64_t *node_offsets;
    uint64_t *node_entries;
    lc_decoded_event *events;
    uint64_t event_capacity;
    uint64_t event_count;
    lc_time_t last_observation;
    int ready;
    int has_observation;
    int sealed;
};

typedef struct lc_decoder_close_key {
    lc_time_t end;
    uint64_t entry;
} lc_decoder_close_key;

typedef struct lc_decoder_identity_key {
    uint32_t decoder;
    uint32_t window;
} lc_decoder_identity_key;

static int lc_compare_decoder_close_key(const void *left, const void *right) {
    const lc_decoder_close_key *a = (const lc_decoder_close_key *)left;
    const lc_decoder_close_key *b = (const lc_decoder_close_key *)right;
    if (a->end < b->end) {
        return -1;
    }
    if (a->end > b->end) {
        return 1;
    }
    return a->entry < b->entry ? -1 : a->entry > b->entry;
}

static int lc_compare_decoder_identity_key(const void *left, const void *right) {
    const lc_decoder_identity_key *a = (const lc_decoder_identity_key *)left;
    const lc_decoder_identity_key *b = (const lc_decoder_identity_key *)right;
    if (a->decoder < b->decoder) {
        return -1;
    }
    if (a->decoder > b->decoder) {
        return 1;
    }
    return a->window < b->window ? -1 : a->window > b->window;
}

static int lc_compare_decoder_query_item(const void *left, const void *right) {
    const lc_decoder_query_item *a = (const lc_decoder_query_item *)left;
    const lc_decoder_query_item *b = (const lc_decoder_query_item *)right;
    if (a->t < b->t) {
        return -1;
    }
    if (a->t > b->t) {
        return 1;
    }
    return a->entry < b->entry ? -1 : a->entry > b->entry;
}

static void lc_decoder_bank_release(lc_compiled_decoders *compiled) {
    if (compiled == NULL) {
        return;
    }
    if (compiled->references > 1U) {
        compiled->references--;
        return;
    }
    free(compiled->specs);
    free(compiled->node_offsets);
    free(compiled->decoder_indices);
    free(compiled);
}

/* Copy decoder specs and build the canonical node-to-decoder index. */
lc_status lc_decoder_bank_compile(
    const lc_decoder_spec *specs,
    uint32_t spec_count,
    uint32_t node_count,
    lc_compiled_decoders **compiled
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (compiled != NULL) *compiled = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_decoders *result;
    uint64_t *cursors;
    uint32_t index;
    size_t node;
    if (compiled == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *compiled = NULL;
    if (specs == NULL || spec_count == 0U || node_count == 0U ||
        lc_codec_allocation_size_overflows(spec_count, sizeof(lc_decoder_spec)) ||
        lc_codec_allocation_size_overflows(spec_count, sizeof(uint32_t)) ||
        lc_codec_allocation_size_overflows(
            (uint64_t)node_count + 1U, sizeof(uint64_t)
        ) ||
        lc_codec_allocation_size_overflows(node_count, sizeof(uint64_t))) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < spec_count; ++index) {
        if (specs[index].node >= node_count ||
            lc_validate_decoder(&specs[index]) != LC_OK) {
            return LC_INVALID_ARGUMENT;
        }
    }
    result = (lc_compiled_decoders *)calloc(1U, sizeof(lc_compiled_decoders));
    cursors = (uint64_t *)calloc(node_count, sizeof(uint64_t));
    if (result == NULL || cursors == NULL) {
        free(cursors);
        free(result);
        return LC_ALLOCATION_FAILED;
    }
    result->references = 1U;
    result->spec_count = spec_count;
    result->node_count = node_count;
    result->specs = (lc_decoder_spec *)calloc(spec_count, sizeof(lc_decoder_spec));
    result->node_offsets = (uint64_t *)calloc(
        (size_t)node_count + 1U, sizeof(uint64_t)
    );
    result->decoder_indices = (uint32_t *)calloc(spec_count, sizeof(uint32_t));
    if (result->specs == NULL || result->node_offsets == NULL ||
        result->decoder_indices == NULL) {
        free(cursors);
        lc_decoder_bank_release(result);
        return LC_ALLOCATION_FAILED;
    }
    memcpy(result->specs, specs, spec_count * sizeof(lc_decoder_spec));
    for (index = 0U; index < spec_count; ++index) {
        result->node_offsets[specs[index].node + 1U]++;
    }
    for (node = 1U; node <= (size_t)node_count; ++node) {
        result->node_offsets[node] += result->node_offsets[node - 1U];
    }
    memcpy(cursors, result->node_offsets, node_count * sizeof(uint64_t));
    for (index = 0U; index < spec_count; ++index) {
        uint32_t target = specs[index].node;
        result->decoder_indices[cursors[target]++] = index;
    }
    free(cursors);
    *compiled = result;
    return LC_OK;
}

void lc_decoder_bank_destroy(lc_compiled_decoders *compiled) {
    lc_decoder_bank_release(compiled);
}

/* Allocate mutable counters and schedules for one decoder execution. */
lc_status lc_decoder_run_create(
    lc_compiled_decoders *compiled,
    lc_decoder_run **run
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (run != NULL) *run = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_decoder_run *result;
    if (run == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *run = NULL;
    if (compiled == NULL || compiled->references == UINT64_MAX) {
        return LC_INVALID_ARGUMENT;
    }
    result = (lc_decoder_run *)calloc(1U, sizeof(lc_decoder_run));
    if (result == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    result->compiled = compiled;
    compiled->references++;
    *run = result;
    return LC_OK;
}

/* Replace decoded-event storage only while the run is inactive. */
lc_status lc_decoder_run_reserve_events(
    lc_decoder_run *run,
    uint64_t capacity
) {
    lc_decoded_event *events;
    if (run == NULL || run->compiled == NULL || run->event_count != 0U ||
        lc_codec_allocation_size_overflows(capacity, sizeof(lc_decoded_event))) {
        return LC_INVALID_ARGUMENT;
    }
    if (capacity == 0U) {
        free(run->events);
        run->events = NULL;
        run->event_capacity = 0U;
        return LC_OK;
    }
    events = (lc_decoded_event *)realloc(
        run->events, (size_t)capacity * sizeof(lc_decoded_event)
    );
    if (events == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    run->events = events;
    run->event_capacity = capacity;
    return LC_OK;
}

/* Reset every decoder onto one shared observation window. */
lc_status lc_decoder_run_reset(
    lc_decoder_run *run,
    const lc_decode_window *window
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_decoder_run_reset_windows(run, window, 1U);
}

/* Expand shared windows into the canonical decoder schedule. */
lc_status lc_decoder_run_reset_windows(
    lc_decoder_run *run,
    const lc_decode_window *windows,
    uint32_t window_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_decoder_window_binding *bindings;
    uint64_t binding_count;
    uint64_t entry = 0U;
    uint32_t window;
    uint32_t decoder;
    lc_status status;
    if (run == NULL || run->compiled == NULL || windows == NULL ||
        window_count == 0U) {
        return LC_INVALID_ARGUMENT;
    }
    binding_count = (uint64_t)window_count * run->compiled->spec_count;
    if (lc_codec_allocation_size_overflows(
            binding_count, sizeof(lc_decoder_window_binding)
        )) {
        return LC_INVALID_ARGUMENT;
    }
    bindings = (lc_decoder_window_binding *)malloc(
        (size_t)binding_count * sizeof(lc_decoder_window_binding)
    );
    if (bindings == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    for (window = 0U; window < window_count; ++window) {
        if (!lc_isfinite(windows[window].t_start) ||
            !lc_isfinite(windows[window].t_end) ||
            !(windows[window].t_start < windows[window].t_end) ||
            (window > 0U && windows[window].t_end < windows[window - 1U].t_end)) {
            free(bindings);
            return LC_INVALID_ARGUMENT;
        }
        for (decoder = 0U; decoder < run->compiled->spec_count; ++decoder) {
            bindings[entry].decoder = decoder;
            bindings[entry].window = window;
            bindings[entry].t_start = windows[window].t_start;
            bindings[entry].t_end = windows[window].t_end;
            entry++;
        }
    }
    status = lc_decoder_run_reset_schedule(run, bindings, binding_count);
    free(bindings);
    return status;
}

/* Validate and install sparse decoder windows in close-time order. */
lc_status lc_decoder_run_reset_schedule(
    lc_decoder_run *run,
    const lc_decoder_window_binding *bindings,
    uint64_t binding_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_decode_window *new_windows;
    lc_stream_decoder_state *new_states;
    uint8_t *new_closed;
    uint32_t *new_decoders;
    uint32_t *new_window_ids;
    uint64_t *new_close_order;
    uint64_t *new_node_offsets;
    uint64_t *new_node_entries;
    lc_decoder_close_key *close_keys;
    lc_decoder_identity_key *identity_keys;
    uint64_t *cursors;
    uint64_t entry;
    uint32_t node;
    size_t node_index;
    if (run == NULL || run->compiled == NULL || bindings == NULL ||
        binding_count == 0U ||
        lc_codec_allocation_size_overflows(binding_count, sizeof(lc_decode_window)) ||
        lc_codec_allocation_size_overflows(
            binding_count, sizeof(lc_stream_decoder_state)
        ) ||
        lc_codec_allocation_size_overflows(binding_count, sizeof(uint8_t)) ||
        lc_codec_allocation_size_overflows(binding_count, sizeof(uint32_t)) ||
        lc_codec_allocation_size_overflows(binding_count, sizeof(uint64_t)) ||
        lc_codec_allocation_size_overflows(
            binding_count, sizeof(lc_decoder_close_key)
        ) ||
        lc_codec_allocation_size_overflows(
            binding_count, sizeof(lc_decoder_identity_key)
        )) {
        return LC_INVALID_ARGUMENT;
    }
    new_windows = (lc_decode_window *)malloc(
        (size_t)binding_count * sizeof(lc_decode_window)
    );
    new_states = (lc_stream_decoder_state *)calloc(
        (size_t)binding_count, sizeof(lc_stream_decoder_state)
    );
    new_closed = (uint8_t *)calloc((size_t)binding_count, sizeof(uint8_t));
    new_decoders = (uint32_t *)malloc(
        (size_t)binding_count * sizeof(uint32_t)
    );
    new_window_ids = (uint32_t *)malloc(
        (size_t)binding_count * sizeof(uint32_t)
    );
    new_close_order = (uint64_t *)malloc(
        (size_t)binding_count * sizeof(uint64_t)
    );
    new_node_offsets = (uint64_t *)calloc(
        (size_t)run->compiled->node_count + 1U, sizeof(uint64_t)
    );
    new_node_entries = (uint64_t *)malloc(
        (size_t)binding_count * sizeof(uint64_t)
    );
    close_keys = (lc_decoder_close_key *)malloc(
        (size_t)binding_count * sizeof(lc_decoder_close_key)
    );
    identity_keys = (lc_decoder_identity_key *)malloc(
        (size_t)binding_count * sizeof(lc_decoder_identity_key)
    );
    cursors = (uint64_t *)calloc(
        run->compiled->node_count, sizeof(uint64_t)
    );
    if (new_windows == NULL || new_states == NULL || new_closed == NULL ||
        new_decoders == NULL || new_window_ids == NULL ||
        new_close_order == NULL || new_node_offsets == NULL ||
        new_node_entries == NULL || close_keys == NULL ||
        identity_keys == NULL || cursors == NULL) {
        free(new_windows);
        free(new_states);
        free(new_closed);
        free(new_decoders);
        free(new_window_ids);
        free(new_close_order);
        free(new_node_offsets);
        free(new_node_entries);
        free(close_keys);
        free(identity_keys);
        free(cursors);
        return LC_ALLOCATION_FAILED;
    }
    for (entry = 0U; entry < binding_count; ++entry) {
        uint32_t decoder = bindings[entry].decoder;
        const lc_decoder_spec *spec;
        lc_time_t start = bindings[entry].t_start;
        if (decoder >= run->compiled->spec_count ||
            !lc_isfinite(start) ||
            !lc_isfinite(bindings[entry].t_end) ||
            !(start < bindings[entry].t_end)) {
            goto invalid_schedule;
        }
        spec = &run->compiled->specs[decoder];
        if (spec->kind == LC_DECODER_RATE) {
            if (spec->mode == LC_RATE_SLIDING &&
                start < bindings[entry].t_end - spec->width) {
                start = bindings[entry].t_end - spec->width;
            } else if (spec->mode == LC_RATE_CUMULATIVE) {
                start = spec->origin;
            }
        }
        if (!lc_isfinite(start) || !(start < bindings[entry].t_end)) {
            goto invalid_schedule;
        }
        new_windows[entry].t_start = bindings[entry].t_start;
        new_windows[entry].t_end = bindings[entry].t_end;
        new_states[entry].start = start;
        new_decoders[entry] = decoder;
        new_window_ids[entry] = bindings[entry].window;
        close_keys[entry].end = bindings[entry].t_end;
        close_keys[entry].entry = entry;
        identity_keys[entry].decoder = decoder;
        identity_keys[entry].window = bindings[entry].window;
        node = run->compiled->specs[decoder].node;
        new_node_offsets[node + 1U]++;
    }
    for (node_index = 1U; node_index <= run->compiled->node_count; ++node_index) {
        new_node_offsets[node_index] += new_node_offsets[node_index - 1U];
    }
    memcpy(
        cursors, new_node_offsets,
        run->compiled->node_count * sizeof(uint64_t)
    );
    for (entry = 0U; entry < binding_count; ++entry) {
        node = run->compiled->specs[new_decoders[entry]].node;
        new_node_entries[cursors[node]++] = entry;
    }
    qsort(
        close_keys, (size_t)binding_count, sizeof(lc_decoder_close_key),
        lc_compare_decoder_close_key
    );
    qsort(
        identity_keys, (size_t)binding_count,
        sizeof(lc_decoder_identity_key), lc_compare_decoder_identity_key
    );
    for (entry = 1U; entry < binding_count; ++entry) {
        if (identity_keys[entry].decoder == identity_keys[entry - 1U].decoder &&
            identity_keys[entry].window == identity_keys[entry - 1U].window) {
            goto invalid_schedule;
        }
    }
    for (entry = 0U; entry < binding_count; ++entry) {
        new_close_order[entry] = close_keys[entry].entry;
    }
    free(run->windows);
    free(run->states);
    free(run->window_closed);
    free(run->entry_decoders);
    free(run->window_ids);
    free(run->close_order);
    free(run->node_offsets);
    free(run->node_entries);
    free(run->queries);
    run->windows = new_windows;
    run->states = new_states;
    run->window_closed = new_closed;
    run->entry_decoders = new_decoders;
    run->window_ids = new_window_ids;
    run->entry_count = binding_count;
    run->close_order = new_close_order;
    run->close_position = 0U;
    run->queries = NULL;
    run->query_count = 0U;
    run->query_position = 0U;
    run->node_offsets = new_node_offsets;
    run->node_entries = new_node_entries;
    run->last_observation = LC_TIME_C(0.0);
    run->has_observation = 0;
    run->event_count = 0U;
    run->sealed = 0;
    run->ready = 1;
    free(close_keys);
    free(identity_keys);
    free(cursors);
    return LC_OK;

invalid_schedule:
    free(new_windows);
    free(new_states);
    free(new_closed);
    free(new_decoders);
    free(new_window_ids);
    free(new_close_order);
    free(new_node_offsets);
    free(new_node_entries);
    free(close_keys);
    free(identity_keys);
    free(cursors);
    return LC_INVALID_ARGUMENT;
}

/* Install exact-time queries after confirming each window assignment. */
lc_status lc_decoder_run_set_queries(
    lc_decoder_run *run,
    const lc_decoder_query_binding *bindings,
    uint64_t binding_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_decoder_query_item *queries = NULL;
    uint64_t index;
    if (run == NULL || run->compiled == NULL || !run->ready || run->sealed ||
        run->has_observation || run->event_count != 0U ||
        (binding_count > 0U && bindings == NULL) ||
        lc_codec_allocation_size_overflows(
            binding_count, sizeof(lc_decoder_query_item)
        )) {
        return LC_INVALID_ARGUMENT;
    }
    if (binding_count > 0U) {
        queries = (lc_decoder_query_item *)malloc(
            (size_t)binding_count * sizeof(lc_decoder_query_item)
        );
        if (queries == NULL) {
            return LC_ALLOCATION_FAILED;
        }
    }
    for (index = 0U; index < binding_count; ++index) {
        uint64_t entry;
        int found = 0;
        if (!lc_isfinite(bindings[index].t)) {
            free(queries);
            return LC_INVALID_ARGUMENT;
        }
        for (entry = 0U; entry < run->entry_count; ++entry) {
            const lc_decoder_spec *spec;
            if (run->entry_decoders[entry] != bindings[index].decoder ||
                run->window_ids[entry] != bindings[index].window) {
                continue;
            }
            spec = &run->compiled->specs[bindings[index].decoder];
            if (spec->emission != LC_EMIT_ON_QUERY ||
                bindings[index].t < run->windows[entry].t_start ||
                bindings[index].t > run->windows[entry].t_end ||
                (spec->kind == LC_DECODER_RATE &&
                 !(bindings[index].t > run->states[entry].start))) {
                free(queries);
                return LC_INVALID_ARGUMENT;
            }
            queries[index].t = bindings[index].t;
            queries[index].entry = entry;
            found = 1;
            break;
        }
        if (!found) {
            free(queries);
            return LC_INVALID_ARGUMENT;
        }
    }
    if (binding_count > 1U) {
        qsort(
            queries, (size_t)binding_count, sizeof(lc_decoder_query_item),
            lc_compare_decoder_query_item
        );
    }
    for (index = 1U; index < binding_count; ++index) {
        if (queries[index].entry == queries[index - 1U].entry &&
            queries[index].t == queries[index - 1U].t) {
            free(queries);
            return LC_INVALID_ARGUMENT;
        }
    }
    free(run->queries);
    run->queries = queries;
    run->query_count = binding_count;
    run->query_position = 0U;
    return LC_OK;
}

/* Compute the current value from one decoder accumulator. */
static lc_real_t lc_stream_decoder_value(
    const lc_decoder_spec *spec,
    const lc_stream_decoder_state *state,
    const lc_decode_window *window,
    lc_time_t observed_through
) {
    if (spec->kind == LC_DECODER_RATE) {
        lc_real_t interval;
        if (!lc_codec_real_interval(observed_through, state->start, &interval) ||
            interval <= LC_REAL_C(0.0)) return NAN;
        return (lc_real_t)state->count / interval;
    }
    if (spec->kind == LC_DECODER_TTFS) {
        lc_real_t value;
        if (!lc_codec_real_interval(state->first_spike, state->start, &value)) return NAN;
        if (spec->normalize) {
            lc_real_t interval;
            if (!lc_codec_real_interval(window->t_end, state->start, &interval) ||
                interval <= LC_REAL_C(0.0)) return NAN;
            value /= interval;
        }
        return value;
    }
    if (spec->normalize && state->count > 0U) {
        return state->weighted / (lc_real_t)state->count;
    }
    return state->weighted;
}

/* Check event retention capacity before mutating decoder state. */
static lc_status lc_decoder_require_event_slots(
    const lc_decoder_run *run,
    uint64_t required
) {
    if (run->event_capacity == 0U || required == 0U) {
        return LC_OK;
    }
    if (run->event_count > run->event_capacity ||
        required > run->event_capacity - run->event_count) {
        return LC_DECODER_OUTPUT_OVERFLOW;
    }
    return LC_OK;
}

/* Materialize one decoded event from the current window accumulator. */
static lc_status lc_decoder_append_event(
    lc_decoder_run *run,
    uint64_t entry,
    uint32_t kind,
    uint32_t valid,
    uint32_t has_source,
    lc_time_t emitted_at,
    lc_time_t source_spike_time,
    lc_time_t observed_through
) {
    const lc_stream_decoder_state *state;
    const lc_decode_window *decode_window;
    uint32_t decoder;
    lc_decoded_event *event;
    lc_real_t value;
    if (run->event_capacity == 0U) {
        return LC_OK;
    }
    state = &run->states[entry];
    decode_window = &run->windows[entry];
    decoder = run->entry_decoders[entry];
    value = valid
        ? lc_stream_decoder_value(&run->compiled->specs[decoder], state,
                                  decode_window, observed_through)
        : LC_REAL_C(0.0);
    if (!lc_isfinite(value)) return LC_NUMERIC_ERROR;
    event = &run->events[run->event_count++];
    memset(event, 0, sizeof(*event));
    event->decoder = decoder;
    event->window = run->window_ids[entry];
    event->kind = kind;
    event->valid = valid;
    event->has_source = has_source;
    event->has_first_spike = state->count > 0U;
    event->count = state->count;
    event->emitted_at = emitted_at;
    event->source_spike_time = has_source ? source_spike_time : LC_TIME_C(0.0);
    event->window_start = state->start;
    event->window_end = decode_window->t_end;
    event->observed_through = observed_through;
    event->value = value;
    event->first_spike = state->count > 0U ? state->first_spike : LC_TIME_C(0.0);
    return LC_OK;
}

/* Report whether a decoder publishes when its window closes. */
static int lc_decoder_emits_at_close(
    const lc_decoder_spec *spec,
    const lc_stream_decoder_state *state
) {
    if (spec->emission == LC_EMIT_ON_QUERY) {
        return 0;
    }
    return spec->kind == LC_DECODER_RATE ||
           spec->emission == LC_EMIT_ON_WINDOW_CLOSE ||
           (spec->kind == LC_DECODER_TTFS && state->count == 0U) ||
           (spec->kind == LC_DECODER_TEMPORAL_WEIGHT &&
            spec->emission == LC_EMIT_ON_EVENT_AND_WINDOW_CLOSE);
}

/* Emit due queries and close windows under the selected boundary rule. */
static lc_status lc_decoder_run_advance_internal(
    lc_decoder_run *run,
    lc_time_t boundary,
    int inclusive
) {
    uint64_t required = 0U;
    uint64_t position;
    if (run == NULL || run->compiled == NULL || !run->ready || run->sealed ||
        !lc_isfinite(boundary) ||
        (run->has_observation && boundary < run->last_observation)) {
        return LC_INVALID_ARGUMENT;
    }
    for (position = run->query_position; position < run->query_count; ++position) {
        if (run->queries[position].t > boundary ||
            (!inclusive && run->queries[position].t == boundary)) {
            break;
        }
        required++;
    }
    for (position = run->close_position; position < run->entry_count; ++position) {
        uint64_t entry = run->close_order[position];
        uint32_t decoder;
        if (run->windows[entry].t_end > boundary ||
            (!inclusive && run->windows[entry].t_end == boundary)) {
            break;
        }
        if (run->window_closed[entry]) {
            continue;
        }
        decoder = run->entry_decoders[entry];
        if (lc_decoder_emits_at_close(
                &run->compiled->specs[decoder], &run->states[entry]
            )) {
            required++;
        }
    }
    if (lc_decoder_require_event_slots(run, required) != LC_OK) {
        return LC_DECODER_OUTPUT_OVERFLOW;
    }
    for (;;) {
        int query_due = run->query_position < run->query_count &&
                        (run->queries[run->query_position].t < boundary ||
                         (inclusive &&
                          run->queries[run->query_position].t == boundary));
        int close_due = run->close_position < run->entry_count &&
                        (run->windows[
                             run->close_order[run->close_position]
                         ].t_end < boundary ||
                         (inclusive &&
                          run->windows[
                              run->close_order[run->close_position]
                          ].t_end == boundary));
        if (!query_due && !close_due) {
            break;
        }
        if (query_due &&
            (!close_due ||
             run->queries[run->query_position].t <=
                 run->windows[run->close_order[run->close_position]].t_end)) {
            const lc_decoder_query_item *query =
                &run->queries[run->query_position++];
            uint32_t decoder = run->entry_decoders[query->entry];
            const lc_decoder_spec *spec = &run->compiled->specs[decoder];
            const lc_stream_decoder_state *state = &run->states[query->entry];
            uint32_t valid = spec->kind != LC_DECODER_TTFS || state->count > 0U;
            lc_status status = lc_decoder_append_event(
                run, query->entry, LC_DECODE_QUERY, valid, 0U, query->t, LC_TIME_C(0.0),
                query->t
            );
            if (status != LC_OK) return status;
            continue;
        }
        {
            uint64_t entry = run->close_order[run->close_position++];
            uint32_t decoder = run->entry_decoders[entry];
            const lc_decoder_spec *spec = &run->compiled->specs[decoder];
            const lc_stream_decoder_state *state = &run->states[entry];
            if (lc_decoder_emits_at_close(spec, state)) {
                uint32_t valid = spec->kind != LC_DECODER_TTFS ||
                                 state->count > 0U;
                lc_status status = lc_decoder_append_event(
                    run, entry,
                    valid ? LC_DECODE_FINAL : LC_DECODE_NO_SPIKE, valid, 0U,
                    run->windows[entry].t_end, LC_TIME_C(0.0), run->windows[entry].t_end
                );
                if (status != LC_OK) return status;
            }
            run->window_closed[entry] = 1U;
        }
    }
    run->last_observation = boundary;
    run->has_observation = 1;
    return LC_OK;
}

/* Advance through observations at the supplied closed boundary. */
lc_status lc_decoder_run_advance(
    lc_decoder_run *run,
    lc_time_t observed_through
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_decoder_run_advance_internal(run, observed_through, 1);
}

/* Advance strictly before an open right boundary. */
lc_status lc_decoder_run_advance_before(
    lc_decoder_run *run,
    lc_time_t observed_before
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_decoder_run_advance_internal(run, observed_before, 0);
}

/* Apply one chronological spike to every decoder bound to its node. */
lc_status lc_decoder_run_consume(
    lc_decoder_run *run,
    const lc_output_spike *spike
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint64_t position;
    uint64_t required = 0U;
    lc_status status;
    if (run == NULL || run->compiled == NULL || !run->ready || run->sealed ||
        spike == NULL ||
        spike->node >= run->compiled->node_count || !lc_isfinite(spike->t)) {
        return LC_INVALID_ARGUMENT;
    }
    status = lc_decoder_run_advance(run, spike->t);
    if (status != LC_OK) {
        return status;
    }
    for (position = run->node_offsets[spike->node];
         position < run->node_offsets[spike->node + 1U]; ++position) {
        uint64_t entry = run->node_entries[position];
        uint32_t decoder = run->entry_decoders[entry];
        const lc_decoder_spec *spec = &run->compiled->specs[decoder];
        const lc_stream_decoder_state *state = &run->states[entry];
        if (run->window_closed[entry] ||
            !(spike->t >= state->start &&
              spike->t < run->windows[entry].t_end) ||
            ((spec->kind == LC_DECODER_TTFS ||
              (spec->kind == LC_DECODER_TEMPORAL_WEIGHT &&
               spec->first_only)) &&
             state->count > 0U)) {
            continue;
        }
        if ((spec->kind == LC_DECODER_TTFS &&
             spec->emission == LC_EMIT_ON_EVENT) ||
            (spec->kind == LC_DECODER_TEMPORAL_WEIGHT &&
             (spec->emission == LC_EMIT_ON_EVENT ||
              spec->emission == LC_EMIT_ON_EVENT_AND_WINDOW_CLOSE))) {
            required++;
        }
    }
    if (lc_decoder_require_event_slots(run, required) != LC_OK) {
        return LC_DECODER_OUTPUT_OVERFLOW;
    }
    for (position = run->node_offsets[spike->node];
         position < run->node_offsets[spike->node + 1U]; ++position) {
        uint64_t entry = run->node_entries[position];
        uint32_t decoder = run->entry_decoders[entry];
        const lc_decoder_spec *spec = &run->compiled->specs[decoder];
        lc_stream_decoder_state *state = &run->states[entry];
        lc_real_t contribution;
        if (run->window_closed[entry] ||
            !(spike->t >= state->start &&
              spike->t < run->windows[entry].t_end)) {
            continue;
        }
        if ((spec->kind == LC_DECODER_TTFS ||
             (spec->kind == LC_DECODER_TEMPORAL_WEIGHT &&
              spec->first_only)) &&
            state->count > 0U) {
            continue;
        }
        if (state->count == UINT64_MAX) {
            return LC_NUMERIC_ERROR;
        }
        if (state->count == 0U) {
            state->first_spike = spike->t;
        }
        state->count++;
        if (spec->kind == LC_DECODER_TEMPORAL_WEIGHT) {
            if (!lc_codec_temporal_contribution(
                    spike->t, state->start, spec->tau, &contribution))
                return LC_NUMERIC_ERROR;
            state->weighted += contribution;
            if (!lc_isfinite(state->weighted)) {
                return LC_NUMERIC_ERROR;
            }
        }
        if (spec->kind == LC_DECODER_TTFS &&
            spec->emission == LC_EMIT_ON_EVENT) {
            status = lc_decoder_append_event(
                run, entry, LC_DECODE_FINAL, 1U, 1U, spike->t, spike->t,
                spike->t
            );
            if (status != LC_OK) return status;
        } else if (spec->kind == LC_DECODER_TEMPORAL_WEIGHT &&
                   (spec->emission == LC_EMIT_ON_EVENT ||
                    spec->emission == LC_EMIT_ON_EVENT_AND_WINDOW_CLOSE)) {
            status = lc_decoder_append_event(
                run, entry, LC_DECODE_UPDATE, 1U, 1U, spike->t, spike->t,
                spike->t
            );
            if (status != LC_OK) return status;
        }
    }
    return LC_OK;
}

/* Confirm that every compiled decoder targets an available node. */
lc_status lc_decoder_run_validate_for_nodes(
    const lc_decoder_run *run,
    uint32_t node_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (run == NULL || run->compiled == NULL || !run->ready || run->sealed ||
        run->compiled->node_count != node_count) {
        return LC_INVALID_ARGUMENT;
    }
    return LC_OK;
}

/* Confirm that an active schedule fits the network and run horizon. */
lc_status lc_decoder_run_validate_for_execution(
    const lc_decoder_run *run,
    uint32_t node_count,
    lc_time_t t_end
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (lc_decoder_run_validate_for_nodes(run, node_count) != LC_OK ||
        !lc_isfinite(t_end) || run->entry_count == 0U ||
        run->windows[run->close_order[run->entry_count - 1U]].t_end > t_end) {
        return LC_INVALID_ARGUMENT;
    }
    return LC_OK;
}

/* Return the exact number of final window results required. */
lc_status lc_decoder_run_result_count(
    const lc_decoder_run *run,
    uint64_t *result_count
) {
    if (run == NULL || run->compiled == NULL || !run->ready ||
        result_count == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *result_count = run->entry_count;
    return LC_OK;
}

/* Close remaining windows and copy results in canonical schedule order. */
lc_status lc_decoder_run_finalize(
    lc_decoder_run *run,
    lc_decode_result *results,
    uint64_t result_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint64_t entry;
    uint64_t expected;
    lc_status status;
    if (run == NULL || run->compiled == NULL || !run->ready ||
        results == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    expected = run->entry_count;
    if (result_count != expected) {
        return LC_INVALID_ARGUMENT;
    }
    if (!run->sealed) {
        lc_time_t through =
            run->windows[run->close_order[run->entry_count - 1U]].t_end;
        if (run->has_observation && run->last_observation > through) {
            through = run->last_observation;
        }
        status = lc_decoder_run_advance(run, through);
        if (status != LC_OK) {
            return status;
        }
        run->sealed = 1;
    }
    for (entry = 0U; entry < run->entry_count; ++entry) {
        uint32_t decoder = run->entry_decoders[entry];
        const lc_decoder_spec *spec = &run->compiled->specs[decoder];
        const lc_stream_decoder_state *state = &run->states[entry];
        lc_decode_result *result = &results[entry];
        result->decoder = decoder;
        result->window = run->window_ids[entry];
        result->valid = spec->kind == LC_DECODER_TTFS
                            ? state->count > 0U
                            : 1U;
        result->count = state->count;
        result->value = LC_REAL_C(0.0);
        result->first_spike = state->count > 0U ? state->first_spike : LC_TIME_C(0.0);
        result->window_start = state->start;
        result->window_end = run->windows[entry].t_end;
        if (spec->kind != LC_DECODER_TTFS || state->count > 0U) {
            result->value = lc_stream_decoder_value(
                spec, state, &run->windows[entry], run->windows[entry].t_end
            );
            if (!lc_isfinite(result->value)) return LC_NUMERIC_ERROR;
        }
    }
    return LC_OK;
}

/* Copy retained streaming events without changing decoder state. */
lc_status lc_decoder_run_copy_events(
    const lc_decoder_run *run,
    lc_decoded_event *events,
    uint64_t event_capacity,
    uint64_t *event_count
) {
    if (run == NULL || run->compiled == NULL || !run->ready ||
        event_count == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *event_count = run->event_count;
    if (run->event_count > event_capacity) {
        return LC_DECODER_OUTPUT_OVERFLOW;
    }
    if (run->event_count > 0U) {
        if (events == NULL || run->events == NULL) {
            return LC_INVALID_ARGUMENT;
        }
        memcpy(
            events, run->events,
            (size_t)run->event_count * sizeof(lc_decoded_event)
        );
    }
    return LC_OK;
}

/* Report retained event count and configured capacity. */
lc_status lc_decoder_run_event_usage(
    const lc_decoder_run *run,
    uint64_t *event_count,
    uint64_t *event_capacity
) {
    if (run == NULL || run->compiled == NULL || event_count == NULL ||
        event_capacity == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *event_count = run->event_count;
    *event_capacity = run->event_capacity;
    return LC_OK;
}

void lc_decoder_run_destroy(lc_decoder_run *run) {
    lc_compiled_decoders *compiled;
    if (run == NULL) {
        return;
    }
    compiled = run->compiled;
    free(run->windows);
    free(run->states);
    free(run->window_closed);
    free(run->entry_decoders);
    free(run->window_ids);
    free(run->close_order);
    free(run->node_offsets);
    free(run->node_entries);
    free(run->queries);
    free(run->events);
    free(run);
    lc_decoder_bank_release(compiled);
}
