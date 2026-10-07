#include "lacuna.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

static int lc_model_valid(const lc_scalar_lif_model *model) {
    return model != NULL && lc_isfinite(model->a) && lc_isfinite(model->b) &&
           lc_isfinite(model->threshold) && lc_isfinite(model->reset) &&
           lc_isfinite(model->refractory) && model->a < LC_REAL_C(0.0) &&
           model->refractory >= LC_REAL_C(0.0) && model->polarity <= LC_MIXED;
}

uint32_t lc_abi_version(void) {
    return LC_ABI_VERSION;
}

uint32_t lc_numeric_property(uint32_t field) {
    switch (field) {
        case LC_NUMERIC_PROPERTY_VERSION:
            return LC_NUMERIC_METADATA_VERSION;
        case LC_NUMERIC_PROPERTY_PROFILE:
            if (sizeof(lc_real_t) * CHAR_BIT != LACUNA_REAL_BITS ||
                sizeof(lc_time_t) * CHAR_BIT != LACUNA_TIME_BITS ||
                FLT_RADIX != 2 || DBL_MANT_DIG != 53 ||
                DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021 ||
                FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
                FLT_MIN_EXP != -125) {
                return LC_NUMERIC_PROFILE_UNKNOWN;
            }
            return LACUNA_REAL_BITS == 16 ? LC_NUMERIC_PROFILE_BINARY16 :
                LACUNA_REAL_BITS == 64 ? LC_NUMERIC_PROFILE_BINARY64 :
                (LACUNA_TIME_BITS == 64 ? LC_NUMERIC_PROFILE_BINARY32_TIME64 :
                 LC_NUMERIC_PROFILE_BINARY32);
        case LC_NUMERIC_PROPERTY_REAL_BITS:
            return (uint32_t)(sizeof(lc_real_t) * CHAR_BIT);
        case LC_NUMERIC_PROPERTY_TIME_BITS:
            return (uint32_t)(sizeof(lc_time_t) * CHAR_BIT);
        case LC_NUMERIC_PROPERTY_REAL_MANT_DIG:
            return LC_REAL_MANT_DIG;
        case LC_NUMERIC_PROPERTY_TIME_MANT_DIG:
            return LC_TIME_MANT_DIG;
        case LC_NUMERIC_PROPERTY_REAL_MAX_EXP:
            return LC_REAL_MAX_EXP;
        case LC_NUMERIC_PROPERTY_TIME_MAX_EXP:
            return LC_TIME_MAX_EXP;
        case LC_NUMERIC_PROPERTY_WIDE_BITS:
            return (uint32_t)(sizeof(lc_wide_t) * CHAR_BIT);
        case LC_NUMERIC_PROPERTY_WIDE_MANT_DIG:
            return LC_WIDE_MANT_DIG;
        case LC_NUMERIC_PROPERTY_WIDE_MAX_EXP:
            return LC_WIDE_MAX_EXP;
        case LC_NUMERIC_PROPERTY_RADIX:
            return FLT_RADIX;
        case LC_NUMERIC_PROPERTY_ARITHMETIC_REVISION:
            return LC_NUMERIC_ARITHMETIC_REVISION;
        default:
            return 0U;
    }
}

lc_status lc_numeric_profile_check(
    uint32_t expected_profile,
    uint32_t expected_real_bits,
    uint32_t expected_time_bits,
    uint32_t expected_arithmetic_revision
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (expected_profile == LC_NUMERIC_PROFILE_UNKNOWN ||
        expected_profile != lc_numeric_property(LC_NUMERIC_PROPERTY_PROFILE) ||
        expected_real_bits != lc_numeric_property(LC_NUMERIC_PROPERTY_REAL_BITS) ||
        expected_time_bits != lc_numeric_property(LC_NUMERIC_PROPERTY_TIME_BITS) ||
        expected_arithmetic_revision !=
            lc_numeric_property(LC_NUMERIC_PROPERTY_ARITHMETIC_REVISION)) {
        return LC_INVALID_ARGUMENT;
    }
    return LC_OK;
}

uint64_t lc_sizeof_network_error(void) {
    return (uint64_t)sizeof(lc_network_error);
}

const char *lc_status_string(lc_status status) {
    switch (status) {
        case LC_OK:
            return "ok";
        case LC_NO_CROSSING:
            return "no crossing";
        case LC_INVALID_ARGUMENT:
            return "invalid argument";
        case LC_TIME_REVERSED:
            return "time precedes state timestamp";
        case LC_NUMERIC_ERROR:
            return "non-finite numeric result";
        case LC_UNSUPPORTED_MODEL:
            return "model is outside the scalar stable-LIF capability";
        case LC_QUEUE_OVERFLOW:
            return "event queue capacity exceeded";
        case LC_OUTPUT_OVERFLOW:
            return "output spike capacity exceeded";
        case LC_CASCADE_LIMIT:
            return "same-time cascade limit exceeded";
        case LC_ALLOCATION_FAILED:
            return "memory allocation failed";
        case LC_ROOT_NONCONVERGENCE:
            return "certified root finder did not converge";
        case LC_DECODER_OUTPUT_OVERFLOW:
            return "decoded event capacity exceeded";
        case LC_TRACE_OVERFLOW:
            return "trace record capacity exceeded";
        case LC_INSPECTION_OVERFLOW:
            return "state inspection capacity exceeded";
        case LC_STEP_LIMIT:
            return "numerical integration step limit exceeded";
        case LC_RHS_EVALUATION_LIMIT:
            return "numerical right-hand-side evaluation limit exceeded";
        case LC_IMAGE_INVALID:
            return "compiled graph image is invalid";
        case LC_IMAGE_INCOMPATIBLE:
            return "compiled graph image is incompatible";
        case LC_IMAGE_CHECKSUM_MISMATCH:
            return "compiled graph image checksum mismatch";
        default:
            return "unknown status";
    }
}

/* Advance a stable scalar LIF trajectory exactly to the requested time. */
lc_status lc_scalar_advance(
    const lc_scalar_lif_model *model,
    lc_scalar_state *state,
    lc_time_t t
) {
    lc_time_t delta;
    lc_real_t model_delta;
    lc_real_t e;
    lc_real_t response;
    lc_real_t next;

#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (!lc_model_valid(model) || state == NULL || !lc_isfinite(state->value) ||
        !lc_isfinite(state->t_last) || !lc_isfinite(t)) {
        return LC_INVALID_ARGUMENT;
    }
    if (t < state->t_last) {
        return LC_TIME_REVERSED;
    }
    delta = t - state->t_last;
    if (delta == LC_REAL_C(0.0)) {
        return LC_OK;
    }

    model_delta = (lc_real_t)delta;
    if (!lc_isfinite(model_delta) || model_delta <= LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    e = lc_real_exp(model->a * model_delta);
    /* expm1 avoids cancellation when the event gap is small. */
    response = (lc_real_expm1(model->a * model_delta) / model->a) * model->b;
    next = e * state->value + response;
    if (!lc_isfinite(next)) {
        return LC_NUMERIC_ERROR;
    }
    state->value = next;
    state->t_last = t;
    return LC_OK;
}

/* Predict the next rising threshold crossing of a scalar LIF trajectory. */
lc_status lc_scalar_predict(
    const lc_scalar_lif_model *model,
    const lc_scalar_state *state,
    lc_time_t *t_spike,
    lc_dispatch_form *dispatch
) {
    lc_real_t asymptote;
    lc_real_t ratio;
    lc_time_t delta;
    lc_time_t next_time;

#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (!lc_model_valid(model) || state == NULL || t_spike == NULL ||
        dispatch == NULL || !lc_isfinite(state->value) || !lc_isfinite(state->t_last)) {
        return LC_INVALID_ARGUMENT;
    }
    if (state->value >= model->threshold) {
        /* Discontinuous post-deposit firing belongs to the event handler. */
        return LC_INVALID_ARGUMENT;
    }

    asymptote = -model->b / model->a;
    if (!lc_isfinite(asymptote)) {
        return LC_NUMERIC_ERROR;
    }
    if (asymptote <= model->threshold) {
        *dispatch = LC_REACTIVE;
        return LC_NO_CROSSING;
    }

    ratio = (model->threshold - asymptote) / (state->value - asymptote);
    if (!(ratio > LC_REAL_C(0.0) && ratio < LC_REAL_C(1.0))) {
        return LC_NUMERIC_ERROR;
    }
    delta = lc_real_log(ratio) / model->a;
    next_time = state->t_last + delta;
    if (!lc_isfinite(delta) || delta <= LC_REAL_C(0.0) || !lc_isfinite(next_time) ||
        next_time <= state->t_last) {
        return LC_NUMERIC_ERROR;
    }
    *t_spike = next_time;
    *dispatch = LC_CLOSED_FORM;
    return LC_OK;
}
