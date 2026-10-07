#include "lacuna.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define CAPACITY 128U

static lc_encoder_spec encoder_spec(uint32_t kind) {
    lc_encoder_spec spec = {0};
    spec.kind = kind;
    spec.stream = 3U;
    spec.amplitude = LC_REAL_C(0.5);
    spec.rate_min = LC_REAL_C(1.0);
    spec.rate_max = LC_REAL_C(3.0);
    spec.latency_min = LC_TIME_C(0.25);
    spec.latency_max = LC_TIME_C(1.25);
    spec.duration = LC_TIME_C(2.0);
    spec.gain = LC_REAL_C(2.0);
    spec.offset = LC_REAL_C(-0.25);
    spec.baseline = LC_REAL_C(0.125);
    return spec;
}

static lc_status encode(
    const lc_encoder_spec *spec, lc_encoder_state *state,
    const lc_presentation *presentation,
    lc_encoded_spike *spikes, uint64_t *count,
    lc_encoded_drive *drives, uint64_t *drive_count
) {
    return lc_encode_presentations(spec, 1U, state, 1U, presentation, 1U,
        spikes, CAPACITY, count, drives, CAPACITY, drive_count);
}

static void compare_spikes(
    const lc_encoded_spike *left, const lc_encoded_spike *right, uint64_t count
) {
    uint64_t index;
    for (index = 0U; index < count; ++index) {
        assert(left[index].t == right[index].t);
        assert(left[index].encoder == right[index].encoder);
        assert(left[index].value == right[index].value);
    }
}

static void test_encoder(uint32_t kind) {
    lc_encoder_spec spec = encoder_spec(kind);
    lc_encoder_state state;
    lc_presentation presentation = {LC_TIME_C(0.0), LC_TIME_C(4.0), 0U, LC_REAL_C(0.5)};
    lc_encoded_spike spikes[CAPACITY];
    lc_encoded_spike replay[CAPACITY];
    lc_encoded_drive drives[CAPACITY];
    lc_encoded_drive replay_drives[CAPACITY];
    uint64_t count = 0U, replay_count = 0U, drive_count = 0U, replay_drive_count = 0U;
    uint64_t index;
    lc_encoder_run *run = NULL;
    assert(lc_encoder_state_reset(&state, 1U, 91U) == LC_OK);
    if (kind == LC_ENCODER_NATIVE_EVENT) {
        /* Native events already are primitives, not scalar presentations. */
        assert(encode(&spec, &state, &presentation, spikes, &count,
            drives, &drive_count) == LC_INVALID_ARGUMENT);
        assert(lc_encoder_run_create(&spec, 1U, 91U, LC_TIME_C(0.0), &run) == LC_OK);
        assert(lc_encoder_run_advance(run, &presentation, 1U, LC_TIME_C(4.0), 1U,
            replay, CAPACITY, &replay_count, replay_drives, CAPACITY,
            &replay_drive_count) == LC_INVALID_ARGUMENT);
        assert(lc_encoder_run_advance(run, NULL, 0U, LC_TIME_C(4.0), 1U,
            replay, CAPACITY, &replay_count, replay_drives, CAPACITY,
            &replay_drive_count) == LC_OK);
        assert(replay_count == 0U && replay_drive_count == 0U);
        lc_encoder_run_destroy(run);
        return;
    }
    assert(encode(&spec, &state, &presentation, spikes, &count, drives, &drive_count) == LC_OK);
    assert(lc_encoder_state_reset(&state, 1U, 91U) == LC_OK);
    assert(encode(&spec, &state, &presentation, replay, &replay_count,
        replay_drives, &replay_drive_count) == LC_OK);
    assert(replay_count == count && replay_drive_count == drive_count);
    compare_spikes(spikes, replay, count);
    for (index = 0U; index < count; ++index) {
        assert(lc_isfinite(spikes[index].t) && lc_isfinite(spikes[index].value));
        assert(spikes[index].t >= presentation.t_start && spikes[index].t < presentation.t_end);
        assert(spikes[index].value == spec.amplitude);
        if (index > 0U) assert(spikes[index].t > spikes[index - 1U].t);
    }
    assert(lc_encoder_run_create(&spec, 1U, 91U, LC_TIME_C(0.0), &run) == LC_OK);
    assert(lc_encoder_run_advance(run, &presentation, 1U, LC_TIME_C(4.0), 1U,
        replay, CAPACITY, &replay_count, replay_drives, CAPACITY,
        &replay_drive_count) == LC_OK);
    assert(replay_count == count && replay_drive_count == drive_count);
    compare_spikes(spikes, replay, count);
    for (index = 0U; index < drive_count; ++index) {
        assert(drives[index].t == replay_drives[index].t);
        assert(drives[index].value == replay_drives[index].value);
        assert(lc_isfinite(drives[index].value));
    }
    lc_encoder_run_destroy(run);
    if (kind == LC_ENCODER_REGULAR_RATE) {
        assert(count == 7U);
        for (index = 0U; index < count; ++index)
            assert(spikes[index].t == (lc_time_t)(index + 1U) / LC_TIME_C(2.0));
    } else if (kind == LC_ENCODER_POISSON_RATE) {
        assert(count > 0U);
        assert(lc_encoder_state_reset(&state, 1U, 92U) == LC_OK);
        assert(encode(&spec, &state, &presentation, replay, &replay_count,
            replay_drives, &replay_drive_count) == LC_OK);
        assert(replay_count > 0U && spikes[0].t != replay[0].t);
    } else if (kind == LC_ENCODER_TTFS) {
        assert(count == 1U && spikes[0].t == LC_TIME_C(0.75));
    } else if (kind == LC_ENCODER_BURST) {
        assert(count == 4U);
        for (index = 0U; index < count; ++index)
            assert(spikes[index].t == (lc_time_t)index / LC_TIME_C(2.0));
    } else if (kind == LC_ENCODER_LATENCY_BURST) {
        assert(count == 6U);
        for (index = 0U; index < count; ++index)
            assert(spikes[index].t == LC_TIME_C(0.75) + (lc_time_t)index / LC_TIME_C(3.0));
    } else {
        assert(count == 0U && drive_count == 2U);
        assert(drives[0].t == LC_TIME_C(0.0) && drives[0].value == LC_REAL_C(0.75));
        assert(drives[1].t == LC_TIME_C(4.0) && drives[1].value == spec.baseline);
    }
}

static void test_encoder_open_boundary(void) {
    lc_encoder_spec spec = encoder_spec(LC_ENCODER_REGULAR_RATE);
    lc_presentation presentation = {LC_TIME_C(0.0), LC_TIME_C(4.0), 0U, LC_REAL_C(0.5)};
    lc_encoded_spike spikes[CAPACITY];
    lc_encoded_drive drives[CAPACITY];
    uint64_t count, drive_count;
    lc_encoder_run *run = NULL;
    assert(lc_encoder_run_create(&spec, 1U, 91U, LC_TIME_C(0.0), &run) == LC_OK);
    assert(lc_encoder_run_advance(run, &presentation, 1U, LC_TIME_C(1.0), 0U,
        spikes, CAPACITY, &count, drives, CAPACITY, &drive_count) == LC_OK);
    assert(count == 1U && spikes[0].t == LC_TIME_C(0.5));
    assert(lc_encoder_run_advance(run, NULL, 0U, LC_TIME_C(1.0), 1U,
        spikes, CAPACITY, &count, drives, CAPACITY, &drive_count) == LC_OK);
    assert(count == 1U && spikes[0].t == LC_TIME_C(1.0));
    lc_encoder_run_destroy(run);
}

static lc_time_t coarse_clock(void) {
#if LACUNA_TIME_BITS == 16
    return LC_TIME_C(0x1p11);
#elif LACUNA_TIME_BITS == 32
    return LC_TIME_C(0x1p24);
#else
    return LC_TIME_C(0x1p53);
#endif
}

static void test_encoder_clock_progress(void) {
    lc_encoder_spec spec = encoder_spec(LC_ENCODER_REGULAR_RATE);
    lc_encoder_state state;
    lc_presentation presentation = {0};
    lc_encoded_spike spikes[CAPACITY];
    lc_encoded_drive drives[CAPACITY];
    uint64_t count, drive_count;
    lc_encoder_run *run = NULL;
    presentation.t_start = coarse_clock();
    presentation.t_end = presentation.t_start + LC_TIME_C(8.0);
    presentation.value = LC_REAL_C(1.0);
    spec.rate_min = spec.rate_max = LC_REAL_C(0.75);
    assert(lc_encoder_state_reset(&state, 1U, 0U) == LC_OK);
    assert(encode(&spec, &state, &presentation, spikes, &count,
        drives, &drive_count) == LC_NUMERIC_ERROR);
    assert(count == 1U); /* The first interval advances; the second would tie. */
    assert(lc_encoder_run_create(&spec, 1U, 0U, presentation.t_start, &run) == LC_OK);
    assert(lc_encoder_run_advance(run, &presentation, 1U, presentation.t_end, 1U,
        spikes, CAPACITY, &count, drives, CAPACITY, &drive_count) == LC_NUMERIC_ERROR);
    lc_encoder_run_destroy(run);
    spec.kind = LC_ENCODER_POISSON_RATE;
    spec.rate_min = spec.rate_max = LC_REAL_C(1.0);
    spec.stream = 0U;
    assert(lc_encoder_state_reset(&state, 1U, 0U) == LC_OK);
    state.poisson_ready = 1U;
    state.poisson_remaining = LC_REAL_C(1.5);
    assert(encode(&spec, &state, &presentation, spikes, &count,
        drives, &drive_count) == LC_NUMERIC_ERROR);
    assert(count == 1U);
    spec.kind = LC_ENCODER_TTFS;
    spec.latency_min = spec.latency_max = LC_TIME_C(0.25);
    assert(lc_encoder_state_reset(&state, 1U, 0U) == LC_OK);
    assert(encode(&spec, &state, &presentation, spikes, &count,
        drives, &drive_count) == LC_NUMERIC_ERROR);
    spec.kind = LC_ENCODER_BURST;
    spec.rate_min = spec.rate_max = LC_REAL_C(2.0);
    spec.duration = LC_TIME_C(8.0);
    assert(lc_encoder_state_reset(&state, 1U, 0U) == LC_OK);
    assert(encode(&spec, &state, &presentation, spikes, &count,
        drives, &drive_count) == LC_NUMERIC_ERROR);
    assert(count == 1U); /* Legitimate onset spike, then a collapsed interval. */
}

static void compare_results(const lc_decode_result *a, const lc_decode_result *b) {
    assert(a->valid == b->valid && a->count == b->count);
    assert(a->value == b->value && lc_isfinite(a->value));
    assert(a->first_spike == b->first_spike);
    assert(a->window_start == b->window_start && a->window_end == b->window_end);
}

static void test_decoders(void) {
    lc_decoder_spec specs[9] = {0};
    lc_decode_window window = {LC_TIME_C(0.0), LC_TIME_C(3.0)};
    lc_output_spike spikes[4] = {
        {LC_TIME_C(0.5), 0U}, {LC_TIME_C(1.25), 0U},
        {LC_TIME_C(2.25), 0U}, {LC_TIME_C(3.0), 0U}
    };
    lc_decode_result batch[9], streamed[9];
    lc_decoded_event events[32];
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *run = NULL;
    uint64_t event_count;
    uint32_t index;
    for (index = 0U; index < 9U; ++index) {
        specs[index].emission = LC_EMIT_ON_WINDOW_CLOSE;
        if (index < 3U) {
            specs[index].kind = LC_DECODER_RATE;
            specs[index].mode = index;
            specs[index].width = LC_TIME_C(2.0);
            specs[index].origin = LC_TIME_C(-1.0);
        } else if (index < 5U) {
            specs[index].kind = LC_DECODER_TTFS;
            specs[index].normalize = index - 3U;
        } else {
            specs[index].kind = LC_DECODER_TEMPORAL_WEIGHT;
            specs[index].tau = LC_REAL_C(0.75);
            specs[index].normalize = (index - 5U) % 2U;
            specs[index].first_only = (index - 5U) / 2U;
        }
    }
    assert(lc_decode_spikes(specs, 9U, spikes, 4U, &window, batch, 9U) == LC_OK);
    assert(batch[0].value == LC_REAL_C(1.0));
    assert(batch[1].value == LC_REAL_C(1.0) && batch[1].count == 2U);
    assert(batch[2].value == LC_REAL_C(0.75));
    assert(batch[3].value == LC_REAL_C(0.5));
    assert(batch[4].value == LC_REAL_C(0.5) / LC_REAL_C(3.0));
    assert(batch[7].value == lc_real_exp(-LC_REAL_C(0.5) / LC_REAL_C(0.75)));
    assert(batch[8].value == batch[7].value);
    assert(lc_decoder_bank_compile(specs, 9U, 1U, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &run) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 32U) == LC_OK);
    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    for (index = 0U; index < 4U; ++index)
        assert(lc_decoder_run_consume(run, &spikes[index]) == LC_OK);
    assert(lc_decoder_run_finalize(run, streamed, 9U) == LC_OK);
    for (index = 0U; index < 9U; ++index) compare_results(&batch[index], &streamed[index]);
    assert(lc_decoder_run_copy_events(run, events, 32U, &event_count) == LC_OK);
    assert(event_count == 9U);
    for (index = 0U; index < event_count; ++index)
        assert(lc_isfinite(events[index].value));
    lc_decoder_run_destroy(run);
    lc_decoder_bank_destroy(bank);
}

static void test_decoder_numeric_error(void) {
    lc_decoder_spec spec = {0};
    lc_decode_window window = {0};
    lc_output_spike spike = {LC_TIME_C(0.0), 0U};
    lc_decode_result result;
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *run = NULL;
    spec.kind = LC_DECODER_RATE;
    spec.emission = LC_EMIT_ON_WINDOW_CLOSE;
    window.t_end = lc_time_nextafter(LC_TIME_C(0.0), LC_TIME_C(1.0));
    assert(lc_decode_spikes(&spec, 1U, &spike, 1U, &window, &result, 1U) == LC_NUMERIC_ERROR);
    assert(lc_decoder_bank_compile(&spec, 1U, 1U, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &run) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 2U) == LC_OK);
    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_consume(run, &spike) == LC_OK);
    assert(lc_decoder_run_finalize(run, &result, 1U) == LC_NUMERIC_ERROR);
    lc_decoder_run_destroy(run);
    lc_decoder_bank_destroy(bank);
}

static void test_decoder_real_arithmetic(void) {
    lc_decoder_spec spec = {0};
    lc_decode_window window = {LC_TIME_C(0.0), LC_TIME_C(3.0000001)};
    lc_output_spike spikes[2] = {{LC_TIME_C(1.0000001), 0U}, {LC_TIME_C(2.0), 0U}};
    lc_decode_result result;
    spec.kind = LC_DECODER_RATE;
    spec.emission = LC_EMIT_ON_WINDOW_CLOSE;
    assert(lc_decode_spikes(&spec, 1U, spikes, 2U, &window, &result, 1U) == LC_OK);
    assert(result.value == LC_REAL_C(2.0) / (lc_real_t)window.t_end);
    spec.kind = LC_DECODER_TTFS;
    spec.normalize = 1U;
    assert(lc_decode_spikes(&spec, 1U, spikes, 2U, &window, &result, 1U) == LC_OK);
    assert(result.value == (lc_real_t)spikes[0].t / (lc_real_t)window.t_end);
    spec.kind = LC_DECODER_TEMPORAL_WEIGHT;
    spec.tau = LC_REAL_C(0.7);
    spec.normalize = 0U;
    spec.first_only = 1U;
    assert(lc_decode_spikes(&spec, 1U, spikes, 2U, &window, &result, 1U) == LC_OK);
    assert(result.value == lc_real_exp(-(lc_real_t)spikes[0].t / spec.tau));
}

int main(void) {
    uint32_t kind;
    for (kind = LC_ENCODER_NATIVE_EVENT; kind <= LC_ENCODER_HELD_CURRENT; ++kind)
        test_encoder(kind);
    test_encoder_open_boundary();
    test_encoder_clock_progress();
    test_decoders();
    test_decoder_numeric_error();
    test_decoder_real_arithmetic();
    return 0;
}
