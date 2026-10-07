#include "lacuna.h"
#include "../../c/src/network_internal.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if LACUNA_REAL_BITS == 16
#include <fenv.h>
#endif

/* This test intentionally uses no double expressions on strict-float32 builds. */
static const lc_expr_node program[] = {
    {LC_EXPR_VAR, 0, 0, 0, LC_REAL_C(0.0)},
    {LC_EXPR_VAR, 0, 0, 1, LC_REAL_C(0.0)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(-0.1)},
    {LC_EXPR_MUL, 0, 2, 0, LC_REAL_C(0.0)},
    {LC_EXPR_EXP, 3, 0, 0, LC_REAL_C(0.0)},
    {LC_EXPR_MUL, 1, 4, 0, LC_REAL_C(0.0)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(-6.5)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(65.0)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(1.0)},
    {LC_EXPR_SUB, 4, 8, 0, LC_REAL_C(0.0)},
    {LC_EXPR_MUL, 7, 9, 0, LC_REAL_C(0.0)},
    {LC_EXPR_ADD, 5, 10, 0, LC_REAL_C(0.0)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(-65.0)},
    {LC_EXPR_CONST, 0, 0, 0, LC_REAL_C(-50.0)}
};

static uint32_t read_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) |
        ((uint32_t)bytes[2] << 16U) | ((uint32_t)bytes[3] << 24U);
}

static void write_u32(uint8_t *bytes, uint32_t value) {
    uint32_t index;
    for (index = 0U; index < 4U; ++index)
        bytes[index] = (uint8_t)(value >> (index * 8U));
}

static void write_u64(uint8_t *bytes, uint64_t value) {
    uint32_t index;
    for (index = 0U; index < 8U; ++index)
        bytes[index] = (uint8_t)(value >> (index * 8U));
}

static uint64_t header_size(void) {
    return LACUNA_REAL_BITS == 64 ? 40U : 56U;
}

static void update_checksum(uint8_t *image, uint64_t size) {
    uint32_t crc = UINT32_MAX;
    uint64_t position;
    for (position = header_size(); position < size; ++position) {
        uint32_t bit;
        crc ^= image[position];
        for (bit = 0U; bit < 8U; ++bit)
            crc = (crc >> 1U) ^
                (UINT32_C(0xedb88320) & (uint32_t)-(int32_t)(crc & 1U));
    }
    write_u32(&image[32], ~crc);
}

static uint8_t *serialize(const lc_compiled_graph *graph, uint64_t *size) {
    uint64_t written = 0U;
    uint8_t *image;
    assert(lc_compiled_graph_image_size(graph, size) == LC_OK);
    assert(*size > header_size() + 80U);
    image = malloc((size_t)*size);
    assert(image != NULL);
    assert(lc_compiled_graph_serialize(graph, image, *size - 1U, &written) ==
           LC_OUTPUT_OVERFLOW);
    assert(written == *size);
    assert(lc_compiled_graph_serialize(graph, image, *size, &written) == LC_OK);
    assert(written == *size);
    return image;
}

static lc_compiled_graph *make_graph(uint32_t plastic) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edges[2] = {0};
    lc_plasticity_rule rules[2] = {0};
    lc_real_t parameters[6] = {
        LC_REAL_C(-0.0), LC_REAL_C(1.25), LC_REAL_C(0.0)
    };
    lc_compiled_graph *graph = NULL;
    uint32_t index;
    parameters[1] = lc_real_nextafter(LC_REAL_C(1.25), LC_REAL_C(2.0));
    parameters[2] = lc_real_nextafter(LC_REAL_C(0.0), LC_REAL_C(1.0));
    memcpy(&parameters[3], parameters, 3U * sizeof(lc_real_t));
    for (index = 0U; index < 2U; ++index) {
        lc_mixed_node *node = &nodes[index];
        node->dispatch = LC_REACTIVE;
        node->crossing_kind = LC_CROSSING_SCALAR_LOG;
        node->state_offset = index;
        node->state_count = 1U;
        node->parameter_count = 3U;
        node->parameter_offset = index * 3U;
        node->threshold = LC_REAL_C(-50.0);
        node->refractory = lc_time_nextafter(LC_TIME_C(0.25), LC_TIME_C(1.0));
        node->program_nodes = program;
        node->program_node_count = (uint32_t)(sizeof(program) / sizeof(program[0]));
        node->normal_roots[0] = 11U;
        node->clamped_roots[0] = 1U;
        node->reset_roots[0] = 12U;
        node->scalar_log_hint = (lc_scalar_log_hint){2U, 6U, 13U};
        edges[index].pre = 0U;
        edges[index].post = 1U;
        edges[index].weight = LC_REAL_C(0.5);
        edges[index].delay = lc_time_nextafter(LC_TIME_C(0.25), LC_TIME_C(1.0));
        edges[index].deposit_scale = LC_REAL_C(1.0);
        rules[index].kind = LC_PLASTICITY_PAIR;
        rules[index].weight_group = 1U;
        rules[index].tau_pre = LC_REAL_C(5.0);
        rules[index].tau_post = LC_REAL_C(7.0);
        rules[index].a2_plus = LC_REAL_C(0.02);
        rules[index].a2_minus = LC_REAL_C(0.01);
        rules[index].learning_rate = LC_REAL_C(0.125);
        rules[index].weight_max = LC_REAL_C(1.0);
    }
    if (plastic == 2U) {
        lc_expr_node learning_nodes[3] = {
            {LC_EXPR_VAR, 0U, 0U, 0U, LC_REAL_C(0.0)},
            {LC_EXPR_PARAM, 0U, 0U, 0U, LC_REAL_C(0.0)},
            {LC_EXPR_ADD, 0U, 1U, 0U, LC_REAL_C(0.0)}
        };
        lc_learning_program learning = {0};
        lc_learning_binding bindings[2] = {0};
        lc_real_t learning_parameters[2];
        learning.parameter_count = 1U;
        learning.variable_count = 4U;
        learning.clamp_normalized_weight = 1U;
        for (index = 0U; index < LC_LEARNING_EVENT_COUNT; ++index)
            learning.events[index].weight_root = UINT32_MAX;
        learning.events[LC_LEARNING_PRE_SPIKE].nodes = learning_nodes;
        learning.events[LC_LEARNING_PRE_SPIKE].node_count = 3U;
        learning.events[LC_LEARNING_PRE_SPIKE].variable_mask = 1U;
        learning.events[LC_LEARNING_PRE_SPIKE].weight_root = 2U;
        learning.events[LC_LEARNING_MODULATION_POSITIVE] =
            learning.events[LC_LEARNING_PRE_SPIKE];
        for (index = 0U; index < 2U; ++index) {
            bindings[index].parameter_offset = index;
            bindings[index].weight_group = 1U;
            bindings[index].weight_max = LC_REAL_C(1.0);
            learning_parameters[index] =
                lc_real_nextafter(LC_REAL_C(0.025), LC_REAL_C(1.0));
        }
        assert(lc_mixed_graph_compile_learning(
            nodes, 2U, 2U, parameters, 6U, edges, &learning, 1U,
            learning_parameters, 2U, bindings, 2U, &graph
        ) == LC_OK);
    } else if (plastic) {
        assert(lc_mixed_graph_compile_plastic(
            nodes, 2U, 2U, parameters, 6U, edges, rules, 2U, &graph
        ) == LC_OK);
    } else {
        assert(lc_mixed_graph_compile(
            nodes, 2U, 2U, parameters, 6U, edges, 2U, &graph
        ) == LC_OK);
    }
    return graph;
}

static void compare_execution(lc_compiled_graph *original, lc_compiled_graph *loaded) {
    lc_compiled_graph *graphs[2] = {original, loaded};
    lc_mixed_input_spike inputs[2] = {
        {LC_TIME_C(1.0), 0U, LC_DEPOSIT_STATE_ADD, 0U, LC_REAL_C(20.0)},
        {LC_TIME_C(2.0), 1U, LC_DEPOSIT_STATE_ADD, 0U, LC_REAL_C(20.0)}
    };
    lc_real_t initial[2] = {LC_REAL_C(-65.0), LC_REAL_C(-65.0)};
    lc_time_t initial_times[2] = {LC_TIME_C(0.0), LC_TIME_C(0.0)};
    lc_real_t states[2][2];
    lc_real_t weights[2][2];
    lc_time_t times[2][2];
    lc_output_spike outputs[2][8];
    lc_run_config config = {LC_TIME_C(3.0), 32U, 8U, 8U, 0U};
    uint64_t counts[2] = {0U, 0U};
    uint32_t index;
    uint64_t spike;
    for (index = 0U; index < 2U; ++index) {
        lc_mixed_run *run = NULL;
        lc_run_stats stats;
        lc_network_error error;
        assert(lc_mixed_run_create(graphs[index], initial, 2U, initial_times, 2U, &run) == LC_OK);
        assert(lc_mixed_run_execute(run, inputs, 2U, NULL, 0U, &config,
            outputs[index], &counts[index], &stats, &error) == LC_OK);
        assert(lc_mixed_run_copy_state(run, states[index], 2U, times[index], 2U) == LC_OK);
        assert(lc_mixed_run_copy_weights(run, weights[index], 2U) == LC_OK);
        lc_mixed_run_destroy(run);
    }
    assert(counts[0] == counts[1] && counts[0] == 2U);
    for (spike = 0U; spike < counts[0]; ++spike) {
        assert(outputs[0][spike].t == outputs[1][spike].t);
        assert(outputs[0][spike].node == outputs[1][spike].node);
    }
    assert(memcmp(states[0], states[1], sizeof(states[0])) == 0);
    assert(memcmp(times[0], times[1], sizeof(times[0])) == 0);
    assert(memcmp(weights[0], weights[1], sizeof(weights[0])) == 0);
    if (original->plastic_edge_count > 0U) {
        assert(weights[0][0] > LC_REAL_C(0.5));
        assert(weights[0][0] <= LC_REAL_C(1.0));
        assert(weights[0][0] == weights[0][1]);
    }
}

static void expect_failure(const uint8_t *image, uint64_t size, lc_status expected) {
    lc_compiled_graph *loaded = (lc_compiled_graph *)(uintptr_t)1U;
    lc_status actual = lc_compiled_graph_deserialize(image, size, &loaded);
    if (actual != expected)
        fprintf(stderr, "image failure: size=%llu expected=%u actual=%u\n",
            (unsigned long long)size, (unsigned)expected, (unsigned)actual);
    assert(actual == expected);
    assert(loaded == NULL);
}

static void test_malformed(const uint8_t *image, uint64_t size) {
    uint8_t *changed = malloc((size_t)size + 1U);
    uint64_t cut;
    uint32_t offset;
    assert(changed != NULL);
    for (cut = 0U; cut < size; ++cut)
        expect_failure(image, cut, LC_IMAGE_INVALID);
    memcpy(changed, image, (size_t)size);
    changed[size - 1U] ^= 1U;
    expect_failure(changed, size, LC_IMAGE_CHECKSUM_MISMATCH);
    for (offset = 8U; offset <= 16U; offset += 4U) {
        memcpy(changed, image, (size_t)size);
        write_u32(&changed[offset], UINT32_MAX);
        expect_failure(changed, size, LC_IMAGE_INCOMPATIBLE);
    }
    for (offset = 20U; offset <= 36U; offset += 16U) {
        memcpy(changed, image, (size_t)size);
        write_u32(&changed[offset], 1U);
        expect_failure(changed, size, LC_IMAGE_INVALID);
    }
    if (LACUNA_REAL_BITS < 64) {
        for (offset = 40U; offset <= 48U; offset += 4U) {
            memcpy(changed, image, (size_t)size);
            write_u32(&changed[offset], UINT32_MAX);
            expect_failure(changed, size, LC_IMAGE_INCOMPATIBLE);
        }
        memcpy(changed, image, (size_t)size);
        write_u32(&changed[52], 1U);
        expect_failure(changed, size, LC_IMAGE_INVALID);
    }
    /* CRC-correct hostile counts must fail before they can drive allocation. */
    for (offset = 8U; offset < 56U; offset += 4U) {
        memcpy(changed, image, (size_t)size);
        write_u32(&changed[header_size() + offset], UINT32_MAX);
        update_checksum(changed, size);
        expect_failure(changed, size, LC_IMAGE_INVALID);
    }
    for (offset = 56U; offset < 80U; offset += 8U) {
        memcpy(changed, image, (size_t)size);
        write_u64(&changed[header_size() + offset], UINT64_MAX);
        update_checksum(changed, size);
        expect_failure(changed, size, LC_IMAGE_INVALID);
    }
    memcpy(changed, image, (size_t)size);
    changed[size] = 0U;
    write_u64(&changed[24], size + 1U);
    update_checksum(changed, size + 1U);
    expect_failure(changed, size + 1U, LC_IMAGE_INVALID);
    free(changed);
}

static void test_round_trip(uint32_t plastic) {
    lc_compiled_graph *graph = make_graph(plastic);
    lc_compiled_graph *loaded = NULL;
    uint64_t size;
    uint64_t again_size;
    uint8_t *image = serialize(graph, &size);
    uint8_t *again;
    uint32_t index;
    assert(memcmp(image, "LCGIMG01", 8U) == 0);
    assert(read_u32(&image[8]) == (LACUNA_REAL_BITS == 64 ? 1U : 2U));
    assert(read_u32(&image[12]) == LC_ABI_VERSION);
    assert(read_u32(&image[16]) == lc_numeric_property(LC_NUMERIC_PROPERTY_PROFILE));
    if (LACUNA_REAL_BITS == 32) {
        assert(read_u32(&image[40]) == LACUNA_REAL_BITS);
        assert(read_u32(&image[44]) == LACUNA_TIME_BITS);
        assert(read_u32(&image[48]) == LC_NUMERIC_ARITHMETIC_REVISION);
    }
    assert(lc_compiled_graph_deserialize(image, size, &loaded) == LC_OK);
    assert(memcmp(graph->parameters, loaded->parameters, 6U * sizeof(lc_real_t)) == 0);
    if (plastic == 2U) {
        assert(graph->learning_parameter_count == 2U);
        assert(loaded->learning_parameter_count == 2U);
        assert(memcmp(graph->learning_parameters, loaded->learning_parameters,
            2U * sizeof(lc_real_t)) == 0);
    }
    if (plastic) {
        assert(memcmp(graph->weight_learning_scales, loaded->weight_learning_scales,
            2U * sizeof(lc_real_t)) == 0);
    }
    for (index = 0U; index < 2U; ++index) {
        assert(memcmp(&graph->nodes[index].refractory,
            &loaded->nodes[index].refractory, sizeof(lc_time_t)) == 0);
        assert(memcmp(&graph->edges[index].delay,
            &loaded->edges[index].delay, sizeof(lc_time_t)) == 0);
    }
    again = serialize(loaded, &again_size);
    assert(size == again_size);
    assert(memcmp(image, again, (size_t)size) == 0);
    test_malformed(image, size);
    free(again);
    free(image); /* Loaded storage must remain independent of the source image. */
    compare_execution(graph, loaded);
    lc_mixed_graph_destroy(loaded);
    lc_mixed_graph_destroy(graph);
}

#if LACUNA_REAL_BITS == 16 && defined(FE_UPWARD)
static void test_half_environment_rejected_before_mutation(void) {
    fenv_t original_environment;
    lc_compiled_graph *graph;
    lc_compiled_graph *rejected_graph = NULL;
    lc_mixed_run *run = NULL;
    lc_real_t initial[2] = {LC_REAL_C(-65.0), LC_REAL_C(-65.0)};
    lc_time_t times[2] = {LC_TIME_C(0.0), LC_TIME_C(0.0)};
    lc_run_config config = {LC_TIME_C(3.0), 32U, 8U, 8U, 0U};
    lc_output_spike outputs[8];
    lc_run_stats stats;
    lc_network_error error;
    lc_encoder_spec encoder = {0};
    lc_encoder_state encoder_state;
    lc_encoder_run *encoder_run = NULL;
    lc_presentation presentation = {LC_TIME_C(0.0), LC_TIME_C(2.0), 0U, LC_REAL_C(0.5)};
    lc_encoded_spike encoded[8];
    lc_encoded_drive drives[8];
    lc_decoder_spec decoder = {0};
    lc_decode_window window = {LC_TIME_C(0.0), LC_TIME_C(3.0)};
    lc_decode_result decoded;
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *decoder_run = NULL;
    lc_output_spike spike = {LC_TIME_C(1.0), 0U};
    uint64_t count = 0U, drive_count = 0U, size, rejected_size, written;
    uint8_t *image;
    assert(fegetenv(&original_environment) == 0);
    assert(lc_half_environment_valid());
    graph = make_graph(1U);
    image = serialize(graph, &size);
    assert(lc_mixed_run_create(graph, initial, 2U, times, 2U, &run) == LC_OK);
    encoder.kind = LC_ENCODER_REGULAR_RATE;
    encoder.rate_min = encoder.rate_max = LC_REAL_C(1.0);
    encoder.amplitude = LC_REAL_C(1.0);
    assert(lc_encoder_state_reset(&encoder_state, 1U, 0U) == LC_OK);
    assert(lc_encoder_run_create(&encoder, 1U, 0U, LC_TIME_C(0.0), &encoder_run) == LC_OK);
    decoder.kind = LC_DECODER_RATE;
    decoder.emission = LC_EMIT_ON_WINDOW_CLOSE;
    assert(lc_decoder_bank_compile(&decoder, 1U, 1U, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &decoder_run) == LC_OK);
    assert(lc_decoder_run_reset(decoder_run, &window) == LC_OK);

    assert(fesetround(FE_UPWARD) == 0);
    assert(!lc_half_environment_valid());
    assert(lc_mixed_graph_compile(graph->nodes, 2U, 2U, graph->parameters, 6U,
        graph->edges, 2U, &rejected_graph) == LC_NUMERIC_ERROR);
    assert(rejected_graph == NULL);
    assert(lc_mixed_run_execute(run, NULL, 0U, NULL, 0U, &config,
        outputs, &count, &stats, &error) == LC_NUMERIC_ERROR);
    assert(lc_compiled_graph_image_size(graph, &rejected_size) == LC_NUMERIC_ERROR);
    assert(rejected_size == 0U);
    assert(lc_compiled_graph_serialize(graph, image, size, &written) == LC_NUMERIC_ERROR);
    assert(written == 0U);
    assert(lc_compiled_graph_deserialize(image, size, &rejected_graph) == LC_NUMERIC_ERROR);
    assert(rejected_graph == NULL);
    assert(lc_encode_presentations(&encoder, 1U, &encoder_state, 1U,
        &presentation, 1U, encoded, 8U, &count, drives, 8U, &drive_count) == LC_NUMERIC_ERROR);
    assert(lc_encoder_run_advance(encoder_run, &presentation, 1U, LC_TIME_C(2.0), 1U,
        encoded, 8U, &count, drives, 8U, &drive_count) == LC_NUMERIC_ERROR);
    assert(lc_decode_spikes(&decoder, 1U, &spike, 1U, &window, &decoded, 1U) == LC_NUMERIC_ERROR);
    assert(lc_decoder_run_consume(decoder_run, &spike) == LC_NUMERIC_ERROR);
    assert(lc_decoder_run_finalize(decoder_run, &decoded, 1U) == LC_NUMERIC_ERROR);

    assert(fesetenv(&original_environment) == 0);
    assert(lc_half_environment_valid());
    assert(lc_mixed_run_execute(run, NULL, 0U, NULL, 0U, &config,
        outputs, &count, &stats, &error) == LC_OK);
    assert(count == 0U);
    assert(lc_decoder_run_consume(decoder_run, &spike) == LC_OK);
    assert(lc_decoder_run_finalize(decoder_run, &decoded, 1U) == LC_OK);
    assert(decoded.count == 1U);
    lc_decoder_run_destroy(decoder_run);
    lc_decoder_bank_destroy(bank);
    lc_encoder_run_destroy(encoder_run);
    lc_mixed_run_destroy(run);
    lc_mixed_graph_destroy(graph);
    free(image);
}
#endif

int main(void) {
    test_round_trip(0U);
    test_round_trip(1U);
    test_round_trip(2U);
#if LACUNA_REAL_BITS == 16 && defined(FE_UPWARD)
    test_half_environment_rejected_before_mutation();
#endif
    return 0;
}
