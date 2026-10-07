#include "lacuna.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

static lc_real_t first_target(uint64_t seed) {
    uint64_t bits = splitmix64(seed);
#if LACUNA_REAL_BITS == 16
    lc_real_t uniform = ((lc_real_t)(bits >> 54U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1p-10);
#elif LACUNA_REAL_BITS == 32
    lc_real_t uniform = ((lc_real_t)(bits >> 41U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1p-23);
#else
    lc_real_t uniform = ((lc_real_t)(bits >> 11U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1p-53);
#endif
    return -lc_real_log(uniform);
}

static lc_compiled_graph *make_affine_graph(
    lc_real_t decay, lc_real_t drive, lc_real_t reset, lc_real_t log_scale, lc_real_t gain,
    uint32_t iterations
) {
    lc_expr_node expressions[13] = {
        {LC_EXPR_VAR, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_VAR, 0U, 0U, 1U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 0U, 2U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_EXP, 3U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 1U, 4U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(100.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 0U, 8U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_PHI1, 3U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 9U, 10U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_ADD, 5U, 11U, 0U, LC_REAL_C(0.0)}
    };
    lc_mixed_node node = {0};
    lc_compiled_graph *graph = NULL;
    expressions[2].value = decay;
    expressions[6].value = reset;
    expressions[8].value = drive;
    node.dispatch = LC_ROOT_FIND;
    node.crossing_kind = LC_CROSSING_INTEGRATED_HAZARD;
    node.state_count = 1U;
    node.threshold = LC_REAL_C(100.0);
    node.program_nodes = expressions;
    node.program_node_count = 13U;
    node.normal_roots[0] = 12U;
    node.clamped_roots[0] = 1U;
    node.reset_roots[0] = 6U;
    node.arithmetic_kind = LC_NODE_ARITHMETIC_SCALAR_AFFINE;
    node.scalar_affine_decay = decay;
    node.scalar_affine_drive = drive;
    node.scalar_affine_reset = reset;
    node.hazard.kind = LC_HAZARD_EXPONENTIAL_VOLTAGE;
    node.hazard.log_scale = log_scale;
    node.hazard.voltage_gain = gain;
#if LACUNA_REAL_BITS == 16
    node.hazard.relative_tolerance = LC_REAL_C(0x1p-7);
    node.hazard.absolute_tolerance = LC_REAL_C(0x1p-24);
    node.hazard.time_tolerance = LC_TIME_C(0x1p-10);
#elif LACUNA_REAL_BITS == 32
    node.hazard.relative_tolerance = LC_REAL_C(64.0) * LC_REAL_EPSILON;
    node.hazard.absolute_tolerance = LC_REAL_C(32.0) * LC_REAL_EPSILON;
    node.hazard.time_tolerance = LC_TIME_C(1e-6);
#else
    node.hazard.relative_tolerance = LC_REAL_C(1e-9);
    node.hazard.absolute_tolerance = LC_REAL_C(1e-11);
    node.hazard.time_tolerance = LC_TIME_C(1e-10);
#endif
    node.hazard.maximum_quadrature_depth = 16U;
    node.hazard.maximum_root_iterations = iterations;
    assert(lc_mixed_graph_compile(&node, 1U, 1U, NULL, 0U, NULL, 0U, &graph) == LC_OK);
    return graph;
}

static lc_compiled_graph *make_graph(
    lc_real_t decay, lc_real_t reset, lc_real_t log_scale, lc_real_t gain,
    uint32_t iterations
) {
    return make_affine_graph(decay, LC_REAL_C(0.0), reset, log_scale, gain, iterations);
}

static lc_status run_graph(
    lc_compiled_graph *graph, lc_real_t initial, lc_time_t start, lc_time_t end,
    uint64_t seed, lc_output_spike *outputs, uint64_t *count
) {
    lc_mixed_run *run = NULL;
    lc_run_config config = {0};
    lc_run_stats stats;
    lc_network_error error;
    lc_status status;
    config.t_end = end;
    config.queue_capacity = 64U;
    config.output_capacity = 64U;
    config.same_time_cascade_limit = 16U;
    config.stochastic_seed = seed;
    assert(lc_mixed_run_create(graph, &initial, 1U, &start, 1U, &run) == LC_OK);
    status = lc_mixed_run_execute(run, NULL, 0U, NULL, 0U,
        &config, outputs, count, &stats, &error);
    lc_mixed_run_destroy(run);
    return status;
}

static lc_compiled_graph *image_copy(lc_compiled_graph *graph) {
    lc_compiled_graph *loaded = NULL;
    uint8_t *image;
    uint64_t size, written;
    assert(lc_compiled_graph_image_size(graph, &size) == LC_OK);
    image = malloc((size_t)size);
    assert(image != NULL);
    assert(lc_compiled_graph_serialize(graph, image, size, &written) == LC_OK);
    assert(written == size);
    assert(lc_compiled_graph_deserialize(image, size, &loaded) == LC_OK);
    free(image);
    return loaded;
}

static void compare_runs(
    lc_compiled_graph *graph, lc_real_t initial, lc_time_t start, lc_time_t end,
    uint64_t seed, lc_status expected, lc_output_spike *outputs, uint64_t *count
) {
    lc_compiled_graph *loaded = image_copy(graph);
    lc_output_spike replay[64];
    uint64_t replay_count = 0U, index;
    assert(run_graph(graph, initial, start, end, seed, outputs, count) == expected);
    assert(run_graph(loaded, initial, start, end, seed, replay, &replay_count) == expected);
    if (expected == LC_OK) {
        assert(*count == replay_count);
        for (index = 0U; index < *count; ++index) {
            assert(outputs[index].t == replay[index].t);
            assert(outputs[index].node == replay[index].node);
            assert(lc_isfinite(outputs[index].t));
            if (index > 0U) assert(outputs[index].t > outputs[index - 1U].t);
        }
    }
    lc_mixed_graph_destroy(loaded);
}

#if LACUNA_REAL_BITS != 16
/* Independent fixed midpoint quadrature of the full decaying-voltage trajectory. */
static lc_real_t reference_integral(lc_time_t end, lc_real_t initial, lc_real_t gain) {
    const uint32_t steps = 4096U;
    lc_real_t h = (lc_real_t)end / (lc_real_t)steps;
    lc_real_t sum = LC_REAL_C(0.0);
    uint32_t index;
    for (index = 0U; index < steps; ++index) {
        lc_real_t t = ((lc_real_t)index + LC_REAL_C(0.5)) * h;
        lc_real_t voltage = initial * lc_real_exp(-t);
        sum += lc_real_exp(gain * voltage);
    }
    return h * sum;
}

static void test_constant_and_varying_rate(void) {
    lc_compiled_graph *graph;
    lc_output_spike outputs[64];
    uint64_t count = 0U;
    lc_real_t target = first_target(0U);
    lc_real_t integral;
    lc_real_t error;
    graph = make_graph(LC_REAL_C(-1.0), LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(1.0), 128U);
    compare_runs(graph, LC_REAL_C(0.0), LC_TIME_C(0.0), LC_TIME_C(2.0),
        0U, LC_OK, outputs, &count);
    assert(count > 0U && outputs[0].t == (lc_time_t)target);
    lc_mixed_graph_destroy(graph);

    /* The tiny voltage is not constant: the hazard gain magnifies its decay. */
    graph = make_graph(LC_REAL_C(-1.0), LC_REAL_C(1e-6), LC_REAL_C(0.0), LC_REAL_C(1e6), 128U);
    compare_runs(graph, LC_REAL_C(1e-6), LC_TIME_C(0.0), LC_TIME_C(2.0),
        0U, LC_OK, outputs, &count);
    assert(count > 0U);
    integral = reference_integral(outputs[0].t, LC_REAL_C(1e-6), LC_REAL_C(1e6));
    error = lc_real_fabs(integral - target);
#if LACUNA_REAL_BITS == 32
    assert(error < LC_REAL_C(5e-4));
#else
    assert(error < LC_REAL_C(5e-8));
#endif
    lc_mixed_graph_destroy(graph);
}
#endif

#if LACUNA_REAL_BITS == 16
static void test_half_hazards(void) {
    lc_compiled_graph *graph;
    lc_output_spike outputs[64];
    uint64_t count = 0U;
    lc_real_t target = first_target(0U);
    lc_real_t sum = LC_REAL_C(0.0);
    lc_real_t step;
    uint32_t index;
    graph = make_graph(-LC_REAL_C(1.0), LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(1.0), 64U);
    compare_runs(graph, LC_REAL_C(0.0), LC_TIME_C(0.0), LC_TIME_C(2.0),
        0U, LC_OK, outputs, &count);
    assert(count > 0U && outputs[0].t == (lc_time_t)target);
    compare_runs(graph, LC_REAL_C(0.0), LC_TIME_C(2048.0), LC_TIME_C(2056.0),
        0U, LC_NUMERIC_ERROR, outputs, &count);
    lc_mixed_graph_destroy(graph);

    graph = make_graph(-LC_REAL_C(1.0), LC_REAL_C(0.25), LC_REAL_C(0.0), LC_REAL_C(1.0), 64U);
    compare_runs(graph, LC_REAL_C(0.25), LC_TIME_C(0.0), LC_TIME_C(2.0),
        0U, LC_OK, outputs, &count);
    assert(count > 0U && outputs[0].t < (lc_time_t)target);
    step = (lc_real_t)outputs[0].t / LC_REAL_C(64.0);
    for (index = 0U; index < 64U; ++index) {
        lc_real_t t = ((lc_real_t)index + LC_REAL_C(0.5)) * step;
        sum += lc_real_exp(LC_REAL_C(0.25) * lc_real_exp(-t));
    }
    /* Independent midpoint integration; local test bound, not a global promise. */
    assert(lc_real_fabs(step * sum - target) < LC_REAL_C(0x1p-8));
    lc_mixed_graph_destroy(graph);

    /* No approximate constant shortcut may bypass a finite search budget. */
    graph = make_graph(-LC_REAL_C(0x1p-14), LC_REAL_C(1.0), -LC_REAL_C(10.0), LC_REAL_C(1.0), 1U);
    compare_runs(graph, LC_REAL_C(1.0), LC_TIME_C(0.0), LC_TIME_C(1024.0),
        0U, LC_ROOT_NONCONVERGENCE, outputs, &count);
    lc_mixed_graph_destroy(graph);

    /* Finite rates can overflow the Simpson weighted sum in half precision;
       saturation must not convert that unresolved integral into success. */
    graph = make_graph(-LC_REAL_C(1.0), LC_REAL_C(0.25), LC_REAL_C(10.0), LC_REAL_C(1.0), 64U);
    compare_runs(graph, LC_REAL_C(0.25), LC_TIME_C(0.0), LC_TIME_C(0x1p-10),
        0U, LC_NUMERIC_ERROR, outputs, &count);
    lc_mixed_graph_destroy(graph);
}
#endif

#if LACUNA_REAL_BITS == 32
static void test_rounded_scalar_equilibrium_is_not_constant(void) {
    lc_scalar_lif_model model = {0};
    lc_scalar_state state = {0};
    lc_real_t equilibrium;
    lc_real_t gain = LC_REAL_C(1e6);
    lc_compiled_graph *graph;
    lc_output_spike outputs[64];
    uint64_t count = 0U;
    model.a = LC_REAL_C(-0.1);
    model.b = LC_REAL_C(1.3);
    model.threshold = LC_REAL_C(100.0);
    equilibrium = -model.b / model.a;
    state.value = equilibrium;
    assert(equilibrium == LC_REAL_C(12.999999046325684));
    assert(lc_scalar_advance(&model, &state, LC_TIME_C(0.1)) == LC_OK);
    assert(state.value == LC_REAL_C(13.0));
    assert(gain * (state.value - equilibrium) > LC_REAL_C(0.9));

    /* One inversion iteration cannot resolve the amplified, nonconstant
       native trajectory. A false constant shortcut incorrectly succeeds. */
    graph = make_affine_graph(model.a, model.b, LC_REAL_C(0.0),
        -(gain * equilibrium), gain, 1U);
    compare_runs(graph, equilibrium, LC_TIME_C(0.0), LC_TIME_C(4.0),
        0U, LC_ROOT_NONCONVERGENCE, outputs, &count);
    lc_mixed_graph_destroy(graph);
}

static void test_reduced_error_paths(void) {
    lc_compiled_graph *graph;
    lc_output_spike outputs[64];
    uint64_t count = 0U;
    lc_time_t start;
    /* A rounded one-unit probe is unchanged, but it is not an equilibrium. */
    graph = make_graph(LC_REAL_C(-1e-10), LC_REAL_C(1.0), LC_REAL_C(-10.0), LC_REAL_C(1.0), 1U);
    compare_runs(graph, LC_REAL_C(1.0), LC_TIME_C(0.0), LC_TIME_C(1024.0),
        0U, LC_ROOT_NONCONVERGENCE, outputs, &count);
    lc_mixed_graph_destroy(graph);
#if LACUNA_TIME_BITS == 32
    start = LC_TIME_C(0x1p24);
#else
    start = LC_TIME_C(0x1p53);
#endif
    graph = make_graph(LC_REAL_C(-1.0), LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(1.0), 128U);
    compare_runs(graph, LC_REAL_C(0.0), start, start + LC_TIME_C(8.0),
        0U, LC_NUMERIC_ERROR, outputs, &count);
    lc_mixed_graph_destroy(graph);
}
#endif

int main(void) {
#if LACUNA_REAL_BITS == 16
    test_half_hazards();
#else
    test_constant_and_varying_rate();
#endif
#if LACUNA_REAL_BITS == 32
    test_rounded_scalar_equilibrium_is_not_constant();
    test_reduced_error_paths();
#endif
    return 0;
}
