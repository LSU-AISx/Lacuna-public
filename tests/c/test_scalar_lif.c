#include "lacuna.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int close_enough(double actual, double expected, double tolerance) {
    return fabs(actual - expected) <= tolerance;
}

static lc_scalar_lif_model reactive_model(void) {
    lc_scalar_lif_model model = {-0.1, -6.5, -50.0, -65.0, 0.0, LC_EXCITATORY};
    return model;
}

static const lc_expr_node reactive_program[14] = {
    {LC_EXPR_VAR, 0, 0, 0, 0.0},
    {LC_EXPR_VAR, 0, 0, 1, 0.0},
    {LC_EXPR_CONST, 0, 0, 0, -0.1},
    {LC_EXPR_MUL, 0, 2, 0, 0.0},
    {LC_EXPR_EXP, 3, 0, 0, 0.0},
    {LC_EXPR_MUL, 1, 4, 0, 0.0},
    {LC_EXPR_CONST, 0, 0, 0, -6.5},
    {LC_EXPR_CONST, 0, 0, 0, 65.0},
    {LC_EXPR_CONST, 0, 0, 0, 1.0},
    {LC_EXPR_SUB, 4, 8, 0, 0.0},
    {LC_EXPR_MUL, 7, 9, 0, 0.0},
    {LC_EXPR_ADD, 5, 10, 0, 0.0},
    {LC_EXPR_CONST, 0, 0, 0, -65.0},
    {LC_EXPR_CONST, 0, 0, 0, -50.0}
};

static void init_reactive_descriptor(lc_mixed_node *node, uint32_t state_offset) {
    memset(node, 0, sizeof(*node));
    node->dispatch = LC_REACTIVE;
    node->crossing_kind = LC_CROSSING_SCALAR_LOG;
    node->state_offset = state_offset;
    node->state_count = 1;
    node->readout = 0;
    node->threshold = -50.0;
    node->program_nodes = reactive_program;
    node->program_node_count = 14;
    node->normal_roots[0] = 11;
    node->clamped_roots[0] = 1;
    node->reset_roots[0] = 12;
    node->scalar_log_hint = (lc_scalar_log_hint){2, 6, 13};
}

static void test_zero_delay_cascade(void) {
    lc_scalar_lif_model models[3] = {reactive_model(), reactive_model(), reactive_model()};
    lc_scalar_state states[3] = {{-65.0, 0.0}, {-65.0, 0.0}, {-65.0, 0.0}};
    lc_delta_edge edges[2] = {{0, 1, 20.0, 0.0}, {1, 2, 20.0, 0.0}};
    lc_input_spike inputs[1] = {{1.0, 0, 20.0}};
    lc_output_spike outputs[3];
    lc_run_config config = {2.0, 32, 3, 8, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;

    assert(lc_delta_network_run(
               models, states, 3, edges, 2, inputs, 1, NULL, 0, &config, outputs,
               &output_count, &stats
           ) == LC_OK);
    assert(output_count == 3);
    assert(outputs[0].t == 1.0 && outputs[0].node == 0);
    assert(outputs[1].t == 1.0 && outputs[1].node == 1);
    assert(outputs[2].t == 1.0 && outputs[2].node == 2);
    assert(stats.max_same_time_cascade_depth == 3);
}

static void test_dale_polarity_signs_nonnegative_edge_magnitudes(void) {
    lc_scalar_lif_model models[2] = {reactive_model(), reactive_model()};
    lc_scalar_state states[2] = {{-65.0, 0.0}, {-65.0, 0.0}};
    lc_delta_edge edge = {0, 1, 20.0, 0.0};
    lc_input_spike input = {1.0, 0, 20.0};
    lc_output_spike output[2];
    lc_run_config config = {2.0, 16, 2, 8, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;

    models[0].polarity = LC_INHIBITORY;
    assert(lc_delta_network_run(
               models, states, 2, &edge, 1, &input, 1, NULL, 0,
               &config, output, &output_count, &stats
           ) == LC_OK);
    assert(output_count == 1 && output[0].node == 0 && output[0].t == 1.0);
    assert(close_enough(states[1].value, -65.0 - 20.0 * exp(-0.1), 1e-12));

    states[0] = (lc_scalar_state){-65.0, 0.0};
    states[1] = (lc_scalar_state){-65.0, 0.0};
    edge.weight = -1.0;
    output_count = 0;
    assert(lc_delta_network_run(
               models, states, 2, &edge, 1, &input, 1, NULL, 0,
               &config, output, &output_count, &stats
           ) == LC_INVALID_ARGUMENT);
}

static void test_same_time_deposits_are_aggregated(void) {
    lc_scalar_lif_model models[1] = {reactive_model()};
    lc_scalar_state states[1] = {{-65.0, 0.0}};
    lc_input_spike inputs[2] = {{1.0, 0, 20.0}, {1.0, 0, -10.0}};
    lc_output_spike outputs[1];
    lc_run_config config = {2.0, 8, 1, 4, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;

    assert(lc_delta_network_run(
               models, states, 1, NULL, 0, inputs, 2, NULL, 0, &config, outputs,
               &output_count, &stats
           ) == LC_OK);
    assert(output_count == 0);
    assert(close_enough(states[0].value, -55.951625819640405, 1e-12));
}

static void test_scalar_mixed_polarity_signed_fanout(void) {
    lc_scalar_lif_model models[3] = {
        reactive_model(), reactive_model(), reactive_model()
    };
    lc_scalar_state states[3] = {
        {-65.0, 0.0}, {-65.0, 0.0}, {-65.0, 0.0}
    };
    lc_delta_edge edges[2] = {{0, 1, 20.0, 0.0}, {0, 2, -20.0, 0.0}};
    lc_input_spike input = {1.0, 0, 20.0};
    lc_output_spike outputs[3];
    lc_run_config config = {2.0, 32, 3, 8, 0};
    lc_run_stats stats;
    uint64_t count;
    uint32_t polarity;

    models[0].polarity = LC_MIXED;
    assert(lc_delta_network_run(
        models, states, 3, edges, 2, &input, 1, NULL, 0,
        &config, outputs, &count, &stats
    ) == LC_OK);
    assert(count == 2U && outputs[0].node == 0U && outputs[1].node == 1U);
    assert(close_enough(states[2].value, -65.0 - 20.0 * exp(-0.1), 1e-12));

    for (polarity = LC_EXCITATORY; polarity <= LC_MIXED + 1U; ++polarity) {
        uint32_t index;
        if (polarity == LC_MIXED) {
            continue;
        }
        models[0].polarity = polarity;
        for (index = 0U; index < 3U; ++index) {
            states[index] = (lc_scalar_state){-65.0, 0.0};
        }
        assert(lc_delta_network_run(
            models, states, 3, edges, 2, &input, 1, NULL, 0,
            &config, outputs, &count, &stats
        ) == LC_INVALID_ARGUMENT);
    }
    assert(lc_scalar_advance(&models[0], &states[0], 1.0) == LC_INVALID_ARGUMENT);
}

static void test_stale_prediction_is_discarded(void) {
    lc_scalar_lif_model models[1] = {
        {-0.1, -4.5, -50.0, -65.0, 0.0, LC_EXCITATORY}
    };
    lc_scalar_state states[1] = {{-65.0, 0.0}};
    lc_input_spike inputs[1] = {{5.0, 0, -5.0}};
    lc_output_spike outputs[1];
    lc_run_config config = {15.0, 8, 1, 4, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;

    assert(lc_delta_network_run(
               models, states, 1, NULL, 0, inputs, 1, NULL, 0, &config, outputs,
               &output_count, &stats
           ) == LC_OK);
    assert(output_count == 0);
    assert(stats.stale_predictions == 1);
}

static void test_fixed_refractory_clamp_releases_and_reschedules(void) {
    lc_scalar_lif_model models[1] = {
        {-0.1, -4.5, -50.0, -65.0, 2.0, LC_EXCITATORY}
    };
    lc_scalar_state states[1] = {{-65.0, 0.0}};
    lc_output_spike outputs[2];
    lc_run_config config = {30.0, 8, 2, 4, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;
    double period = 10.0 * log(4.0);

    assert(lc_delta_network_run(
               models, states, 1, NULL, 0, NULL, 0, NULL, 0, &config, outputs,
               &output_count, &stats
           ) == LC_OK);
    assert(output_count == 2);
    assert(close_enough(outputs[0].t, period, 1e-12));
    assert(close_enough(outputs[1].t, period * 2.0 + 2.0, 1e-12));
}

static void test_drive_update_uses_old_b_before_boundary(void) {
    lc_scalar_lif_model models[1] = {reactive_model()};
    lc_scalar_state states[1] = {{-65.0, 0.0}};
    lc_drive_update updates[1] = {{5.0, 0, -4.5}};
    lc_output_spike outputs[1];
    lc_run_config config = {20.0, 8, 1, 4, 0};
    lc_run_stats stats;
    uint64_t output_count = 0;

    assert(lc_delta_network_run(
               models, states, 1, NULL, 0, NULL, 0, updates, 1, &config, outputs,
               &output_count, &stats
           ) == LC_OK);
    assert(output_count == 1);
    assert(close_enough(outputs[0].t, 5.0 + 10.0 * log(4.0), 1e-12));
}

static void test_expression_dag_evaluator(void) {
    lc_expr_node nodes[4] = {
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_MUL, 0, 1, 0, 0.0},
        {LC_EXPR_EXP, 2, 0, 0, 0.0}
    };
    double variables[1] = {3.0};
    double workspace[4];
    assert(lc_expr_evaluate(nodes, 4, NULL, 0, variables, 1, workspace, 4) == LC_OK);
    assert(close_enough(workspace[3], exp(6.0), 1e-12));
    nodes[2].lhs = 3;
    assert(
        lc_expr_evaluate(nodes, 4, NULL, 0, variables, 1, workspace, 4) ==
        LC_INVALID_ARGUMENT
    );
}

static void test_selected_expression_evaluator_ignores_unreachable_nodes(void) {
    lc_expr_node nodes[5] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_MUL, 0, 1, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_DIV, 1, 3, 0, 0.0}
    };
    double variables[1] = {3.0};
    uint32_t roots[1] = {2U};
    double outputs[1];
    double workspace[5];
    uint8_t active[5];

    assert(
        lc_expr_evaluate(nodes, 5, NULL, 0, variables, 1, workspace, 5) ==
        LC_NUMERIC_ERROR
    );
    assert(
        lc_expr_evaluate_selected(
            nodes, 5, NULL, 0, variables, 1, roots, 1, outputs, workspace, 5,
            active, 5
        ) == LC_OK
    );
    assert(outputs[0] == 6.0);
    assert(active[0] == 1U && active[1] == 1U && active[2] == 1U);
    assert(active[3] == 0U && active[4] == 0U);
}

static void test_expression_state_advance_and_deposit(void) {
    lc_expr_node advance_nodes[7] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_VAR, 0, 0, 1, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -1.0},
        {LC_EXPR_MUL, 0, 2, 0, 0.0},
        {LC_EXPR_EXP, 3, 0, 0, 0.0},
        {LC_EXPR_MUL, 1, 4, 0, 0.0},
        {LC_EXPR_VAR, 0, 0, 2, 0.0}
    };
    uint32_t roots[2] = {5, 6};
    double state[2] = {4.0, 7.0};
    double variables[3];
    double workspace[7];
    lc_time_t t_last = 0.0;
    lc_expr_node deposit_nodes[1] = {{LC_EXPR_VAR, 0, 0, 0, 0.0}};
    double deposit_variable[1];
    double deposit_workspace[1];

    assert(
        lc_expr_state_advance(
            advance_nodes, 7, NULL, 0, roots, 2, state, &t_last, 2.0,
            variables, 3, workspace, 7
        ) == LC_OK
    );
    assert(close_enough(state[0], 4.0 * exp(-2.0), 1e-12));
    assert(state[1] == 7.0);
    assert(t_last == 2.0);
    assert(
        lc_expr_state_deposit(
            deposit_nodes, 1, NULL, 0, 0, 3.5, state, 2, 1,
            deposit_variable, 1, deposit_workspace, 1
        ) == LC_OK
    );
    assert(state[1] == 10.5);
    assert(
        lc_expr_state_advance(
            advance_nodes, 7, NULL, 0, roots, 2, state, &t_last, 1.0,
            variables, 3, workspace, 7
        ) == LC_TIME_REVERSED
    );
}

static void test_stable_phi_expression_operations(void) {
    lc_expr_node nodes[3] = {
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_PHI1, 0, 0, 0, 0.0},
        {LC_EXPR_PHI1_DERIV, 0, 0, 0, 0.0}
    };
    double workspace[3];
    assert(lc_expr_evaluate(nodes, 3, NULL, 0, NULL, 0, workspace, 3) == LC_OK);
    assert(workspace[1] == 1.0);
    assert(workspace[2] == 0.5);
}

static void test_certified_alpha_root_finder(void) {
    lc_expr_node nodes[14] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -1.0},
        {LC_EXPR_MUL, 0, 1, 0, 0.0},
        {LC_EXPR_EXP, 2, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_MUL, 3, 4, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 1.0},
        {LC_EXPR_ADD, 5, 1, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -2.0},
        {LC_EXPR_MUL, 3, 8, 0, 0.0},
        {LC_EXPR_EXP, 0, 0, 0, 0.0},
        {LC_EXPR_MUL, 10, 4, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_VAR, 0, 0, 1, 0.0}
    };
    lc_root_hint hint = {
        7, 9, 11, 11, 6, 8, 12, 12, 1, 8, 12, 1e-10, 0.5
    };
    double state[3] = {-1.0, 0.0, 0.0};
    double variables[4];
    double workspace[14];
    lc_root_result result;
    assert(
        lc_expr_alpha_predict(
            nodes, 14, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 14
        ) == LC_OK
    );
    assert(close_enough(result.t_spike, log(2.0), 5e-10));
    assert(result.bracket_high - result.bracket_low <= result.tolerance);
    assert(result.extrema_count == 0);
}

static void test_adaptive_state_map_and_two_exp_root_finder(void) {
    lc_expr_node reset_nodes[4] = {
        {LC_EXPR_CONST, 0, 0, 0, -65.0},
        {LC_EXPR_VAR, 0, 0, 2, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_ADD, 1, 2, 0, 0.0}
    };
    uint32_t reset_roots[2] = {0, 3};
    double state[2] = {-50.0, 1.25};
    double variables[3];
    double reset_workspace[4];
    lc_expr_node crossing_nodes[12] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.1},
        {LC_EXPR_MUL, 0, 1, 0, 0.0},
        {LC_EXPR_EXP, 2, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 25.0},
        {LC_EXPR_MUL, 3, 4, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -10.0},
        {LC_EXPR_ADD, 5, 6, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -2.5},
        {LC_EXPR_MUL, 3, 8, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.02}
    };
    lc_two_exp_hint hint = {7, 9, 6, 4, 10, 1, 11, 1e-10, 10.0};
    double crossing_workspace[12];
    lc_root_result result;

    assert(
        lc_expr_state_map(
            reset_nodes, 4, NULL, 0, reset_roots, 2, state,
            variables, 3, reset_workspace, 4
        ) == LC_OK
    );
    assert(state[0] == -65.0);
    assert(state[1] == 3.25);
    assert(
        lc_expr_two_exp_predict(
            crossing_nodes, 12, NULL, 0, &hint, state, 2, 0.0,
            &result, variables, 3, crossing_workspace, 12
        ) == LC_OK
    );
    assert(close_enough(result.t_spike, 10.0 * log(2.5), 5e-10));
    assert(result.bracket_high - result.bracket_low <= result.tolerance);
    assert(result.extrema_count == 0U);
}

static void test_bounded_multi_exp_root_finder(void) {
    lc_expr_node nodes[7] = {
        {LC_EXPR_CONST, 0, 0, 0, 15.0},
        {LC_EXPR_CONST, 0, 0, 0, -80.0},
        {LC_EXPR_CONST, 0, 0, 0, 60.0},
        {LC_EXPR_CONST, 0, 0, 0, 10.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.1},
        {LC_EXPR_CONST, 0, 0, 0, -0.2},
        {LC_EXPR_CONST, 0, 0, 0, -0.3}
    };
    lc_multi_exp_hint hint;
    double state[3] = {0.0, 0.0, 0.0};
    double variables[4];
    double workspace[7];
    lc_root_result result;
    double low = 0.0;
    double high = 2.0;
    uint32_t index;

    memset(&hint, 0, sizeof(hint));
    hint.limit_root = 0U;
    hint.coefficient_roots[0] = 1U;
    hint.coefficient_roots[1] = 2U;
    hint.coefficient_roots[2] = 3U;
    hint.rate_roots[0] = 4U;
    hint.rate_roots[1] = 5U;
    hint.rate_roots[2] = 6U;
    hint.mode_count = 3U;
    hint.iteration_cap = 512U;
    hint.relative_tolerance = 1e-10;
    hint.fastest_time_constant = 10.0 / 3.0;
    assert(
        lc_expr_multi_exp_predict(
            nodes, 7, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 7
        ) == LC_OK
    );
    for (index = 0U; index < 200U; ++index) {
        double middle = low + 0.5 * (high - low);
        double value = 15.0 - 80.0 * exp(-0.1 * middle) +
            60.0 * exp(-0.2 * middle) + 10.0 * exp(-0.3 * middle);
        if (value > 0.0) {
            low = middle;
        } else {
            high = middle;
        }
    }
    assert(close_enough(result.t_spike, low + 0.5 * (high - low), 1e-8));
    assert(result.extrema_count >= 1U);
    assert(result.bracket_low <= result.t_spike);
    assert(result.t_spike <= result.bracket_high);

    nodes[1].value = 8.0;
    assert(
        lc_expr_multi_exp_predict(
            nodes, 7, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 7
        ) == LC_NO_CROSSING
    );

    /* (exp(-0.1t) - 2 exp(-0.2t))^2 touches zero but never crosses. */
    nodes[0].value = 0.0;
    nodes[1].value = 1.0;
    nodes[2].value = -4.0;
    nodes[3].value = 4.0;
    nodes[4].value = -0.2;
    nodes[5].value = -0.3;
    nodes[6].value = -0.4;
    assert(
        lc_expr_multi_exp_predict(
            nodes, 7, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 7
        ) == LC_NO_CROSSING
    );
}

static void test_bounded_exp_polynomial_root_finder(void) {
    lc_expr_node nodes[5] = {
        {LC_EXPR_CONST, 0, 0, 0, 15.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.1},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.4}
    };
    lc_exp_poly_hint hint;
    double state[3] = {0.0, 0.0, 0.0};
    double variables[4];
    double workspace[5];
    lc_root_result result;

    memset(&hint, 0, sizeof(hint));
    hint.limit_root = 0U;
    hint.rate_roots[0] = 1U;
    hint.coefficient_roots[0] = 2U;
    hint.coefficient_roots[1] = 3U;
    hint.coefficient_roots[2] = 4U;
    hint.coefficient_offsets[0] = 0U;
    hint.coefficient_offsets[1] = 3U;
    hint.block_count = 1U;
    hint.coefficient_count = 3U;
    hint.iteration_cap = 1024U;
    hint.relative_tolerance = 1e-10;
    hint.fastest_time_constant = 10.0;
    assert(
        lc_expr_exp_poly_predict(
            nodes, 5, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 5
        ) == LC_OK
    );
    assert(close_enough(result.t_spike, 10.195479618410925, 2e-9));
    assert(result.extrema_count == 1U);
    assert(result.bracket_high - result.bracket_low <= result.tolerance);

    /* The exact maximum touches threshold and is not a rising crossing. */
    nodes[4].value = -15.0 * exp(2.0) / 400.0;
    assert(
        lc_expr_exp_poly_predict(
            nodes, 5, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 5
        ) == LC_NO_CROSSING
    );

    /* Zero-limit tails are bounded by the eventual sign of the slow block. */
    nodes[0].value = 0.0;
    nodes[2].value = 1.0;
    nodes[3].value = -0.1;
    nodes[4].value = 0.0;
    assert(
        lc_expr_exp_poly_predict(
            nodes, 5, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 5
        ) == LC_OK
    );
    assert(close_enough(result.t_spike, 10.0, 2e-9));
    assert(result.horizon > result.t_spike);

    /* (1 - 0.1t)^2 exp(-0.1t) touches zero without crossing. */
    nodes[3].value = -0.2;
    nodes[4].value = 0.01;
    assert(
        lc_expr_exp_poly_predict(
            nodes, 5, NULL, 0, &hint, state, 3, 0.0, &result,
            variables, 4, workspace, 5
        ) == LC_NO_CROSSING
    );
}

static void test_mixed_runner_scalar_csr_path(void) {
    lc_mixed_node nodes[3] = {0};
    double state[3] = {-65.0, -65.0, -65.0};
    lc_time_t t_last[3] = {0.0, 0.0, 0.0};
    lc_mixed_edge edges[2] = {
        {0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0},
        {1, 2, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0}
    };
    lc_mixed_input_spike inputs[1] = {
        {1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0}
    };
    lc_output_spike outputs[3];
    lc_run_config config = {2.0, 32, 3, 8, 0};
    lc_run_stats stats;
    lc_network_error error;
    uint64_t output_count = 0;
    uint32_t index;

    for (index = 0; index < 3; ++index) {
        init_reactive_descriptor(&nodes[index], index);
    }

    assert(
        lc_mixed_network_run(
            nodes, 3, state, 3, t_last, NULL, 0, edges, 2, inputs, 1,
            NULL, 0, &config, outputs, &output_count, &stats, &error
        ) == LC_OK
    );
    assert(output_count == 3);
    assert(outputs[0].t == 1.0 && outputs[0].node == 0);
    assert(outputs[1].t == 1.0 && outputs[1].node == 1);
    assert(outputs[2].t == 1.0 && outputs[2].node == 2);
    assert(stats.deliveries_scheduled == 2);
    assert(stats.deliveries_processed == 2);
}

static void test_capability_descriptor_is_independent_of_model_identity(void) {
    lc_expr_node program[7] = {
        {LC_EXPR_VAR, 0, 0, 1, 0.0},
        {LC_EXPR_VAR, 0, 0, 2, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -0.1},
        {LC_EXPR_CONST, 0, 0, 0, -6.5},
        {LC_EXPR_CONST, 0, 0, 0, -50.0},
        {LC_EXPR_CONST, 0, 0, 0, -65.0},
        {LC_EXPR_VAR, 0, 0, 0, 0.0}
    };
    lc_expr_node deposit[3] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_MUL, 0, 1, 0, 0.0}
    };
    lc_mixed_node node = {0};
    double state[2] = {0.0, -65.0};
    lc_time_t t_last[1] = {0.0};
    lc_mixed_input_spike inputs[2] = {
        {1.0, 0, LC_DEPOSIT_PROGRAM, 0, 3.0},
        {1.0, 0, LC_DEPOSIT_STATE_ADD, 1, 20.0}
    };
    lc_output_spike output[1];
    lc_run_config config = {1.0, 8, 1, 4, 0};
    lc_run_stats stats;
    lc_network_error error;
    uint64_t output_count = 0U;

    node.dispatch = LC_REACTIVE;
    node.crossing_kind = LC_CROSSING_SCALAR_LOG;
    node.state_count = 2;
    node.readout = 1;
    node.threshold = -50.0;
    node.program_nodes = program;
    node.program_node_count = 7;
    node.normal_roots[0] = 0;
    node.normal_roots[1] = 1;
    node.clamped_roots[0] = 0;
    node.clamped_roots[1] = 1;
    node.reset_roots[0] = 0;
    node.reset_roots[1] = 5;
    node.scalar_log_hint = (lc_scalar_log_hint){2, 3, 4};
    node.deposit_nodes = deposit;
    node.deposit_node_count = 3;
    node.deposit_root = 2;
    node.deposit_target = 0;

    assert(
        lc_mixed_network_run(
            &node, 1, state, 2, t_last, NULL, 0, NULL, 0, inputs, 2,
            NULL, 0, &config, output, &output_count, &stats, &error
        ) == LC_OK
    );
    assert(output_count == 1U && output[0].node == 0U && output[0].t == 1.0);
    assert(state[0] == 6.0);
    assert(state[1] == -65.0);
}

static void test_compiled_graph_session_reset_and_lifetime(void) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edges[1] = {
        {0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0}
    };
    lc_mixed_input_spike inputs[1] = {
        {1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0}
    };
    double initial[2] = {-65.0, -65.0};
    double final_state[2];
    lc_time_t initial_times[2] = {0.0, 0.0};
    lc_time_t final_times[2];
    lc_output_spike outputs[2];
    lc_run_config config = {2.0, 16, 2, 8, 0};
    lc_run_stats first_stats;
    lc_run_stats second_stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    uint64_t first_count = 0;
    uint64_t second_count = 0;
    uint32_t index;

    for (index = 0; index < 2; ++index) {
        init_reactive_descriptor(&nodes[index], index);
    }
    assert(
        lc_mixed_graph_compile(nodes, 2, 2, NULL, 0, edges, 1, &compiled) == LC_OK
    );
    assert(compiled != NULL);
    assert(
        lc_mixed_run_create(compiled, initial, 2, initial_times, 2, &run) == LC_OK
    );
    lc_mixed_graph_destroy(compiled);
    assert(
        lc_mixed_run_execute(
            run, inputs, 1, NULL, 0, &config, outputs, &first_count,
            &first_stats, &error
        ) == LC_OK
    );
    assert(first_count == 2);
    assert(outputs[0].node == 0 && outputs[0].t == 1.0);
    assert(outputs[1].node == 1 && outputs[1].t == 1.0);
    assert(
        lc_mixed_run_copy_state(run, final_state, 2, final_times, 2) == LC_OK
    );
    assert(final_times[0] == 2.0 && final_times[1] == 2.0);
    assert(
        lc_mixed_run_reset(run, initial, 2, initial_times, 2) == LC_OK
    );
    assert(
        lc_mixed_run_execute(
            run, inputs, 1, NULL, 0, &config, outputs, &second_count,
            &second_stats, &error
        ) == LC_OK
    );
    assert(second_count == first_count);
    first_stats.kernel_seconds = 0.0;
    second_stats.kernel_seconds = 0.0;
    assert(memcmp(&first_stats, &second_stats, sizeof(lc_run_stats)) == 0);
    lc_mixed_run_destroy(run);
}

static void test_compiled_graph_image_round_trip(void) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edge = {
        0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.25, 1.0
    };
    lc_mixed_input_spike input = {
        1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0
    };
    double initial[2] = {-65.0, -65.0};
    double original_state[2];
    double loaded_state[2];
    lc_time_t initial_times[2] = {0.0, 0.0};
    lc_time_t original_times[2];
    lc_time_t loaded_times[2];
    lc_output_spike original_outputs[2];
    lc_output_spike loaded_outputs[2];
    lc_run_config config = {2.0, 16, 2, 8, 0};
    lc_run_stats original_stats;
    lc_run_stats loaded_stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_compiled_graph *loaded = NULL;
    lc_mixed_run *original_run = NULL;
    lc_mixed_run *loaded_run = NULL;
    uint8_t *first_image;
    uint8_t *second_image;
    uint64_t image_size = 0U;
    uint64_t written = 0U;
    uint64_t original_count = 0U;
    uint64_t loaded_count = 0U;
    uint32_t index;

    for (index = 0U; index < 2U; ++index) {
        init_reactive_descriptor(&nodes[index], index);
    }
    assert(
        lc_mixed_graph_compile(
            nodes, 2, 2, NULL, 0, &edge, 1, &compiled
        ) == LC_OK
    );
    assert(lc_compiled_graph_image_size(compiled, &image_size) == LC_OK);
    assert(image_size > 40U);
    first_image = malloc((size_t)image_size);
    second_image = malloc((size_t)image_size);
    assert(first_image != NULL && second_image != NULL);
    assert(
        lc_compiled_graph_serialize(
            compiled, first_image, image_size - 1U, &written
        ) == LC_OUTPUT_OVERFLOW
    );
    assert(written == image_size);
    assert(
        lc_compiled_graph_serialize(
            compiled, first_image, image_size, &written
        ) == LC_OK
    );
    assert(written == image_size);
    assert(
        lc_compiled_graph_serialize(
            compiled, second_image, image_size, &written
        ) == LC_OK
    );
    assert(memcmp(first_image, second_image, (size_t)image_size) == 0);
    second_image[image_size - 1U] ^= 1U;
    assert(
        lc_compiled_graph_deserialize(second_image, image_size, &loaded) ==
        LC_IMAGE_CHECKSUM_MISMATCH
    );
    assert(loaded == NULL);
    assert(
        lc_compiled_graph_deserialize(first_image, image_size, &loaded) ==
        LC_OK
    );
    free(first_image);
    free(second_image);

    assert(
        lc_mixed_run_create(
            compiled, initial, 2, initial_times, 2, &original_run
        ) == LC_OK
    );
    assert(
        lc_mixed_run_create(
            loaded, initial, 2, initial_times, 2, &loaded_run
        ) == LC_OK
    );
    lc_mixed_graph_destroy(compiled);
    lc_mixed_graph_destroy(loaded);
    assert(
        lc_mixed_run_execute(
            original_run, &input, 1, NULL, 0, &config, original_outputs,
            &original_count, &original_stats, &error
        ) == LC_OK
    );
    assert(
        lc_mixed_run_execute(
            loaded_run, &input, 1, NULL, 0, &config, loaded_outputs,
            &loaded_count, &loaded_stats, &error
        ) == LC_OK
    );
    assert(original_count == loaded_count);
    assert(memcmp(
        original_outputs, loaded_outputs,
        (size_t)original_count * sizeof(lc_output_spike)
    ) == 0);
    assert(
        lc_mixed_run_copy_state(
            original_run, original_state, 2, original_times, 2
        ) == LC_OK
    );
    assert(
        lc_mixed_run_copy_state(
            loaded_run, loaded_state, 2, loaded_times, 2
        ) == LC_OK
    );
    assert(memcmp(original_state, loaded_state, sizeof(original_state)) == 0);
    assert(memcmp(original_times, loaded_times, sizeof(original_times)) == 0);
    original_stats.kernel_seconds = 0.0;
    loaded_stats.kernel_seconds = 0.0;
    assert(memcmp(&original_stats, &loaded_stats, sizeof(original_stats)) == 0);
    lc_mixed_run_destroy(original_run);
    lc_mixed_run_destroy(loaded_run);
}

static void test_compiled_mixed_polarity_weights_and_image(void) {
    uint32_t arithmetic;
    for (arithmetic = LC_NODE_ARITHMETIC_EXPRESSIONS;
         arithmetic <= LC_NODE_ARITHMETIC_SCALAR_AFFINE; ++arithmetic) {
        lc_mixed_node nodes[3] = {0};
        lc_mixed_edge edges[2] = {
            {0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0},
            {0, 2, LC_DEPOSIT_STATE_ADD, 0, -20.0, 0.0, 1.0}
        };
        lc_mixed_input_spike input = {1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0};
        double initial[3] = {-65.0, -65.0, -65.0};
        double state[3];
        double copied[2];
        lc_time_t times[3] = {0.0, 0.0, 0.0};
        lc_output_spike outputs[3];
        lc_run_config config = {2.0, 32, 3, 8, 0};
        lc_run_stats stats;
        lc_network_error error;
        lc_compiled_graph *graph = NULL;
        lc_compiled_graph *loaded = NULL;
        lc_mixed_run *run = NULL;
        uint8_t *image;
        uint64_t size;
        uint64_t written;
        uint64_t count;
        uint32_t index;

        for (index = 0U; index < 3U; ++index) {
            init_reactive_descriptor(&nodes[index], index);
            nodes[index].arithmetic_kind = arithmetic;
            nodes[index].scalar_affine_decay = -0.1;
            nodes[index].scalar_affine_drive = -6.5;
            nodes[index].scalar_affine_reset = -65.0;
        }
        nodes[0].polarity = LC_MIXED;
        memcpy(state, initial, sizeof(state));
        assert(lc_mixed_network_run(
            nodes, 3, state, 3, times, NULL, 0, edges, 2, &input, 1,
            NULL, 0, &config, outputs, &count, &stats, &error
        ) == LC_OK);
        assert(count == 2U && outputs[1].node == 1U);
        assert(close_enough(state[2], -65.0 - 20.0 * exp(-0.1), 1e-12));
        memset(times, 0, sizeof(times));

        assert(lc_mixed_graph_compile(
            nodes, 3, 3, NULL, 0, edges, 2, &graph
        ) == LC_OK);
        assert(lc_compiled_graph_image_size(graph, &size) == LC_OK);
        image = malloc((size_t)size);
        assert(image != NULL);
        assert(lc_compiled_graph_serialize(graph, image, size, &written) == LC_OK);
        assert(written == size);
        assert(lc_compiled_graph_deserialize(image, size, &loaded) == LC_OK);
        free(image);
        lc_mixed_graph_destroy(graph);
        graph = NULL;
        assert(lc_mixed_run_create(loaded, initial, 3, times, 3, &run) == LC_OK);
        lc_mixed_graph_destroy(loaded);
        assert(lc_mixed_run_copy_weights(run, copied, 2) == LC_OK);
        assert(copied[0] == 20.0 && copied[1] == -20.0);
        assert(lc_mixed_run_execute(
            run, &input, 1, NULL, 0, &config, outputs, &count, &stats, &error
        ) == LC_OK);
        assert(count == 2U && outputs[1].node == 1U);
        assert(lc_mixed_run_copy_state(run, state, 3, times, 3) == LC_OK);
        assert(close_enough(state[2], -65.0 - 20.0 * exp(-0.1), 1e-12));
        lc_mixed_run_destroy(run);

        for (index = LC_EXCITATORY; index <= LC_MIXED + 1U; ++index) {
            if (index == LC_MIXED) {
                continue;
            }
            nodes[0].polarity = index;
            assert(lc_mixed_graph_compile(
                nodes, 3, 3, NULL, 0, edges, 2, &graph
            ) == LC_INVALID_ARGUMENT);
            assert(graph == NULL);
        }
    }
}

static void test_mixed_polarity_rejects_online_plasticity(void) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edge = {0, 1, LC_DEPOSIT_STATE_ADD, 0, 0.5, 1.0, 1.0};
    lc_plasticity_rule rule = {0};
    lc_compiled_graph *graph = NULL;
    uint32_t kind;

    init_reactive_descriptor(&nodes[0], 0U);
    init_reactive_descriptor(&nodes[1], 1U);
    rule.tau_pre = 5.0;
    rule.tau_post = 5.0;
    rule.tau_pre_slow = 10.0;
    rule.tau_post_slow = 10.0;
    rule.tau_eligibility_plus = 20.0;
    rule.tau_eligibility_minus = 20.0;
    rule.learning_rate = 0.01;
    rule.weight_max = 1.0;
    for (kind = LC_PLASTICITY_PAIR; kind <= LC_PLASTICITY_MODULATED; ++kind) {
        rule.kind = kind;
        nodes[0].polarity = LC_EXCITATORY;
        assert(lc_mixed_graph_compile_plastic(
            nodes, 2, 2, NULL, 0, &edge, &rule, 1, &graph
        ) == LC_OK);
        lc_mixed_graph_destroy(graph);
        graph = NULL;
        nodes[0].polarity = LC_MIXED;
        assert(lc_mixed_graph_compile_plastic(
            nodes, 2, 2, NULL, 0, &edge, &rule, 1, &graph
        ) == LC_INVALID_ARGUMENT);
        assert(graph == NULL);
    }
    rule.kind = LC_PLASTICITY_NONE;
    edge.weight = -0.5;
    assert(lc_mixed_graph_compile_plastic(
        nodes, 2, 2, NULL, 0, &edge, &rule, 1, &graph
    ) == LC_OK);
    lc_mixed_graph_destroy(graph);
}

static void test_incremental_compiled_run_keeps_boundary_open(void) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edge = {0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0};
    lc_mixed_input_spike input = {1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0};
    double initial[2] = {-65.0, -65.0};
    lc_time_t initial_times[2] = {0.0, 0.0};
    lc_output_spike outputs[2];
    lc_run_config config = {2.0, 16, 2, 8, 0};
    lc_run_stats stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    uint64_t output_count = 0U;
    uint32_t index;

    for (index = 0U; index < 2U; ++index) {
        init_reactive_descriptor(&nodes[index], index);
    }
    assert(
        lc_mixed_graph_compile(&nodes[0], 2, 2, NULL, 0, &edge, 1, &compiled) ==
        LC_OK
    );
    assert(
        lc_mixed_run_create(
            compiled, initial, 2, initial_times, 2, &run
        ) == LC_OK
    );
    assert(lc_mixed_run_begin_incremental(run, &config, &error) == LC_OK);
    assert(
        lc_mixed_run_advance_incremental(
            run, NULL, 0, NULL, 0, 1.0, 0, outputs, &output_count, &stats,
            &error, NULL, NULL, NULL
        ) == LC_OK
    );
    assert(output_count == 0U);
    assert(
        lc_mixed_run_advance_incremental(
            run, &input, 1, NULL, 0, 2.0, 0, outputs, &output_count, &stats,
            &error, NULL, NULL, NULL
        ) == LC_OK
    );
    assert(output_count == 2U);
    assert(outputs[0].t == 1.0 && outputs[0].node == 0U);
    assert(outputs[1].t == 1.0 && outputs[1].node == 1U);
    assert(
        lc_mixed_run_advance_incremental(
            run, NULL, 0, NULL, 0, 2.0, 1, outputs, &output_count, &stats,
            &error, NULL, NULL, NULL
        ) == LC_OK
    );
    assert(output_count == 0U);
    assert(
        lc_mixed_run_advance_incremental(
            run, NULL, 0, NULL, 0, 2.0, 1, outputs, &output_count, &stats,
            &error, NULL, NULL, NULL
        ) == LC_INVALID_ARGUMENT
    );
    lc_mixed_run_destroy(run);
    lc_mixed_graph_destroy(compiled);
}

typedef struct trace_collector {
    lc_trace_record records[8];
    uint64_t count;
} trace_collector;

static lc_status collect_trace_record(
    const lc_trace_record *record,
    void *context
) {
    trace_collector *collector = context;
    if (record == NULL || collector == NULL || collector->count >= 8U) {
        return LC_TRACE_OVERFLOW;
    }
    collector->records[collector->count++] = *record;
    return LC_OK;
}

static void test_configurable_causal_trace(void) {
    lc_mixed_node nodes[2] = {0};
    lc_mixed_edge edge = {0, 1, LC_DEPOSIT_STATE_ADD, 0, 20.0, 0.0, 1.0};
    lc_mixed_input_spike input = {1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0};
    double initial[2] = {-65.0, -65.0};
    double direct_state[2] = {-65.0, -65.0};
    lc_time_t initial_times[2] = {0.0, 0.0};
    lc_time_t direct_times[2] = {0.0, 0.0};
    lc_output_spike outputs[2];
    lc_run_config config = {2.0, 16, 2, 8, 0};
    lc_run_stats stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    lc_trace_record records[5];
    uint8_t node_mask[2] = {0, 1};
    uint64_t trace_count = 0U;
    uint64_t output_count = 0U;
    lc_trace_config trace = {0};
    trace_collector collector = {0};
    uint32_t index;

    for (index = 0; index < 2; ++index) {
        init_reactive_descriptor(&nodes[index], index);
    }
    trace.kind_mask =
        (UINT64_C(1) << LC_TRACE_INPUT_SPIKE) |
        (UINT64_C(1) << LC_TRACE_DELIVERY) |
        (UINT64_C(1) << LC_TRACE_DEPOSIT_APPLY) |
        (UINT64_C(1) << LC_TRACE_SPIKE) |
        (UINT64_C(1) << LC_TRACE_RESET) |
        (UINT64_C(1) << LC_TRACE_FINAL_STATE);
    trace.node_mask = node_mask;
    trace.node_mask_count = 2;
    trace.capture_state = 1;
    trace.records = records;
    trace.capacity = 5;
    trace.count = &trace_count;
    assert(
        lc_mixed_network_run_recorded(
            nodes, 2, direct_state, 2, direct_times, NULL, 0, &edge, 1,
            &input, 1, NULL, 0, &config, outputs, &output_count, &stats,
            &error, &trace
        ) == LC_OK
    );
    assert(output_count == 2U);
    assert(trace_count == 5U);
    for (index = 0; index < trace_count; ++index) {
        assert(records[index].sequence == index);
        assert(records[index].node == 1U);
    }
    assert(records[0].kind == LC_TRACE_DELIVERY && records[0].state_count == 0U);
    assert(records[1].kind == LC_TRACE_DEPOSIT_APPLY && records[1].state_count == 1U);
    assert(records[1].before[0] == -65.0 && records[1].after[0] == -45.0);
    assert(records[2].kind == LC_TRACE_SPIKE && records[2].before[0] == -45.0);
    assert(records[3].kind == LC_TRACE_RESET);
    assert(records[3].before[0] == -45.0 && records[3].after[0] == -65.0);
    assert(records[4].kind == LC_TRACE_FINAL_STATE && records[4].t == 2.0);

    assert(lc_mixed_graph_compile(nodes, 2, 2, NULL, 0, &edge, 1, &compiled) == LC_OK);
    assert(lc_mixed_run_create(compiled, initial, 2, initial_times, 2, &run) == LC_OK);
    lc_mixed_graph_destroy(compiled);
    trace.records = NULL;
    trace.capacity = 0U;
    trace.consumer = collect_trace_record;
    trace.consumer_context = &collector;
    output_count = 0U;
    assert(
        lc_mixed_run_execute_recorded(
            run, &input, 1, NULL, 0, &config, outputs, &output_count,
            &stats, &error, &trace
        ) == LC_OK
    );
    assert(trace_count == 5U && collector.count == 5U);
    assert(memcmp(records, collector.records, sizeof(records)) == 0);
    lc_mixed_run_destroy(run);
}

static void test_explicit_time_state_inspection_is_read_only(void) {
    lc_mixed_node node = {0};
    lc_mixed_input_spike input = {
        1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0
    };
    lc_state_inspection_request requests[4] = {
        {0.5, 0}, {1.0, 0}, {2.0, 0}, {3.0, 0}
    };
    lc_state_inspection_result results[4];
    lc_state_inspection_config inspections = {
        requests, 4, results, 3, NULL
    };
    double initial[1] = {-65.0};
    lc_time_t initial_time[1] = {0.0};
    lc_output_spike observed_output[1];
    lc_output_spike baseline_output[1];
    lc_run_config config = {3.0, 16, 1, 8, 0};
    lc_run_stats observed_stats;
    lc_run_stats baseline_stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    uint64_t inspection_count = 0U;
    uint64_t observed_count = 0U;
    uint64_t baseline_count = 0U;

    init_reactive_descriptor(&node, 0);
    node.refractory = 2.0;
    inspections.count = &inspection_count;
    assert(
        lc_mixed_graph_compile(&node, 1, 1, NULL, 0, NULL, 0, &compiled) == LC_OK
    );
    assert(
        lc_mixed_run_create(compiled, initial, 1, initial_time, 1, &run) == LC_OK
    );
    lc_mixed_graph_destroy(compiled);

    requests[1].t = 0.25;
    inspections.capacity = 4U;
    assert(
        lc_mixed_run_execute_observed(
            run, &input, 1, NULL, 0, &config, observed_output,
            &observed_count, &observed_stats, &error, NULL, NULL,
            &inspections
        ) == LC_INVALID_ARGUMENT
    );
    requests[1].t = 1.0;
    inspections.capacity = 3U;
    assert(
        lc_mixed_run_execute_observed(
            run, &input, 1, NULL, 0, &config, observed_output,
            &observed_count, &observed_stats, &error, NULL, NULL,
            &inspections
        ) == LC_INSPECTION_OVERFLOW
    );
    assert(error.resource == LC_RESOURCE_INSPECTION);
    assert(error.capacity == 3U && error.occupancy == 3U && error.peak == 3U);
    assert(error.has_event == 1U && error.event_index == 3U);
    assert(error.node == 0U && error.t == 3.0);
    inspections.capacity = 4U;
    assert(
        lc_mixed_run_execute_observed(
            run, &input, 1, NULL, 0, &config, observed_output,
            &observed_count, &observed_stats, &error, NULL, NULL,
            &inspections
        ) == LC_OK
    );
    assert(inspection_count == 4U && observed_count == 1U);
    assert(results[0].t == 0.5 && results[0].values[0] == -65.0);
    assert(
        results[1].t == 1.0 && results[1].values[0] == -65.0 &&
        results[1].clamped == 1U
    );
    assert(
        results[2].t == 2.0 && results[2].values[0] == -65.0 &&
        results[2].clamped == 1U
    );
    assert(
        results[3].t == 3.0 && results[3].values[0] == -65.0 &&
        results[3].clamped == 0U
    );

    assert(lc_mixed_run_reset(run, initial, 1, initial_time, 1) == LC_OK);
    assert(
        lc_mixed_run_execute(
            run, &input, 1, NULL, 0, &config, baseline_output,
            &baseline_count, &baseline_stats, &error
        ) == LC_OK
    );
    assert(baseline_count == observed_count);
    assert(
        observed_output[0].t == baseline_output[0].t &&
        observed_output[0].node == baseline_output[0].node
    );
    observed_stats.kernel_seconds = 0.0;
    baseline_stats.kernel_seconds = 0.0;
    assert(memcmp(&observed_stats, &baseline_stats, sizeof(observed_stats)) == 0);
    lc_mixed_run_destroy(run);
}

static void test_compiled_alpha_graph_owns_expression_programs(void) {
    lc_expr_node program[16] = {
        {LC_EXPR_VAR, 0, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -1.0},
        {LC_EXPR_MUL, 0, 1, 0, 0.0},
        {LC_EXPR_EXP, 2, 0, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 2.0},
        {LC_EXPR_MUL, 3, 4, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 1.0},
        {LC_EXPR_ADD, 5, 1, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, -2.0},
        {LC_EXPR_MUL, 3, 8, 0, 0.0},
        {LC_EXPR_EXP, 0, 0, 0, 0.0},
        {LC_EXPR_MUL, 10, 4, 0, 0.0},
        {LC_EXPR_CONST, 0, 0, 0, 0.0},
        {LC_EXPR_VAR, 0, 0, 1, 0.0},
        {LC_EXPR_VAR, 0, 0, 2, 0.0},
        {LC_EXPR_VAR, 0, 0, 3, 0.0}
    };
    lc_expr_node deposit[1] = {{LC_EXPR_VAR, 0, 0, 0, 0.0}};
    lc_mixed_node node = {0};
    double initial[3] = {-65.0, 0.0, 0.0};
    lc_time_t initial_time[1] = {0.0};
    lc_output_spike output[1];
    lc_run_config config = {1.0, 8, 1, 4, 0};
    lc_run_stats stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    uint64_t output_count = 0;

    node.dispatch = LC_ROOT_FIND;
    node.crossing_kind = LC_CROSSING_ALPHA_REAL;
    node.state_count = 3;
    node.readout = 0;
    node.threshold = -50.0;
    node.program_nodes = program;
    node.program_node_count = 16;
    node.normal_roots[0] = 13;
    node.normal_roots[1] = 14;
    node.normal_roots[2] = 15;
    node.clamped_roots[0] = 13;
    node.clamped_roots[1] = 14;
    node.clamped_roots[2] = 15;
    node.reset_roots[0] = 13;
    node.reset_roots[1] = 14;
    node.reset_roots[2] = 15;
    node.root_hint = (lc_root_hint){
        7, 9, 11, 11, 6, 8, 12, 12, 1, 8, 12, 1e-10, 0.5
    };
    node.deposit_nodes = deposit;
    node.deposit_node_count = 1;
    node.deposit_root = 0;
    node.deposit_target = 2;

    assert(
        lc_mixed_graph_compile(&node, 1, 3, NULL, 0, NULL, 0, &compiled) == LC_OK
    );
    program[7].op = UINT32_MAX;
    deposit[0].op = UINT32_MAX;
    assert(
        lc_mixed_run_create(compiled, initial, 3, initial_time, 1, &run) == LC_OK
    );
    assert(
        lc_mixed_run_execute(
            run, NULL, 0, NULL, 0, &config, output, &output_count, &stats, &error
        ) == LC_OK
    );
    assert(output_count == 1);
    assert(close_enough(output[0].t, log(2.0), 5e-10));
    lc_mixed_run_destroy(run);
    lc_mixed_graph_destroy(compiled);
}

static void test_host_codecs(void) {
    lc_encoder_spec encoders[3] = {0};
    lc_encoder_state states[3];
    lc_presentation presentations[4] = {
        {0.0, 1.0, 0, 1.0},
        {1.0, 2.0, 0, 1.0},
        {10.0, 14.0, 1, 1.0},
        {10.0, 14.0, 2, 1.0}
    };
    lc_encoded_spike encoded[16];
    lc_encoded_drive drives[1];
    uint64_t encoded_count = 0;
    uint64_t drive_count = 0;
    lc_output_spike output[3] = {{0.5, 0}, {1.5, 0}, {4.0, 0}};
    lc_decoder_spec decoders[3] = {0};
    lc_decode_window window = {0.0, 5.0};
    lc_decode_result decoded[3];

    encoders[0].kind = LC_ENCODER_REGULAR_RATE;
    encoders[0].amplitude = 1.0;
    encoders[0].rate_min = 1.0;
    encoders[0].rate_max = 1.0;
    encoders[1].kind = LC_ENCODER_BURST;
    encoders[1].amplitude = 2.0;
    encoders[1].rate_min = 2.0;
    encoders[1].rate_max = 2.0;
    encoders[1].duration = 1.1;
    encoders[2].kind = LC_ENCODER_LATENCY_BURST;
    encoders[2].amplitude = 3.0;
    encoders[2].rate_max = 2.0;
    encoders[2].latency_min = 1.0;
    encoders[2].latency_max = 5.0;
    encoders[2].duration = 1.1;

    assert(lc_encoder_state_reset(states, 3, 7) == LC_OK);
    assert(
        lc_encode_presentations(
            encoders, 3, states, 3, presentations, 4, encoded, 16,
            &encoded_count, drives, 1, &drive_count
        ) == LC_OK
    );
    assert(encoded_count == 7);
    assert(encoded[0].t == 1.0 && encoded[0].encoder == 0);
    assert(encoded[1].t == 10.0 && encoded[1].encoder == 1);
    assert(encoded[4].t == 11.0 && encoded[4].encoder == 2);
    assert(drive_count == 0);

    decoders[0].kind = LC_DECODER_RATE;
    decoders[0].node = 0;
    decoders[0].mode = LC_RATE_FINITE;
    decoders[0].emission = LC_EMIT_ON_WINDOW_CLOSE;
    decoders[1].kind = LC_DECODER_TTFS;
    decoders[1].node = 0;
    decoders[2].kind = LC_DECODER_TEMPORAL_WEIGHT;
    decoders[2].node = 0;
    decoders[2].tau = 2.0;
    decoders[2].first_only = 1;
    assert(lc_decode_spikes(decoders, 3, output, 3, &window, decoded, 3) == LC_OK);
    assert(decoded[0].count == 3 && close_enough(decoded[0].value, 0.6, 1e-12));
    assert(decoded[1].valid && decoded[1].first_spike == 0.5);
    assert(close_enough(decoded[2].value, exp(-0.25), 1e-12));
}

static void test_streaming_decoder_sink_without_raw_buffer(void) {
    lc_mixed_node node = {0};
    double initial[1] = {-65.0};
    lc_time_t initial_time[1] = {0.0};
    lc_mixed_input_spike input = {
        1.0, 0, LC_DEPOSIT_STATE_ADD, 0, 20.0
    };
    lc_decoder_spec specs[3] = {0};
    lc_decode_window window = {0.0, 2.0};
    lc_decode_result decoded[3];
    lc_run_config config = {2.0, 8, 0, 4, 0};
    lc_run_stats first_stats;
    lc_run_stats second_stats;
    lc_network_error error;
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    lc_compiled_decoders *decoder_bank = NULL;
    lc_decoder_run *decoder_run = NULL;
    uint64_t output_count = 0;

    init_reactive_descriptor(&node, 0);

    specs[0].kind = LC_DECODER_RATE;
    specs[0].node = 0;
    specs[0].mode = LC_RATE_FINITE;
    specs[0].emission = LC_EMIT_ON_WINDOW_CLOSE;
    specs[1].kind = LC_DECODER_TTFS;
    specs[1].node = 0;
    specs[2].kind = LC_DECODER_TEMPORAL_WEIGHT;
    specs[2].node = 0;
    specs[2].tau = 2.0;

    assert(
        lc_mixed_graph_compile(&node, 1, 1, NULL, 0, NULL, 0, &compiled) == LC_OK
    );
    assert(
        lc_mixed_run_create(compiled, initial, 1, initial_time, 1, &run) == LC_OK
    );
    assert(lc_decoder_bank_compile(specs, 3, 1, &decoder_bank) == LC_OK);
    assert(lc_decoder_run_create(decoder_bank, &decoder_run) == LC_OK);
    lc_decoder_bank_destroy(decoder_bank);
    decoder_bank = NULL;
    assert(lc_decoder_run_reset(decoder_run, &window) == LC_OK);
    assert(
        lc_mixed_run_execute_with_decoders(
            run, &input, 1, NULL, 0, &config, NULL, &output_count,
            &first_stats, &error, decoder_run
        ) == LC_OK
    );
    assert(output_count == 0);
    assert(first_stats.output_spikes == 1);
    assert(lc_decoder_run_finalize(decoder_run, decoded, 3) == LC_OK);
    assert(decoded[0].count == 1 && close_enough(decoded[0].value, 0.5, 1e-12));
    assert(decoded[1].valid && decoded[1].first_spike == 1.0);
    assert(close_enough(decoded[2].value, exp(-0.5), 1e-12));

    assert(lc_mixed_run_reset(run, initial, 1, initial_time, 1) == LC_OK);
    assert(lc_decoder_run_reset(decoder_run, &window) == LC_OK);
    assert(
        lc_mixed_run_execute_with_decoders(
            run, &input, 1, NULL, 0, &config, NULL, &output_count,
            &second_stats, &error, decoder_run
        ) == LC_OK
    );
    first_stats.kernel_seconds = 0.0;
    second_stats.kernel_seconds = 0.0;
    assert(memcmp(&first_stats, &second_stats, sizeof(lc_run_stats)) == 0);

    lc_decoder_run_destroy(decoder_run);
    lc_mixed_run_destroy(run);
    lc_mixed_graph_destroy(compiled);
}

static void test_streaming_encoder_run_keeps_active_presentations(void) {
    lc_encoder_spec specs[2] = {0};
    lc_presentation initial[2] = {
        {0.0, 10.0, 0, 1.0},
        {0.0, 3.0, 1, 1.0}
    };
    lc_presentation replacement = {1.5, 2.5, 1, 0.5};
    lc_encoded_spike spikes[8];
    lc_encoded_drive drives[8];
    lc_encoder_run *run = NULL;
    uint64_t spike_count = 0U;
    uint64_t drive_count = 0U;

    specs[0].kind = LC_ENCODER_BURST;
    specs[0].amplitude = 2.0;
    specs[0].rate_min = 1.0;
    specs[0].rate_max = 1.0;
    specs[0].duration = 10.0;
    specs[1].kind = LC_ENCODER_HELD_CURRENT;
    specs[1].gain = 2.0;
    specs[1].offset = -1.0;
    specs[1].baseline = -3.0;

    assert(lc_encoder_run_create(specs, 2, 17, 0.0, &run) == LC_OK);
    assert(
        lc_encoder_run_advance(
            run, initial, 2, 1.0, 0, spikes, 8, &spike_count,
            drives, 8, &drive_count
        ) == LC_OK
    );
    assert(spike_count == 1U && spikes[0].t == 0.0);
    assert(drive_count == 1U && drives[0].t == 0.0 && drives[0].value == 1.0);

    assert(
        lc_encoder_run_advance(
            run, &replacement, 1, 2.0, 0, spikes, 8, &spike_count,
            drives, 8, &drive_count
        ) == LC_OK
    );
    assert(spike_count == 1U && spikes[0].t == 1.0);
    assert(drive_count == 2U);
    assert(drives[0].t == 1.5 && drives[0].value == -3.0);
    assert(drives[1].t == 1.5 && drives[1].value == 0.0);

    assert(
        lc_encoder_run_advance(
            run, NULL, 0, 3.0, 1, spikes, 8, &spike_count,
            drives, 8, &drive_count
        ) == LC_OK
    );
    assert(spike_count == 2U && spikes[0].t == 2.0 && spikes[1].t == 3.0);
    assert(drive_count == 1U && drives[0].t == 2.5 && drives[0].value == -3.0);
    assert(
        lc_encoder_run_advance(
            run, NULL, 0, 3.0, 1, spikes, 8, &spike_count,
            drives, 8, &drive_count
        ) == LC_INVALID_ARGUMENT
    );
    lc_encoder_run_destroy(run);
}

static void test_exact_time_decoder_events(void) {
    lc_decoder_spec specs[3] = {0};
    lc_decode_window window = {0.0, 5.0};
    lc_output_spike spikes[3] = {{0.5, 0}, {1.5, 0}, {5.0, 0}};
    lc_decode_result results[3];
    lc_decoded_event events[5];
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *run = NULL;
    uint64_t event_count = 0U;

    specs[0].kind = LC_DECODER_RATE;
    specs[0].node = 0;
    specs[0].mode = LC_RATE_FINITE;
    specs[0].emission = LC_EMIT_ON_WINDOW_CLOSE;
    specs[1].kind = LC_DECODER_TTFS;
    specs[1].node = 0;
    specs[1].emission = LC_EMIT_ON_EVENT;
    specs[2].kind = LC_DECODER_TEMPORAL_WEIGHT;
    specs[2].node = 0;
    specs[2].tau = 2.0;
    specs[2].emission = LC_EMIT_ON_EVENT_AND_WINDOW_CLOSE;

    assert(lc_decoder_bank_compile(specs, 3, 1, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &run) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 5) == LC_OK);
    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[0]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[1]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[2]) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 3) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 3) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[0]) == LC_INVALID_ARGUMENT);
    assert(lc_decoder_run_copy_events(run, events, 5, &event_count) == LC_OK);
    assert(event_count == 5U);
    assert(events[0].decoder == 1U && events[0].kind == LC_DECODE_FINAL);
    assert(events[0].emitted_at == 0.5 && events[0].source_spike_time == 0.5);
    assert(events[0].observed_through == 0.5 && events[0].value == 0.5);
    assert(events[1].decoder == 2U && events[1].kind == LC_DECODE_UPDATE);
    assert(events[1].emitted_at == 0.5 && events[1].count == 1U);
    assert(events[2].decoder == 2U && events[2].kind == LC_DECODE_UPDATE);
    assert(events[2].emitted_at == 1.5 && events[2].count == 2U);
    assert(events[3].decoder == 0U && events[3].kind == LC_DECODE_FINAL);
    assert(events[3].emitted_at == 5.0 && events[3].count == 2U);
    assert(events[4].decoder == 2U && events[4].kind == LC_DECODE_FINAL);
    assert(events[4].emitted_at == 5.0 && events[4].count == 2U);

    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 3) == LC_OK);
    assert(lc_decoder_run_copy_events(run, events, 5, &event_count) == LC_OK);
    assert(event_count == 3U);
    assert(events[1].decoder == 1U && events[1].kind == LC_DECODE_NO_SPIKE);
    assert(events[1].emitted_at == 5.0 && !events[1].valid);

    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 2) == LC_OK);
    assert(
        lc_decoder_run_finalize(run, results, 3) == LC_DECODER_OUTPUT_OVERFLOW
    );
    assert(lc_decoder_run_copy_events(run, events, 2, &event_count) == LC_OK);
    assert(event_count == 0U);
    assert(lc_decoder_run_reserve_events(run, 3) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 3) == LC_OK);

    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 1) == LC_OK);
    assert(
        lc_decoder_run_consume(run, &spikes[0]) == LC_DECODER_OUTPUT_OVERFLOW
    );
    assert(lc_decoder_run_reset(run, &window) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 0) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[0]) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 3) == LC_OK);
    assert(results[1].valid && results[1].value == 0.5);

    lc_decoder_run_destroy(run);
    lc_decoder_bank_destroy(bank);
}

static void test_multi_window_decoder_schedule(void) {
    lc_decoder_spec specs[2] = {0};
    lc_decode_window windows[3] = {{0.0, 2.0}, {2.0, 4.0}, {4.0, 6.0}};
    lc_output_spike spikes[3] = {{0.0, 0}, {2.0, 0}, {4.0, 0}};
    lc_decode_result results[6];
    lc_decoded_event events[6];
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *run = NULL;
    uint64_t event_count = 0U;
    uint64_t result_count = 0U;

    specs[0].kind = LC_DECODER_RATE;
    specs[0].node = 0;
    specs[0].mode = LC_RATE_FINITE;
    specs[0].emission = LC_EMIT_ON_WINDOW_CLOSE;
    specs[1].kind = LC_DECODER_TTFS;
    specs[1].node = 0;
    specs[1].emission = LC_EMIT_ON_EVENT;

    assert(lc_decoder_bank_compile(specs, 2, 1, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &run) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 6) == LC_OK);
    assert(lc_decoder_run_reset_windows(run, windows, 3) == LC_OK);
    assert(lc_decoder_run_result_count(run, &result_count) == LC_OK);
    assert(result_count == 6U);
    assert(lc_decoder_run_consume(run, &spikes[0]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[1]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[2]) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 6) == LC_OK);
    assert(lc_decoder_run_copy_events(run, events, 6, &event_count) == LC_OK);
    assert(event_count == 6U);
    assert(events[0].window == 0U && events[0].decoder == 1U);
    assert(events[0].emitted_at == 0.0);
    assert(events[1].window == 0U && events[1].decoder == 0U);
    assert(events[1].emitted_at == 2.0 && !events[1].has_source);
    assert(events[2].window == 1U && events[2].decoder == 1U);
    assert(events[2].emitted_at == 2.0 && events[2].has_source);
    assert(events[3].window == 1U && events[3].decoder == 0U);
    assert(events[3].emitted_at == 4.0);
    assert(events[4].window == 2U && events[4].decoder == 1U);
    assert(events[4].emitted_at == 4.0);
    assert(events[5].window == 2U && events[5].decoder == 0U);
    assert(events[5].emitted_at == 6.0);
    assert(results[0].window == 0U && results[0].decoder == 0U);
    assert(results[0].count == 1U && results[0].value == 0.5);
    assert(results[3].window == 1U && results[3].decoder == 1U);
    assert(results[3].count == 1U && results[3].value == 0.0);
    assert(results[4].window == 2U && results[4].decoder == 0U);
    assert(results[4].count == 1U && results[4].value == 0.5);

    lc_decoder_run_destroy(run);
    lc_decoder_bank_destroy(bank);
}

static void test_sparse_per_decoder_window_schedule(void) {
    lc_decoder_spec specs[2] = {0};
    lc_decoder_window_binding schedule[4] = {
        {0, 0, 0.0, 2.0},
        {0, 1, 2.0, 4.0},
        {0, 2, 4.0, 6.0},
        {1, 7, 0.0, 6.0}
    };
    lc_decoder_window_binding duplicate[2] = {
        {0, 0, 0.0, 2.0}, {0, 0, 2.0, 4.0}
    };
    lc_output_spike spikes[3] = {{0.0, 0}, {2.0, 0}, {4.0, 0}};
    lc_decode_result results[4];
    lc_decoded_event events[4];
    lc_compiled_decoders *bank = NULL;
    lc_decoder_run *run = NULL;
    uint64_t event_count = 0U;
    uint64_t result_count = 0U;

    specs[0].kind = LC_DECODER_RATE;
    specs[0].node = 0;
    specs[0].mode = LC_RATE_FINITE;
    specs[0].emission = LC_EMIT_ON_WINDOW_CLOSE;
    specs[1].kind = LC_DECODER_TTFS;
    specs[1].node = 0;
    specs[1].emission = LC_EMIT_ON_EVENT;

    assert(lc_decoder_bank_compile(specs, 2, 1, &bank) == LC_OK);
    assert(lc_decoder_run_create(bank, &run) == LC_OK);
    assert(lc_decoder_run_reserve_events(run, 4) == LC_OK);
    assert(lc_decoder_run_reset_schedule(run, duplicate, 2) == LC_INVALID_ARGUMENT);
    assert(lc_decoder_run_reset_schedule(run, schedule, 4) == LC_OK);
    assert(lc_decoder_run_result_count(run, &result_count) == LC_OK);
    assert(result_count == 4U);
    assert(lc_decoder_run_consume(run, &spikes[0]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[1]) == LC_OK);
    assert(lc_decoder_run_consume(run, &spikes[2]) == LC_OK);
    assert(lc_decoder_run_finalize(run, results, 4) == LC_OK);
    assert(lc_decoder_run_copy_events(run, events, 4, &event_count) == LC_OK);
    assert(event_count == 4U);
    assert(events[0].decoder == 1U && events[0].window == 7U);
    assert(events[0].emitted_at == 0.0);
    assert(events[1].decoder == 0U && events[1].window == 0U);
    assert(events[1].emitted_at == 2.0);
    assert(events[2].decoder == 0U && events[2].window == 1U);
    assert(events[2].emitted_at == 4.0);
    assert(events[3].decoder == 0U && events[3].window == 2U);
    assert(events[3].emitted_at == 6.0);
    assert(results[0].decoder == 0U && results[0].window == 0U);
    assert(results[2].decoder == 0U && results[2].window == 2U);
    assert(results[3].decoder == 1U && results[3].window == 7U);
    assert(results[3].count == 1U && results[3].value == 0.0);

    lc_decoder_run_destroy(run);
    lc_decoder_bank_destroy(bank);
}

static void test_generic_dopri_stepper_and_crossing(void) {
    const lc_expr_node rhs_nodes[1] = {
        {LC_EXPR_VAR, 0, 0, 1, 0.0}
    };
    const uint32_t rhs_roots[1] = {0U};
    lc_step_config config = {
        1e-10, 1e-12, 1e-3, 1e-12, 0.1, 1e-10, 100000U, 700000U
    };
    double variables[2] = {0.0, 0.0};
    double workspace[1] = {0.0};
    double state[1] = {1.0};
    lc_time_t t_last = 0.0;
    lc_step_result result;

    assert(
        lc_expr_step_predict(
            rhs_nodes, 1U, NULL, 0U, rhs_roots, 1U, 0U, 2.0,
            &config, state, 0.0, 2.0, variables, 2U, workspace, 1U,
            &result
        ) == LC_OK
    );
    assert(close_enough(result.t_crossing, log(2.0), 2e-9));
    assert(result.accepted_steps > 0U && result.rhs_evaluations > 0U);
    assert(state[0] == 1.0);

    assert(
        lc_expr_step_advance(
            rhs_nodes, 1U, NULL, 0U, rhs_roots, 1U, 0U, &config,
            state, &t_last, 1.0, 0U, variables, 2U, workspace, 1U,
            &result
        ) == LC_OK
    );
    assert(close_enough(state[0], exp(1.0), 2e-10));
    assert(t_last == 1.0);

    state[0] = -3.0;
    t_last = 0.0;
    assert(
        lc_expr_step_advance(
            rhs_nodes, 1U, NULL, 0U, rhs_roots, 1U, 0U, &config,
            state, &t_last, 1.0, 1U, variables, 2U, workspace, 1U,
            &result
        ) == LC_OK
    );
    assert(state[0] == -3.0);
}

int main(void) {
    lc_scalar_lif_model driven = {
        -0.1, -4.5, -50.0, -65.0, 2.0, LC_EXCITATORY
    };
    lc_scalar_lif_model reactive = {
        -0.1, -5.5, -50.0, -65.0, 2.0, LC_EXCITATORY
    };
    lc_scalar_state state = {-65.0, 0.0};
    lc_time_t crossing = 0.0;
    lc_dispatch_form dispatch = LC_STEPPED;

    assert(lc_abi_version() == LC_ABI_VERSION);
    assert(lc_sizeof_network_error() == sizeof(lc_network_error));
    lc_encoder_run_destroy(NULL);
    lc_decoder_run_destroy(NULL);
    lc_decoder_bank_destroy(NULL);
    lc_mixed_run_destroy(NULL);
    lc_mixed_graph_destroy(NULL);

    assert(lc_scalar_advance(&driven, &state, 10.0) == LC_OK);
    assert(close_enough(state.value, -45.0 - 20.0 * exp(-1.0), 1e-12));
    assert(state.t_last == 10.0);

    state.value = -65.0;
    state.t_last = 0.0;
    assert(lc_scalar_predict(&driven, &state, &crossing, &dispatch) == LC_OK);
    assert(dispatch == LC_CLOSED_FORM);
    assert(close_enough(crossing, 10.0 * log(4.0), 1e-12));

    assert(lc_scalar_predict(&reactive, &state, &crossing, &dispatch) == LC_NO_CROSSING);
    assert(dispatch == LC_REACTIVE);

    assert(lc_scalar_advance(&driven, &state, -1.0) == LC_TIME_REVERSED);
    test_zero_delay_cascade();
    test_dale_polarity_signs_nonnegative_edge_magnitudes();
    test_scalar_mixed_polarity_signed_fanout();
    test_same_time_deposits_are_aggregated();
    test_stale_prediction_is_discarded();
    test_fixed_refractory_clamp_releases_and_reschedules();
    test_drive_update_uses_old_b_before_boundary();
    test_expression_dag_evaluator();
    test_selected_expression_evaluator_ignores_unreachable_nodes();
    test_expression_state_advance_and_deposit();
    test_stable_phi_expression_operations();
    test_certified_alpha_root_finder();
    test_adaptive_state_map_and_two_exp_root_finder();
    test_bounded_multi_exp_root_finder();
    test_bounded_exp_polynomial_root_finder();
    test_mixed_runner_scalar_csr_path();
    test_capability_descriptor_is_independent_of_model_identity();
    test_compiled_graph_session_reset_and_lifetime();
    test_compiled_graph_image_round_trip();
    test_compiled_mixed_polarity_weights_and_image();
    test_mixed_polarity_rejects_online_plasticity();
    test_incremental_compiled_run_keeps_boundary_open();
    test_configurable_causal_trace();
    test_explicit_time_state_inspection_is_read_only();
    test_compiled_alpha_graph_owns_expression_programs();
    test_host_codecs();
    test_streaming_encoder_run_keeps_active_presentations();
    test_streaming_decoder_sink_without_raw_buffer();
    test_exact_time_decoder_events();
    test_multi_window_decoder_schedule();
    test_sparse_per_decoder_window_schedule();
    test_generic_dopri_stepper_and_crossing();
    return 0;
}
