#include "lacuna.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static double accuracy(void) {
    return LACUNA_REAL_BITS == 32 ? 3e-5 : 1e-8;
}

static lc_real_t root_tolerance(void) {
    return LACUNA_REAL_BITS == 32 ? LC_REAL_C(1e-6) : LC_REAL_C(1e-10);
}

static void test_numeric_profile(void) {
    uint32_t profile = LACUNA_REAL_BITS == 64 ? LC_NUMERIC_PROFILE_BINARY64 :
        (LACUNA_TIME_BITS == 64 ? LC_NUMERIC_PROFILE_BINARY32_TIME64 :
         LC_NUMERIC_PROFILE_BINARY32);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_PROFILE) == profile);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_REAL_BITS) == LACUNA_REAL_BITS);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_TIME_BITS) == LACUNA_TIME_BITS);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_REAL_MANT_DIG) == LC_REAL_MANT_DIG);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_TIME_MANT_DIG) == LC_TIME_MANT_DIG);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_WIDE_MANT_DIG) == LC_WIDE_MANT_DIG);
    assert(lc_numeric_profile_check(profile, LACUNA_REAL_BITS,
        LACUNA_TIME_BITS, LC_NUMERIC_ARITHMETIC_REVISION) == LC_OK);
    assert(lc_numeric_profile_check(LC_NUMERIC_PROFILE_UNKNOWN, LACUNA_REAL_BITS,
        LACUNA_TIME_BITS, LC_NUMERIC_ARITHMETIC_REVISION) == LC_INVALID_ARGUMENT);
    assert(lc_abi_version() == (LACUNA_REAL_BITS == 64 ? 17U : 18U));
}

static void test_native_expression_arithmetic(void) {
    lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(16777216.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(1.0)},
        {LC_EXPR_ADD, 0U, 1U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_SUB, 2U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.25)},
        {LC_EXPR_EXP, 4U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_LOG, 5U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_SIN, 4U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_COS, 4U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_TANH, 4U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_POW, 4U, 4U, 0U, LC_REAL_C(0.0)}
    };
    lc_real_t values[11];
    assert(lc_expr_evaluate(nodes, 11U, NULL, 0U, NULL, 0U,
                           values, 11U) == LC_OK);
    assert(values[3] == (LACUNA_REAL_BITS == 32 ? LC_REAL_C(0.0) : LC_REAL_C(1.0)));
    assert(fabs((double)values[5] - exp(0.25)) < accuracy());
    assert(fabs((double)values[6] - 0.25) < accuracy());
    assert(fabs((double)values[7] - sin(0.25)) < accuracy());
    assert(fabs((double)values[8] - cos(0.25)) < accuracy());
    assert(fabs((double)values[9] - tanh(0.25)) < accuracy());
    assert(fabs((double)values[10] - pow(0.25, 0.25)) < accuracy());
}

static void test_phi_derivative_near_zero(void) {
    lc_expr_node nodes[] = {
        {LC_EXPR_PARAM, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_PHI1, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_PHI1_DERIV, 0U, 0U, 0U, LC_REAL_C(0.0)}
    };
    lc_real_t values[3];
    lc_real_t samples[] = {
        LC_REAL_C(0.0), LC_REAL_C(1e-6), LC_REAL_C(-1e-6),
        LC_REAL_C(1e-4), LC_REAL_C(-1e-4), LC_REAL_C(0.12),
        LC_REAL_C(-0.12), LC_REAL_C(0.125), LC_REAL_C(-0.125)
    };
    size_t index;
    for (index = 0U; index < sizeof(samples) / sizeof(samples[0]); ++index) {
        double x = (double)samples[index];
        double phi = x == 0.0 ? 1.0 : expm1(x) / x;
        double derivative = fabs(x) < 1e-3 ?
            0.5 + x * (1.0 / 3.0 + x * (1.0 / 8.0 + x / 30.0)) :
            (x * exp(x) - expm1(x)) / (x * x);
        assert(lc_expr_evaluate(nodes, 3U, &samples[index], 1U, NULL, 0U,
                               values, 3U) == LC_OK);
        assert(fabs((double)values[1] - phi) < accuracy());
        assert(fabs((double)values[2] - derivative) < accuracy());
    }
}

static void test_scalar_trajectory_and_clock(void) {
    lc_scalar_lif_model model = {
        -LC_REAL_C(1.0), LC_REAL_C(2.0), LC_REAL_C(1.0),
        LC_REAL_C(0.0), LC_TIME_C(0.1), LC_EXCITATORY
    };
    lc_scalar_state state = {LC_REAL_C(0.0), LC_TIME_C(0.0)};
    lc_time_t spike;
    lc_dispatch_form dispatch;
    assert(lc_scalar_predict(&model, &state, &spike, &dispatch) == LC_OK);
    assert(dispatch == LC_CLOSED_FORM);
    assert(fabs((double)spike - log(2.0)) < accuracy());
    assert(lc_scalar_advance(&model, &state, LC_TIME_C(0.25)) == LC_OK);
    assert(fabs((double)state.value - 2.0 * (1.0 - exp(-0.25))) < accuracy());
    assert(lc_scalar_predict(&model, &state, &spike, &dispatch) == LC_OK);
    assert(fabs((double)spike - log(2.0)) < accuracy());
#if LACUNA_TIME_BITS == 32
    state.value = LC_REAL_C(0.0);
    state.t_last = LC_TIME_C(16777216.0);
    assert(lc_scalar_predict(&model, &state, &spike, &dispatch) == LC_NUMERIC_ERROR);
#endif
}

static lc_step_config step_config(void) {
    lc_step_config config = {
        LC_REAL_C(1e-5), LC_REAL_C(1e-10), LC_TIME_C(0.1),
        LC_TIME_C(1e-7), LC_TIME_C(0.25), LC_TIME_C(1e-6), 10000U, 70000U
    };
    if (LACUNA_REAL_BITS == 32) {
        config.relative_tolerance = LC_REAL_C(64.0) * LC_REAL_EPSILON;
        config.event_tolerance = LC_TIME_C(8.0) * (lc_time_t)LC_REAL_EPSILON;
    }
    return config;
}

static void test_stepped_time_input(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_VAR, 0U, 0U, 0U, LC_REAL_C(0.0)}
    };
    const uint32_t roots[] = {0U};
    lc_real_t state[] = {LC_REAL_C(0.0)};
    lc_real_t variables[2];
    lc_real_t workspace[1];
    lc_time_t last = LC_TIME_C(0.0);
    lc_step_result result;
    lc_step_config config = step_config();
    assert(lc_expr_step_advance(nodes, 1U, NULL, 0U, roots, 1U, 0U,
        &config, state, &last, LC_TIME_C(2.0), 0U,
        variables, 2U, workspace, 1U, &result) == LC_OK);
    assert(fabs((double)state[0] - 2.0) < accuracy());
}

static void test_stepped_trial_overflow(void) {
#if LACUNA_REAL_BITS == 32
    const lc_expr_node nodes[] = {
        {LC_EXPR_VAR, 0U, 0U, 1U, LC_REAL_C(0.0)},
        {LC_EXPR_EXP, 0U, 0U, 0U, LC_REAL_C(0.0)}
    };
    const uint32_t roots[] = {1U};
    lc_real_t state[] = {LC_REAL_C(0.0)};
    lc_real_t variables[2];
    lc_real_t workspace[2];
    lc_step_result result;
    lc_step_config config = step_config();
    config.initial_step = LC_TIME_C(1.0);
    config.maximum_step = LC_TIME_C(1.0);
    assert(lc_expr_step_predict(nodes, 2U, NULL, 0U, roots, 1U, 0U,
        LC_REAL_C(1.0), &config, state, LC_TIME_C(0.0), LC_TIME_C(1.0),
        variables, 2U, workspace, 2U, &result) == LC_OK);
    assert(fabs((double)result.t_crossing - (1.0 - exp(-1.0))) < 1e-4);
    assert(result.rejected_steps > 0U);
    assert(state[0] == LC_REAL_C(0.0));
    state[0] = LC_REAL_C(1000.0);
    assert(lc_expr_step_predict(nodes, 2U, NULL, 0U, roots, 1U, 0U,
        LC_REAL_C(2000.0), &config, state, LC_TIME_C(0.0), LC_TIME_C(1.0),
        variables, 2U, workspace, 2U, &result) == LC_NUMERIC_ERROR);
#endif
}

static void test_stepped_oscillator(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_VAR, 0U, 0U, 2U, LC_REAL_C(0.0)},
        {LC_EXPR_VAR, 0U, 0U, 1U, LC_REAL_C(0.0)},
        {LC_EXPR_NEG, 1U, 0U, 0U, LC_REAL_C(0.0)}
    };
    const uint32_t roots[] = {0U, 2U};
    lc_real_t state[] = {LC_REAL_C(0.0), LC_REAL_C(1.0)};
    lc_real_t variables[3];
    lc_real_t workspace[3];
    lc_time_t last = LC_TIME_C(0.0);
    lc_step_result result;
    lc_step_config config = step_config();
    assert(lc_expr_step_advance(nodes, 3U, NULL, 0U, roots, 2U, 0U,
        &config, state, &last, LC_TIME_C(6.283185307179586), 0U,
        variables, 3U, workspace, 3U, &result) == LC_OK);
    assert(fabs((double)state[0]) < 1e-4);
    assert(fabs((double)state[1] - 1.0) < 1e-4);
    assert(result.accepted_steps > 0U);
}

static void test_stepped_interior_crossing(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(4.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(8.0)},
        {LC_EXPR_VAR, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 1U, 2U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_SUB, 0U, 3U, 0U, LC_REAL_C(0.0)}
    };
    const uint32_t roots[] = {4U};
    const lc_real_t state[] = {LC_REAL_C(0.0)};
    lc_real_t variables[2];
    lc_real_t workspace[5];
    lc_step_result result;
    lc_step_config config = step_config();
    config.initial_step = LC_TIME_C(1.0);
    config.maximum_step = LC_TIME_C(1.0);
    assert(lc_expr_step_predict(nodes, 5U, NULL, 0U, roots, 1U, 0U,
        LC_REAL_C(0.75), &config, state, LC_TIME_C(0.0), LC_TIME_C(1.0),
        variables, 2U, workspace, 5U, &result) == LC_OK);
    assert(fabs((double)result.t_crossing - 0.25) < 1e-4);
    config.event_tolerance = LC_TIME_C(1e-20);
    assert(lc_expr_step_predict(nodes, 5U, NULL, 0U, roots, 1U, 0U,
        LC_REAL_C(0.75), &config, state, LC_TIME_C(0.0), LC_TIME_C(1.0),
        variables, 2U, workspace, 5U, &result) == LC_OK);
    assert(fabs((double)result.t_crossing - 0.25) < 1e-4);
}

static void test_multi_exponential_roots(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(2.0)}
    };
    const lc_real_t state[] = {LC_REAL_C(0.0), LC_REAL_C(0.0)};
    lc_real_t variables[3];
    lc_real_t workspace[4];
    lc_multi_exp_hint hint;
    lc_root_result result;
    memset(&hint, 0, sizeof(hint));
    hint.limit_root = 0U;
    hint.coefficient_roots[0] = 1U;
    hint.coefficient_roots[1] = 1U;
    hint.rate_roots[0] = 2U;
    hint.rate_roots[1] = 3U;
    hint.mode_count = 2U;
    hint.iteration_cap = 512U;
    hint.relative_tolerance = root_tolerance();
    hint.fastest_time_constant = LC_TIME_C(0.5);
    assert(lc_expr_multi_exp_predict(nodes, 4U, NULL, 0U, &hint,
        state, 2U, LC_TIME_C(0.0), &result, variables, 3U,
        workspace, 4U) == LC_OK);
    assert(fabs((double)result.t_spike - 0.48121182505960347) < accuracy());
}

static void test_polynomial_exponential_roots(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.25)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0)}
    };
    const lc_real_t state[] = {LC_REAL_C(0.0), LC_REAL_C(0.0)};
    lc_real_t variables[3];
    lc_real_t workspace[3];
    lc_exp_poly_hint hint;
    lc_root_result result;
    memset(&hint, 0, sizeof(hint));
    hint.limit_root = 0U;
    hint.rate_roots[0] = 1U;
    hint.coefficient_roots[0] = 2U;
    hint.coefficient_roots[1] = 1U;
    hint.coefficient_offsets[1] = 2U;
    hint.block_count = 1U;
    hint.coefficient_count = 2U;
    hint.iteration_cap = 512U;
    hint.relative_tolerance = root_tolerance();
    hint.fastest_time_constant = LC_TIME_C(1.0);
    assert(lc_expr_exp_poly_predict(nodes, 3U, NULL, 0U, &hint,
        state, 2U, LC_TIME_C(0.0), &result, variables, 3U,
        workspace, 3U) == LC_OK);
    assert(fabs((double)result.t_spike - 0.3574029561813889) < accuracy());
    hint.iteration_cap = 1U;
    assert(lc_expr_exp_poly_predict(nodes, 3U, NULL, 0U, &hint,
        state, 2U, LC_TIME_C(0.0), &result, variables, 3U,
        workspace, 3U) == LC_ROOT_NONCONVERGENCE);
}

static void test_threshold_tangencies_are_not_spikes(void) {
    lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.25)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(1.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(2.0)}
    };
    const lc_real_t state[] = {LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(0.0)};
    lc_real_t variables[4];
    lc_real_t workspace[4];
    lc_multi_exp_hint multi;
    lc_exp_poly_hint poly;
    lc_root_result result;
    lc_status status;
    memset(&multi, 0, sizeof(multi));
    multi.limit_root = 0U;
    multi.coefficient_roots[0] = 1U;
    multi.coefficient_roots[1] = 2U;
    multi.rate_roots[0] = 1U;
    multi.rate_roots[1] = 3U;
    multi.mode_count = 2U;
    multi.iteration_cap = 512U;
    multi.relative_tolerance = root_tolerance();
    multi.fastest_time_constant = LC_TIME_C(0.5);
    status = lc_expr_multi_exp_predict(nodes, 4U, NULL, 0U, &multi,
        state, 3U, LC_TIME_C(0.0), &result, variables, 4U, workspace, 4U);
    assert(status == (LACUNA_REAL_BITS == 32 ? LC_ROOT_NONCONVERGENCE : LC_NO_CROSSING));

    memset(&poly, 0, sizeof(poly));
    nodes[0].value = LC_REAL_C(0.0);
    poly.limit_root = 0U;
    poly.rate_roots[0] = 1U;
    poly.coefficient_roots[0] = 2U;
    poly.coefficient_roots[1] = 3U;
    poly.coefficient_roots[2] = 2U;
    poly.coefficient_offsets[1] = 3U;
    poly.coefficient_count = 3U;
    poly.block_count = 1U;
    poly.iteration_cap = 512U;
    poly.relative_tolerance = root_tolerance();
    poly.fastest_time_constant = LC_TIME_C(1.0);
    status = lc_expr_exp_poly_predict(nodes, 4U, NULL, 0U, &poly,
        state, 3U, LC_TIME_C(0.0), &result, variables, 4U, workspace, 4U);
    assert(status == (LACUNA_REAL_BITS == 32 ? LC_ROOT_NONCONVERGENCE : LC_NO_CROSSING));
}

int main(void) {
    test_numeric_profile();
    test_native_expression_arithmetic();
    test_phi_derivative_near_zero();
    test_scalar_trajectory_and_clock();
    test_stepped_oscillator();
    test_stepped_time_input();
    test_stepped_trial_overflow();
    test_stepped_interior_crossing();
    test_multi_exponential_roots();
    test_polynomial_exponential_roots();
    test_threshold_tangencies_are_not_spikes();
    return 0;
}
