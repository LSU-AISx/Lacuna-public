#include "lacuna.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#if LACUNA_REAL_BITS != 16 || LACUNA_TIME_BITS != 16
#error "This regression requires the uniform binary16 runtime"
#endif

static uint16_t real_bits(lc_real_t value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void test_half_metadata_and_expression_rounding(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(2048.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(1.0)},
        {LC_EXPR_ADD, 0U, 1U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_SUB, 2U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.5)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0001)},
        {LC_EXPR_ADD, 4U, 5U, 0U, LC_REAL_C(0.0)}
    };
    lc_real_t workspace[7];
    assert(sizeof(lc_real_t) == 2U && sizeof(lc_time_t) == 2U && sizeof(lc_wide_t) == 2U);
    assert(lc_abi_version() == 19U);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_PROFILE) == LC_NUMERIC_PROFILE_BINARY16);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_REAL_MANT_DIG) == 11U);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_TIME_MANT_DIG) == 11U);
    assert(lc_numeric_property(LC_NUMERIC_PROPERTY_WIDE_MANT_DIG) == 11U);
    assert(lc_numeric_profile_check(LC_NUMERIC_PROFILE_BINARY16, 16U, 16U,
        LC_NUMERIC_ARITHMETIC_REVISION) == LC_OK);
    assert(lc_numeric_profile_check(LC_NUMERIC_PROFILE_BINARY32, 32U, 32U,
        LC_NUMERIC_ARITHMETIC_REVISION) == LC_INVALID_ARGUMENT);
    assert(lc_expr_evaluate(nodes, 7U, NULL, 0U, NULL, 0U, workspace, 7U) == LC_OK);
    assert(workspace[2] == LC_REAL_C(2048.0));
    assert(workspace[3] == LC_REAL_C(0.0));
    assert(workspace[6] == LC_REAL_C(0.5));
    assert(real_bits(lc_real_nextafter(LC_REAL_C(0.0), LC_REAL_C(1.0))) == 1U);
    assert(real_bits(-LC_REAL_C(0.0)) == UINT16_C(0x8000));
}

static void test_half_scalar_lif(void) {
    lc_scalar_lif_model model = {
        -LC_REAL_C(1.0), LC_REAL_C(2.0), LC_REAL_C(1.0),
        LC_REAL_C(0.0), LC_TIME_C(0.25), LC_EXCITATORY
    };
    lc_scalar_state state = {LC_REAL_C(0.0), LC_TIME_C(0.0)};
    lc_time_t spike;
    lc_dispatch_form dispatch;
    assert(lc_scalar_predict(&model, &state, &spike, &dispatch) == LC_OK);
    assert(dispatch == LC_CLOSED_FORM);
    assert(spike == LC_TIME_C(0.693359375)); /* Correctly rounded log(2). */
    assert(lc_scalar_advance(&model, &state, LC_TIME_C(0.25)) == LC_OK);
    assert(state.value == LC_REAL_C(0.4423828125));
    assert(state.t_last == LC_TIME_C(0.25));
    state.value = LC_REAL_C(0.0);
    state.t_last = LC_TIME_C(2048.0);
    assert(lc_scalar_predict(&model, &state, &spike, &dispatch) == LC_NUMERIC_ERROR);
}

static void test_half_numerical_crossing_cannot_collapse(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(1.0)}
    };
    const uint32_t roots[] = {0U};
    lc_real_t state[] = {LC_REAL_C(0.0)};
    lc_real_t variables[2], workspace[1];
    lc_step_config config = {
        LC_REAL_C(0x1p-7), LC_REAL_C(0x1p-24), LC_TIME_C(2.0),
        LC_TIME_C(0x1p-24), LC_TIME_C(2.0), LC_TIME_C(0x1p-10), 1000U, 7000U
    };
    lc_step_result result;
    /* The complete step advances, but the crossing's positive 0.25 interval
       cannot advance this coarse clock. It must not become a same-time event. */
    assert(lc_expr_step_predict(nodes, 1U, NULL, 0U, roots, 1U, 0U,
        LC_REAL_C(0.25), &config, state, LC_TIME_C(2048.0), LC_TIME_C(2050.0),
        variables, 2U, workspace, 1U, &result) == LC_NUMERIC_ERROR);
}

static void test_half_derivative_underflow_cannot_certify_no_crossing(void) {
    const lc_expr_node nodes[] = {
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0x1.8p-16)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0x1p-13)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(0x1p-13)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(0x1p-13)},
        {LC_EXPR_CONST, 0U, 0U, 0U, -LC_REAL_C(0x1p-14)},
        {LC_EXPR_VAR, 0U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 3U, 5U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_EXP, 6U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 1U, 7U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 4U, 5U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_EXP, 9U, 0U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 2U, 10U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_ADD, 0U, 8U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_ADD, 12U, 11U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 1U, 3U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 14U, 7U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 2U, 4U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_MUL, 16U, 10U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_ADD, 15U, 17U, 0U, LC_REAL_C(0.0)},
        {LC_EXPR_CONST, 0U, 0U, 0U, LC_REAL_C(0.0)}
    };
    lc_multi_exp_hint multi = {0};
    lc_exp_poly_hint polynomial = {0};
    lc_two_exp_hint two = {13U, 18U, 0U, 1U, 2U, 3U, 4U,
        LC_REAL_C(0x1p-9), LC_TIME_C(8192.0)};
    lc_root_hint alpha = {13U, 18U, 18U, 19U, 19U, 2U, 1U, 19U, 3U, 4U, 0U,
        LC_REAL_C(0x1p-9), LC_TIME_C(8192.0)};
    lc_root_result result;
    lc_real_t state[3] = {0};
    lc_real_t variables[4] = {LC_REAL_C(11360.0), 0, 0, 0};
    lc_real_t workspace[20];
    /* The half-evaluated pulse really crosses, despite both derived rates
       underflowing to zero. Returning NO_CROSSING would lose that event. */
    assert(lc_expr_evaluate(nodes, 20U, NULL, 0U, variables, 4U, workspace, 20U) == LC_OK);
    assert(workspace[13] < LC_REAL_C(0.0));
    assert(workspace[14] == LC_REAL_C(0.0) && workspace[16] == LC_REAL_C(0.0));
    multi.limit_root = 0U;
    multi.coefficient_roots[0] = 1U;
    multi.coefficient_roots[1] = 2U;
    multi.rate_roots[0] = 3U;
    multi.rate_roots[1] = 4U;
    multi.mode_count = 2U;
    multi.iteration_cap = 96U;
    multi.relative_tolerance = LC_REAL_C(0x1p-9);
    multi.fastest_time_constant = LC_TIME_C(8192.0);
    polynomial.limit_root = 0U;
    polynomial.coefficient_roots[0] = 1U;
    polynomial.coefficient_roots[1] = 2U;
    polynomial.rate_roots[0] = 3U;
    polynomial.rate_roots[1] = 4U;
    polynomial.coefficient_offsets[1] = 1U;
    polynomial.coefficient_offsets[2] = 2U;
    polynomial.block_count = polynomial.coefficient_count = 2U;
    polynomial.iteration_cap = 96U;
    polynomial.relative_tolerance = LC_REAL_C(0x1p-9);
    polynomial.fastest_time_constant = LC_TIME_C(8192.0);
    assert(lc_expr_multi_exp_predict(nodes, 20U, NULL, 0U, &multi, state, 3U,
        LC_TIME_C(0.0), &result, variables, 4U, workspace, 20U) == LC_ROOT_NONCONVERGENCE);
    assert(lc_expr_exp_poly_predict(nodes, 20U, NULL, 0U, &polynomial, state, 3U,
        LC_TIME_C(0.0), &result, variables, 4U, workspace, 20U) == LC_ROOT_NONCONVERGENCE);
    assert(lc_expr_two_exp_predict(nodes, 20U, NULL, 0U, &two, state, 2U,
        LC_TIME_C(0.0), &result, variables, 3U, workspace, 20U) == LC_ROOT_NONCONVERGENCE);
    assert(lc_expr_alpha_predict(nodes, 20U, NULL, 0U, &alpha, state, 3U,
        LC_TIME_C(0.0), &result, variables, 4U, workspace, 20U) == LC_ROOT_NONCONVERGENCE);
}

int main(void) {
    test_half_metadata_and_expression_rounding();
    test_half_scalar_lif();
    test_half_numerical_crossing_cannot_collapse();
    test_half_derivative_underflow_cannot_certify_no_crossing();
    return 0;
}
