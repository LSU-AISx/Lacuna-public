#include "lacuna_target.h"

#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

static void test_wire_queries(void) {
    assert(lc_target_abi_version() == LC_TARGET_ABI_VERSION);
    assert(lc_target_sizeof_expr_node() == sizeof(lc_target_expr_node));
    assert(lc_target_expr_node_offset(0U) == offsetof(lc_target_expr_node, op));
    assert(lc_target_expr_node_offset(1U) == offsetof(lc_target_expr_node, lhs));
    assert(lc_target_expr_node_offset(2U) == offsetof(lc_target_expr_node, rhs));
    assert(lc_target_expr_node_offset(3U) == offsetof(lc_target_expr_node, binding));
    assert(lc_target_expr_node_offset(4U) == offsetof(lc_target_expr_node, value));
    assert(lc_target_expr_node_offset(5U) == UINT32_MAX);
}

static void test_rounded_operation_order(void) {
    const lc_target_expr_node nodes[] = {
        {LC_TARGET_CONST, 0U, 0U, 0U, 16777216.0},
        {LC_TARGET_CONST, 0U, 0U, 0U, 1.0},
        {LC_TARGET_ADD, 0U, 1U, 0U, 0.0},
        {LC_TARGET_SUB, 2U, 0U, 0U, 0.0},
        {LC_TARGET_DIV, 1U, 0U, 0U, 0.0},
        {LC_TARGET_NEG, 4U, 0U, 0U, 0.0},
        {LC_TARGET_MAX, 4U, 5U, 0U, 0.0}
    };
    double values[7];
    uint8_t kinds[7];
    uint32_t failed;
    uint32_t index;
    assert(lc_target_bind(nodes, 7U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 7U, &failed) == LC_TARGET_OK);
    assert(failed == UINT32_MAX);
    assert(values[2] == 16777216.0);
    assert(values[3] == 0.0);
    assert(values[4] == 0x1p-24);
    assert(values[5] == -0x1p-24);
    assert(values[6] == 0x1p-24);
    for (index = 0U; index < 7U; ++index) {
        assert(kinds[index] == LC_TARGET_BOUND);
    }
    assert(lc_target_bind(nodes, 7U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 7U, NULL) == LC_TARGET_OK);
    assert(values[2] == 16777217.0);
    assert(values[3] == 1.0);
}

static void test_parameter_binding_and_dynamic_closure(void) {
    const lc_target_expr_node nodes[] = {
        {LC_TARGET_PARAM, 0U, 0U, 0U, 0.0},
        {LC_TARGET_PARAM, 0U, 0U, 1U, 0.0},
        {LC_TARGET_ADD, 0U, 1U, 0U, 0.0},
        {LC_TARGET_VAR, 0U, 0U, 0U, 0.0},
        {LC_TARGET_EXP, 0U, 0U, 0U, 0.0},
        {LC_TARGET_MUL, 4U, 0U, 0U, 0.0},
        {LC_TARGET_ADD, 5U, 3U, 0U, 0.0},
        {LC_TARGET_POW, 0U, 0U, 0U, 0.0},
        {LC_TARGET_PHI1, 0U, 0U, 0U, 0.0}
    };
    const double parameters[] = {1.0 + 0x1p-24, 2.0};
    const uint8_t constant[] = {1U, 0U};
    double values[9];
    uint8_t kinds[9];
    assert(lc_target_bind(nodes, 9U, parameters, 2U, constant, 1U, 32U,
                          values, kinds, 9U, NULL) == LC_TARGET_OK);
    assert(values[0] == 1.0);
    assert(kinds[0] == LC_TARGET_BOUND);
    assert(kinds[1] == LC_TARGET_DYNAMIC);
    assert(kinds[2] == LC_TARGET_DYNAMIC);
    assert(kinds[3] == LC_TARGET_DYNAMIC);
    assert(kinds[4] == LC_TARGET_UNSUPPORTED);
    assert(kinds[5] == LC_TARGET_UNSUPPORTED);
    assert(kinds[6] == LC_TARGET_DYNAMIC);
    assert(kinds[7] == LC_TARGET_UNSUPPORTED);
    assert(kinds[8] == LC_TARGET_UNSUPPORTED);
    assert(values[1] == 0.0 && values[4] == 0.0);
}

static void test_signed_zero_and_subnormals(void) {
    const lc_target_expr_node nodes[] = {
        {LC_TARGET_CONST, 0U, 0U, 0U, -0.0},
        {LC_TARGET_NEG, 0U, 0U, 0U, 0.0},
        {LC_TARGET_CONST, 0U, 0U, 0U, 0x1p-149},
        {LC_TARGET_CONST, 0U, 0U, 0U, 2.0},
        {LC_TARGET_MUL, 2U, 3U, 0U, 0.0}
    };
    double values[5];
    uint8_t kinds[5];
    uint64_t wire_bits;
    assert(lc_target_bind(nodes, 5U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 5U, NULL) == LC_TARGET_OK);
    memcpy(&wire_bits, &values[0], sizeof(wire_bits));
    assert(wire_bits == UINT64_C(0x8000000000000000));
    assert(signbit(values[0]));
    assert(!signbit(values[1]));
    assert(values[2] == 0x1p-149);
    assert(values[4] == 0x1p-148);
}

static void test_arithmetic_errors(void) {
    lc_target_expr_node nodes[] = {
        {LC_TARGET_CONST, 0U, 0U, 0U, FLT_MAX},
        {LC_TARGET_CONST, 0U, 0U, 0U, 2.0},
        {LC_TARGET_MUL, 0U, 1U, 0U, 0.0},
        {LC_TARGET_DIV, 2U, 1U, 0U, 0.0}
    };
    double values[4];
    uint8_t kinds[4];
    uint32_t failed;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_NONFINITE);
    assert(failed == 2U);
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 4U, &failed) == LC_TARGET_OK);
    assert(values[3] == (double)FLT_MAX);
    nodes[0].value = DBL_MAX;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 4U, &failed) == LC_TARGET_NONFINITE);
    assert(failed == 2U);
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_NONFINITE);
    assert(failed == 0U);

    nodes[0].value = 0x1p-149;
    nodes[1].value = 0.5;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_UNDERFLOW);
    assert(failed == 2U);
    nodes[0].value = 0x1p-150;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_UNDERFLOW);
    assert(failed == 0U);
    nodes[0].value = 0x1p-149;
    nodes[1].value = 2.0;
    nodes[2].op = LC_TARGET_DIV;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_UNDERFLOW);
    assert(failed == 2U);
    nodes[0].value = 0x1p-1074;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 4U, &failed) == LC_TARGET_UNDERFLOW);
    assert(failed == 2U);
    nodes[0].value = 1.0;
    nodes[1].value = -0.0;
    nodes[2].op = LC_TARGET_DIV;
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 4U, &failed) == LC_TARGET_DIVISION_BY_ZERO);
    assert(failed == 2U);
    assert(lc_target_bind(nodes, 4U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 4U, &failed) == LC_TARGET_DIVISION_BY_ZERO);
}

static void test_validation_before_output_writes(void) {
    lc_target_expr_node nodes[] = {
        {LC_TARGET_CONST, 0U, 0U, 0U, 1.0},
        {LC_TARGET_ADD, 0U, 2U, 0U, 0.0}
    };
    double values[] = {123.0, 456.0};
    uint8_t kinds[] = {9U, 9U};
    double parameters[] = {1.0};
    uint8_t constant[] = {1U};
    uint32_t failed;
    assert(lc_target_bind(nodes, 2U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 2U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    assert(failed == 1U);
    assert(values[0] == 123.0 && values[1] == 456.0);
    assert(kinds[0] == 9U && kinds[1] == 9U);
    nodes[1].rhs = 0U;
    assert(lc_target_bind(nodes, 2U, NULL, 0U, NULL, 0U, 32U,
                          values, kinds, 1U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    assert(failed == UINT32_MAX);
    nodes[1].op = 99U;
    assert(lc_target_bind(nodes, 2U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 2U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    nodes[1].op = LC_TARGET_VAR;
    assert(lc_target_bind(nodes, 2U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 2U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    nodes[1].op = LC_TARGET_PARAM;
    nodes[1].binding = 1U;
    assert(lc_target_bind(nodes, 2U, parameters, 1U, constant, 0U, 64U,
                          values, kinds, 2U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    nodes[1].binding = 0U;
    constant[0] = 2U;
    assert(lc_target_bind(nodes, 2U, parameters, 1U, constant, 0U, 64U,
                          values, kinds, 2U, &failed) == LC_TARGET_INVALID_ARGUMENT);
    constant[0] = 1U;
    parameters[0] = NAN;
    assert(lc_target_bind(nodes, 2U, parameters, 1U, constant, 0U, 64U,
                          values, kinds, 2U, &failed) == LC_TARGET_NONFINITE);
    parameters[0] = 1.0;
    nodes[0].value = INFINITY;
    assert(lc_target_bind(nodes, 2U, parameters, 1U, constant, 0U, 32U,
                          values, kinds, 2U, &failed) == LC_TARGET_NONFINITE);
    assert(failed == 0U);
    assert(values[0] == 123.0 && values[1] == 456.0);
    assert(kinds[0] == 9U && kinds[1] == 9U);
    assert(lc_target_bind(NULL, 0U, NULL, 0U, NULL, 0U, 16U,
                          NULL, NULL, 0U, NULL) == LC_TARGET_INVALID_ARGUMENT);
    assert(lc_target_bind(NULL, 1U, NULL, 0U, NULL, 0U, 64U,
                          values, kinds, 2U, NULL) == LC_TARGET_INVALID_ARGUMENT);
    assert(lc_target_bind(NULL, 0U, NULL, 0U, NULL, 0U, 64U,
                          NULL, NULL, 0U, NULL) == LC_TARGET_OK);
}

static void test_rounding_mode_guard(void) {
    const lc_target_expr_node node = {LC_TARGET_CONST, 0U, 0U, 0U, 1.0};
    double value = 123.0;
    uint8_t kind = 9U;
    int previous = fegetround();
    assert(previous == FE_TONEAREST);
    if (fesetround(FE_DOWNWARD) == 0) {
        lc_target_status status = lc_target_bind(
            &node, 1U, NULL, 0U, NULL, 0U, 32U, &value, &kind, 1U, NULL
        );
        assert(fesetround(previous) == 0);
        assert(status == LC_TARGET_UNSUPPORTED_ENVIRONMENT);
        assert(value == 123.0 && kind == 9U);
    }
}

static void test_subnormal_mode_guard(void) {
    const lc_target_expr_node node = {LC_TARGET_CONST, 0U, 0U, 0U, 1.0};
    double value = 123.0;
    uint8_t kind = 9U;
    lc_target_status status;
#if defined(__SSE2__)
    unsigned int previous = _mm_getcsr();
    _mm_setcsr(previous | 0x8000U);
    status = lc_target_bind(
        &node, 1U, NULL, 0U, NULL, 0U, 32U, &value, &kind, 1U, NULL
    );
    _mm_setcsr(previous);
    assert(status == LC_TARGET_UNSUPPORTED_ENVIRONMENT);
    assert(value == 123.0 && kind == 9U);
    _mm_setcsr(previous | 0x0040U);
    status = lc_target_bind(
        &node, 1U, NULL, 0U, NULL, 0U, 32U, &value, &kind, 1U, NULL
    );
    _mm_setcsr(previous);
    assert(status == LC_TARGET_UNSUPPORTED_ENVIRONMENT);
    assert(value == 123.0 && kind == 9U);
#elif defined(__aarch64__)
    uint64_t previous;
    uint64_t flush;
    __asm__ volatile("mrs %0, fpcr" : "=r"(previous));
    flush = previous | (UINT64_C(1) << 24U);
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(flush));
    status = lc_target_bind(
        &node, 1U, NULL, 0U, NULL, 0U, 32U, &value, &kind, 1U, NULL
    );
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(previous));
    assert(status == LC_TARGET_UNSUPPORTED_ENVIRONMENT);
    assert(value == 123.0 && kind == 9U);
#else
    (void)node;
    (void)value;
    (void)kind;
    (void)status;
#endif
}

int main(void) {
    test_wire_queries();
    test_rounded_operation_order();
    test_parameter_binding_and_dynamic_closure();
    test_signed_zero_and_subnormals();
    test_arithmetic_errors();
    test_validation_before_output_writes();
    test_rounding_mode_guard();
    test_subnormal_mode_guard();
    return 0;
}
