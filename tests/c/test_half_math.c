#include "lacuna_half.h"

#include <assert.h>
#include <string.h>

static lc_half_t value(uint16_t bits) {
    lc_half_t result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t bits(lc_half_t input) {
    uint16_t result;
    memcpy(&result, &input, sizeof(result));
    return result;
}

static void test_known_values(void) {
    assert(bits(lc_half_exp(value(0x3c00U))) == 0x4170U);
    assert(bits(lc_half_expm1(value(1U))) == 1U);
    assert(bits(lc_half_log(value(0x3c00U))) == 0U);
    assert(bits(lc_half_log1p(value(1U))) == 1U);
    assert(bits(lc_half_sqrt(value(1U))) == 0x0c00U);
    assert(bits(lc_half_sin(value(0x3c00U))) == 0x3abbU);
    assert(bits(lc_half_cos(value(0x3c00U))) == 0x3853U);
    assert(bits(lc_half_tanh(value(0x3c00U))) == 0x3a18U);
    assert(bits(lc_half_phi1(value(0U))) == 0x3c00U);
    assert(bits(lc_half_phi1_deriv(value(0U))) == 0x3800U);
    assert(bits(lc_half_phi1(value(1U))) == 0x3c00U);
    assert(bits(lc_half_phi1_deriv(value(1U))) == 0x3800U);
}

static void test_special_values(void) {
    assert(bits(lc_half_exp(value(0xfc00U))) == 0U);
    assert(bits(lc_half_expm1(value(0xfc00U))) == 0xbc00U);
    assert(bits(lc_half_log(value(0U))) == 0xfc00U);
    assert(bits(lc_half_log1p(value(0xbc00U))) == 0xfc00U);
    assert(bits(lc_half_sqrt(value(0x8000U))) == 0x8000U);
    assert(bits(lc_half_sqrt(value(0xbc00U))) == 0x7e00U);
    assert(bits(lc_half_sin(value(0x8000U))) == 0x8000U);
    assert(bits(lc_half_sin(value(0x7c00U))) == 0x7e00U);
    assert(bits(lc_half_tanh(value(0xfc00U))) == 0xbc00U);
    assert(bits(lc_half_pow(value(0x7e00U), value(0U))) == 0x3c00U);
    assert(bits(lc_half_pow(value(0x8000U), value(0xbc00U))) == 0xfc00U);
    assert(bits(lc_half_pow(value(0x8000U), value(0x3800U))) == 0U);
    assert(bits(lc_half_pow(value(0xfc00U), value(0x3800U))) == 0x7c00U);
    assert(bits(lc_half_pow(value(0xc000U), value(0x4200U))) == 0xc800U);
    assert(bits(lc_half_pow(value(0x4000U), value(0x3555U))) == 0x7e00U);
    assert(bits(lc_half_pow(value(0x3c00U), value(0x3555U))) == 0x7e00U);
    assert(lc_half_pow_supported_bits(0x3555U) == 0U);
    assert(lc_half_pow_supported_bits(0xb266U) == 1U);
    assert(lc_half_pow_supported_bits(0x4800U) == 1U);
}

static void test_bit_operations(void) {
    uint32_t index;
    assert(bits(lc_half_floor(value(0xb800U))) == 0xbc00U);
    assert(bits(lc_half_ceil(value(0xb800U))) == 0x8000U);
    assert(bits(lc_half_floor(value(0x3800U))) == 0U);
    assert(bits(lc_half_ceil(value(0x3800U))) == 0x3c00U);
    assert(bits(lc_half_fmin(value(0U), value(0x8000U))) == 0x8000U);
    assert(bits(lc_half_fmax(value(0x8000U), value(0U))) == 0U);
    assert(bits(lc_half_fmin(value(0x7e00U), value(0x3c00U))) == 0x3c00U);
    assert(bits(lc_half_nextafter(value(0U), value(0x8000U))) == 0x8000U);
    assert(bits(lc_half_nextafter(value(0U), value(0xbc00U))) == 0x8001U);
    assert(bits(lc_half_nextafter(value(0x7c00U), value(0U))) == 0x7bffU);
    assert(bits(lc_half_nextafter(value(0xfc00U), value(0U))) == 0xfbffU);
    for (index = 1U; index < 0x7c00U; ++index) {
        uint16_t input = (uint16_t)index;
        assert(bits(lc_half_nextafter(value(input), value(0x7c00U))) == input + 1U);
        assert(bits(lc_half_nextafter(value(input), value(0U))) == input - 1U);
        assert(bits(lc_half_nextafter(value((uint16_t)(input | 0x8000U)),
                                    value(0xfc00U))) == (input | 0x8000U) + 1U);
        assert(bits(lc_half_copysign(value(input), value(0x8000U))) ==
               (input | 0x8000U));
    }
}

static void test_native_half_rounding(void) {
    volatile lc_half_t large = value(0x6800U);
    volatile lc_half_t one = value(0x3c00U);
    volatile lc_half_t cancellation = (large + one) - large;
    assert(bits(cancellation) == 0U);
}

static void test_power_tables_against_native_primitives(void) {
    volatile lc_half_t one = value(0x3c00U);
    uint32_t index;
    for (index = 0U; index < 65536U; ++index) {
        volatile lc_half_t input = value((uint16_t)index);
        volatile lc_half_t squared;
        volatile lc_half_t reciprocal;
        if ((index & 0x7fffU) > 0x7c00U) continue;
        squared = input * input;
        reciprocal = one / input;
        assert(bits(lc_half_pow(input, value(0x4000U))) == bits(squared));
        assert(bits(lc_half_pow(input, value(0xbc00U))) == bits(reciprocal));
    }
}

int main(void) {
    assert(lc_half_environment_valid() == 1U);
    test_known_values();
    test_special_values();
    test_bit_operations();
    test_native_half_rounding();
    test_power_tables_against_native_primitives();
    assert(lc_half_math_revision() == 1U);
    assert(lc_half_math_table_bytes() > 0U);
    return 0;
}
