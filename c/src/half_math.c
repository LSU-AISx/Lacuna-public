#include "lacuna_half.h"

#include <string.h>

#include "generated/half_math_tables.inc"

static uint16_t half_bits(lc_half_t value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static lc_half_t half_value(uint16_t bits) {
    lc_half_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t half_nan(uint16_t bits) {
    return (bits & 0x7fffU) > 0x7c00U;
}

static uint16_t half_order(uint16_t bits) {
    return (uint16_t)((bits & 0x8000U) != 0U ? ~bits : bits | 0x8000U);
}

static lc_half_t half_lookup(uint32_t row, uint16_t bits) {
    uint32_t block = lc_half_block_index[
        row * LC_HALF_TABLE_ROW_BLOCKS + (bits >> LC_HALF_TABLE_BLOCK_BITS)];
    return half_value(lc_half_blocks[
        (block << LC_HALF_TABLE_BLOCK_BITS) + (bits & LC_HALF_TABLE_BLOCK_MASK)]);
}

#define LC_HALF_UNARY(name, row) \
    lc_half_t lc_half_##name(lc_half_t value) { \
        return half_lookup(row, half_bits(value)); \
    }

LC_HALF_UNARY(exp, LC_HALF_ROW_EXP)
LC_HALF_UNARY(expm1, LC_HALF_ROW_EXPM1)
LC_HALF_UNARY(log, LC_HALF_ROW_LOG)
LC_HALF_UNARY(log1p, LC_HALF_ROW_LOG1P)
LC_HALF_UNARY(sqrt, LC_HALF_ROW_SQRT)
LC_HALF_UNARY(sin, LC_HALF_ROW_SIN)
LC_HALF_UNARY(cos, LC_HALF_ROW_COS)
LC_HALF_UNARY(tanh, LC_HALF_ROW_TANH)
LC_HALF_UNARY(phi1, LC_HALF_ROW_PHI1)
LC_HALF_UNARY(phi1_deriv, LC_HALF_ROW_PHI1_DERIV)

static uint32_t half_power_row(uint16_t exponent) {
    switch (exponent) {
        case 0xbc00U: return LC_HALF_ROW_POW_NEG1;
        case 0x4000U: return LC_HALF_ROW_POW_2;
        case 0x4200U: return LC_HALF_ROW_POW_3;
        case 0x4400U: return LC_HALF_ROW_POW_4;
        case 0x4500U: return LC_HALF_ROW_POW_5;
        case 0x4600U: return LC_HALF_ROW_POW_6;
        case 0x4700U: return LC_HALF_ROW_POW_7;
        case 0x4800U: return LC_HALF_ROW_POW_8;
        case 0xc000U: return LC_HALF_ROW_POW_NEG2;
        case 0xb266U: return LC_HALF_ROW_POW_NEG_POINT2;
        case 0x3266U: return LC_HALF_ROW_POW_POINT2;
        default: return UINT32_MAX;
    }
}

uint32_t lc_half_pow_supported_bits(uint16_t exponent) {
    return (exponent & 0x7fffU) == 0U || exponent == 0x3c00U ||
        exponent == 0x3800U || half_power_row(exponent) != UINT32_MAX;
}

lc_half_t lc_half_pow(lc_half_t base, lc_half_t exponent) {
    uint16_t x = half_bits(base);
    uint16_t y = half_bits(exponent);
    uint32_t row;
    if ((y & 0x7fffU) == 0U) return half_value(0x3c00U);
    if (half_nan(x) || half_nan(y)) return half_value(0x7e00U);
    if (y == 0x3c00U) return base;
    if (y == 0x3800U) {
        if ((x & 0x7fffU) == 0U) return half_value(0U);
        if ((x & 0x7fffU) == 0x7c00U) return half_value(0x7c00U);
        return half_lookup(LC_HALF_ROW_SQRT, x);
    }
    row = half_power_row(y);
    return row == UINT32_MAX ? half_value(0x7e00U) : half_lookup(row, x);
}

lc_half_t lc_half_fabs(lc_half_t value) {
    return half_value((uint16_t)(half_bits(value) & 0x7fffU));
}

lc_half_t lc_half_copysign(lc_half_t value, lc_half_t sign) {
    return half_value((uint16_t)((half_bits(value) & 0x7fffU) |
                                (half_bits(sign) & 0x8000U)));
}

lc_half_t lc_half_fmin(lc_half_t left, lc_half_t right) {
    uint16_t a = half_bits(left), b = half_bits(right);
    if (half_nan(a)) return half_nan(b) ? half_value(0x7e00U) : right;
    if (half_nan(b)) return left;
    return half_order(a) < half_order(b) ? left : right;
}

lc_half_t lc_half_fmax(lc_half_t left, lc_half_t right) {
    uint16_t a = half_bits(left), b = half_bits(right);
    if (half_nan(a)) return half_nan(b) ? half_value(0x7e00U) : right;
    if (half_nan(b)) return left;
    return half_order(a) > half_order(b) ? left : right;
}

static lc_half_t half_round_integer(lc_half_t value, uint32_t upward) {
    uint16_t bits = half_bits(value);
    uint16_t magnitude = (uint16_t)(bits & 0x7fffU);
    uint32_t negative = (bits & 0x8000U) != 0U;
    int exponent = (int)(magnitude >> 10U) - 15;
    uint16_t mask;
    if (half_nan(bits)) return half_value(0x7e00U);
    if (magnitude == 0U || exponent >= 10) return value;
    if (exponent < 0) {
        if (negative) return half_value(upward ? 0x8000U : 0xbc00U);
        return half_value(upward ? 0x3c00U : 0U);
    }
    mask = (uint16_t)((1U << (10 - exponent)) - 1U);
    if ((bits & mask) != 0U) {
        bits = (uint16_t)(bits & (uint16_t)~mask);
        if ((upward != 0U) != (negative != 0U))
            bits = (uint16_t)(bits + mask + 1U);
    }
    return half_value(bits);
}

lc_half_t lc_half_floor(lc_half_t value) {
    return half_round_integer(value, 0U);
}

lc_half_t lc_half_ceil(lc_half_t value) {
    return half_round_integer(value, 1U);
}

lc_half_t lc_half_nextafter(lc_half_t value, lc_half_t toward) {
    uint16_t a = half_bits(value), b = half_bits(toward);
    if (half_nan(a) || half_nan(b)) return half_value(0x7e00U);
    if (a == b || ((a | b) & 0x7fffU) == 0U) return toward;
    if ((a & 0x7fffU) == 0U) return half_value((uint16_t)((b & 0x8000U) | 1U));
    if ((half_order(a) < half_order(b)) != ((a & 0x8000U) != 0U)) a++;
    else a--;
    return half_value(a);
}

uint32_t lc_half_math_table_bytes(void) {
    return (uint32_t)(sizeof(lc_half_block_index) + sizeof(lc_half_blocks));
}

uint32_t lc_half_math_revision(void) {
    return 1U;
}

uint32_t lc_half_environment_valid(void) {
    volatile lc_half_t one = half_value(0x3c00U);
    volatile lc_half_t odd = half_value(0x3c01U);
    volatile lc_half_t negative = half_value(0xbc00U);
    volatile lc_half_t tie = half_value(0x1000U);
    volatile lc_half_t minimum = half_value(0x0400U);
    volatile lc_half_t subnormal = half_value(1U);
    volatile lc_half_t two = half_value(0x4000U);
    volatile lc_half_t even_result = one + tie;
    volatile lc_half_t odd_result = odd + tie;
    volatile lc_half_t negative_result = negative - tie;
    volatile lc_half_t subnormal_output = minimum / two;
    volatile lc_half_t subnormal_input = subnormal * two;
    return half_bits(even_result) == 0x3c00U &&
        half_bits(odd_result) == 0x3c02U &&
        half_bits(negative_result) == 0xbc00U &&
        half_bits(subnormal_output) == 0x0200U &&
        half_bits(subnormal_input) == 2U;
}
