#include "lacuna.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

/* This handshake has no precision-bearing parameters or return values. */
static void test_numeric_metadata(void) {
    uint32_t (*property)(uint32_t) = lc_numeric_property;
    lc_status (*check)(uint32_t, uint32_t, uint32_t, uint32_t) =
        lc_numeric_profile_check;
    uint32_t expected_profile =
        sizeof(lc_real_t) * CHAR_BIT == 64U &&
        sizeof(lc_time_t) * CHAR_BIT == 64U && FLT_RADIX == 2 &&
        DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021
        ? LC_NUMERIC_PROFILE_BINARY64 : LC_NUMERIC_PROFILE_UNKNOWN;

    assert(LC_ABI_VERSION == 17U);
    assert(LC_COMPILED_GRAPH_IMAGE_VERSION == 1U);
    assert(LC_NUMERIC_METADATA_VERSION == 1U);
    assert(LC_NUMERIC_ARITHMETIC_REVISION == 1U);
    assert(LC_NUMERIC_PROFILE_UNKNOWN == 0U);
    assert(LC_NUMERIC_PROFILE_BINARY64 == 1U);
    assert(LC_NUMERIC_PROFILE_BINARY32_TIME64 == 2U);
    assert(LC_NUMERIC_PROFILE_BINARY32 == 3U);
    assert(LC_NUMERIC_PROPERTY_VERSION == 1U);
    assert(LC_NUMERIC_PROPERTY_PROFILE == 2U);
    assert(LC_NUMERIC_PROPERTY_REAL_BITS == 3U);
    assert(LC_NUMERIC_PROPERTY_TIME_BITS == 4U);
    assert(LC_NUMERIC_PROPERTY_REAL_MANT_DIG == 5U);
    assert(LC_NUMERIC_PROPERTY_TIME_MANT_DIG == 6U);
    assert(LC_NUMERIC_PROPERTY_REAL_MAX_EXP == 7U);
    assert(LC_NUMERIC_PROPERTY_TIME_MAX_EXP == 8U);
    assert(LC_NUMERIC_PROPERTY_WIDE_BITS == 9U);
    assert(LC_NUMERIC_PROPERTY_WIDE_MANT_DIG == 10U);
    assert(LC_NUMERIC_PROPERTY_WIDE_MAX_EXP == 11U);
    assert(LC_NUMERIC_PROPERTY_RADIX == 12U);
    assert(LC_NUMERIC_PROPERTY_ARITHMETIC_REVISION == 13U);
    assert(LACUNA_REAL_BITS == 64);
    assert(LACUNA_TIME_BITS == 64);
    assert(property(LC_NUMERIC_PROPERTY_VERSION) == 1U);
    assert(property(LC_NUMERIC_PROPERTY_PROFILE) == expected_profile);
    assert(property(LC_NUMERIC_PROPERTY_REAL_BITS) == sizeof(lc_real_t) * CHAR_BIT);
    assert(property(LC_NUMERIC_PROPERTY_TIME_BITS) == sizeof(lc_time_t) * CHAR_BIT);
    assert(property(LC_NUMERIC_PROPERTY_REAL_MANT_DIG) == DBL_MANT_DIG);
    assert(property(LC_NUMERIC_PROPERTY_TIME_MANT_DIG) == DBL_MANT_DIG);
    assert(property(LC_NUMERIC_PROPERTY_REAL_MAX_EXP) == DBL_MAX_EXP);
    assert(property(LC_NUMERIC_PROPERTY_TIME_MAX_EXP) == DBL_MAX_EXP);
    assert(property(LC_NUMERIC_PROPERTY_WIDE_BITS) == sizeof(lc_wide_t) * CHAR_BIT);
    assert(property(LC_NUMERIC_PROPERTY_WIDE_MANT_DIG) == LDBL_MANT_DIG);
    assert(property(LC_NUMERIC_PROPERTY_WIDE_MAX_EXP) == LDBL_MAX_EXP);
    assert(property(LC_NUMERIC_PROPERTY_RADIX) == FLT_RADIX);
    assert(property(LC_NUMERIC_PROPERTY_ARITHMETIC_REVISION) == 1U);
    assert(property(0U) == 0U);
    assert(property(14U) == 0U);
    assert(property(UINT32_MAX) == 0U);
    assert(check(LC_NUMERIC_PROFILE_BINARY64, 64U, 64U, 1U) ==
           (expected_profile == LC_NUMERIC_PROFILE_BINARY64
                ? LC_OK : LC_INVALID_ARGUMENT));
    assert(check(LC_NUMERIC_PROFILE_UNKNOWN, 64U, 64U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY32_TIME64, 64U, 64U, 1U) ==
           LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY32, 64U, 64U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY32_TIME64, 32U, 64U, 1U) ==
           LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY32, 32U, 32U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(UINT32_MAX, 64U, 64U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY64, 32U, 64U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY64, 64U, 32U, 1U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY64, 64U, 64U, 0U) == LC_INVALID_ARGUMENT);
    assert(check(LC_NUMERIC_PROFILE_BINARY64, 64U, 64U, 2U) == LC_INVALID_ARGUMENT);
}

/* This slice must preserve the existing scalar and clock representation. */
static void test_default_types(void) {
    double value = -0.0;
    long double wide = 1.25L;
    lc_real_t *real_pointer = &value;
    lc_time_t *time_pointer = &value;
    lc_wide_t *wide_pointer = &wide;

    assert(real_pointer == &value);
    assert(time_pointer == &value);
    assert(wide_pointer == &wide);
    assert(sizeof(lc_real_t) == sizeof(double));
    assert(sizeof(lc_time_t) == sizeof(double));
    assert(sizeof(lc_wide_t) == sizeof(long double));
    assert(LC_REAL_EPSILON == DBL_EPSILON);
    assert(LC_REAL_MIN == DBL_MIN);
    assert(LC_REAL_MAX == DBL_MAX);
    assert(LC_TIME_EPSILON == DBL_EPSILON);
    assert(LC_TIME_MIN == DBL_MIN);
    assert(LC_TIME_MAX == DBL_MAX);
    assert(LC_WIDE_EPSILON == LDBL_EPSILON);
    assert(LC_WIDE_MIN == LDBL_MIN);
    assert(LC_WIDE_MAX == LDBL_MAX);
    assert(LC_REAL_C(1.25) == value + 1.25);
    assert(LC_TIME_C(1.25) == value + 1.25);
    assert(LC_WIDE_C(1.25) == wide);
}

static void assert_same_bits(double actual, double expected) {
    assert(memcmp(&actual, &expected, sizeof(actual)) == 0);
}

/* Alias calls must retain the original functions, including signed zero. */
static void test_math_aliases(void) {
    double (*real_unary[])(double) = {
        lc_real_exp, lc_real_expm1, lc_real_log, lc_real_log1p,
        lc_real_fabs, lc_real_sqrt, lc_real_sin, lc_real_cos,
        lc_real_tanh, lc_real_floor, lc_real_ceil
    };
    double (*reference_unary[])(double) = {
        exp, expm1, log, log1p, fabs, sqrt, sin, cos, tanh, floor, ceil
    };
    double (*real_binary[])(double, double) = {
        lc_real_pow, lc_real_fmin, lc_real_fmax,
        lc_real_nextafter, lc_real_copysign
    };
    double (*reference_binary[])(double, double) = {
        pow, fmin, fmax, nextafter, copysign
    };
    double values[] = {0.0, -0.0, 1.0e-12, 0.25, 1.0};
    unsigned function;
    unsigned index;

    for (function = 0; function < sizeof(real_unary) / sizeof(real_unary[0]);
         ++function) {
        assert(real_unary[function] == reference_unary[function]);
        for (index = 0; index < sizeof(values) / sizeof(values[0]); ++index) {
            assert_same_bits(
                real_unary[function](values[index]),
                reference_unary[function](values[index])
            );
        }
    }
    for (function = 0; function < sizeof(real_binary) / sizeof(real_binary[0]);
         ++function) {
        assert(real_binary[function] == reference_binary[function]);
        for (index = 0; index < sizeof(values) / sizeof(values[0]); ++index) {
            assert_same_bits(
                real_binary[function](values[index], 1.0),
                reference_binary[function](values[index], 1.0)
            );
        }
    }
    assert(lc_time_fabs == fabs);
    assert(lc_time_fmin == fmin);
    assert(lc_time_fmax == fmax);
    assert(lc_time_floor == floor);
    assert(lc_time_ceil == ceil);
    assert(lc_time_nextafter == nextafter);
    assert(lc_wide_exp == expl);
    assert(lc_wide_fabs == fabsl);
    assert(lc_wide_fmax == fmaxl);
    assert(lc_wide_pow == powl);
}

int main(void) {
    test_numeric_metadata();
    test_default_types();
    test_math_aliases();
    return 0;
}
