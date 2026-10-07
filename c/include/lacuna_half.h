#ifndef LACUNA_HALF_H
#define LACUNA_HALF_H

#include <stdint.h>

#if !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && !defined(__AVX512FP16__)
#error "Strict binary16 requires native half arithmetic and a final-code audit"
#endif

#if !defined(__FLT16_MANT_DIG__) || __FLT16_MANT_DIG__ != 11 || \
    !defined(__FLT16_MAX_EXP__) || __FLT16_MAX_EXP__ != 16
#error "Strict binary16 requires an IEEE binary16 arithmetic type"
#endif

__extension__ typedef _Float16 lc_half_t;
typedef char lc_half_storage_is_two_bytes[(sizeof(lc_half_t) == 2U) ? 1 : -1];

#ifdef __cplusplus
extern "C" {
#endif

/* Table inputs and outputs are IEEE binary16 with round-to-nearest, ties-to-even. */
lc_half_t lc_half_exp(lc_half_t value);
lc_half_t lc_half_expm1(lc_half_t value);
lc_half_t lc_half_log(lc_half_t value);
lc_half_t lc_half_log1p(lc_half_t value);
lc_half_t lc_half_sqrt(lc_half_t value);
lc_half_t lc_half_sin(lc_half_t value);
lc_half_t lc_half_cos(lc_half_t value);
lc_half_t lc_half_tanh(lc_half_t value);
lc_half_t lc_half_phi1(lc_half_t value);
lc_half_t lc_half_phi1_deriv(lc_half_t value);

/* Unsupported finite exponents return a quiet NaN instead of an approximation. */
lc_half_t lc_half_pow(lc_half_t base, lc_half_t exponent);
uint32_t lc_half_pow_supported_bits(uint16_t exponent);

lc_half_t lc_half_fabs(lc_half_t value);
lc_half_t lc_half_fmin(lc_half_t left, lc_half_t right);
lc_half_t lc_half_fmax(lc_half_t left, lc_half_t right);
lc_half_t lc_half_floor(lc_half_t value);
lc_half_t lc_half_ceil(lc_half_t value);
lc_half_t lc_half_nextafter(lc_half_t value, lc_half_t toward);
lc_half_t lc_half_copysign(lc_half_t value, lc_half_t sign);

uint32_t lc_half_math_table_bytes(void);
uint32_t lc_half_math_revision(void);
/* Check the current rounding and subnormal modes without changing them. */
uint32_t lc_half_environment_valid(void);

#ifdef __cplusplus
}
#endif

#endif
