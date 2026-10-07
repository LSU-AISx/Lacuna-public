#ifndef LACUNA_NUMERIC_H
#define LACUNA_NUMERIC_H

#include <float.h>
#include <math.h>

/* Model arithmetic and event-clock widths are independent build settings. */
#ifndef LACUNA_REAL_BITS
#define LACUNA_REAL_BITS 64
#endif
#ifndef LACUNA_TIME_BITS
#define LACUNA_TIME_BITS 64
#endif
#if (LACUNA_REAL_BITS != 16 && LACUNA_REAL_BITS != 32 && LACUNA_REAL_BITS != 64) || \
    (LACUNA_TIME_BITS != 16 && LACUNA_TIME_BITS != 32 && LACUNA_TIME_BITS != 64) || \
    (LACUNA_REAL_BITS == 64 && LACUNA_TIME_BITS != 64) || \
    ((LACUNA_REAL_BITS == 16 || LACUNA_TIME_BITS == 16) && \
     LACUNA_REAL_BITS != LACUNA_TIME_BITS)
#error "Supported numeric profiles are 64/64, 32/64, 32/32, and strict 16/16"
#endif
#ifndef LACUNA_ENABLE_PROFILING
#define LACUNA_ENABLE_PROFILING (LACUNA_TIME_BITS == 64)
#endif
#if LACUNA_TIME_BITS <= 32 && LACUNA_ENABLE_PROFILING
#error "Strict reduced-precision execution requires LACUNA_ENABLE_PROFILING=0"
#endif
#if LACUNA_REAL_BITS == 32 && FLT_EVAL_METHOD != 0
#error "Binary32 execution requires arithmetic without excess intermediate precision"
#endif
#if LACUNA_REAL_BITS <= 32 && defined(__FAST_MATH__)
#error "Reduced-precision execution requires strict floating-point compiler settings"
#endif

#define LC_NUMERIC_METADATA_VERSION 1U
#define LC_NUMERIC_ARITHMETIC_REVISION 1U

/* Some system classification macros widen float arguments before testing. */
#if LACUNA_REAL_BITS <= 32 && (defined(__GNUC__) || defined(__clang__))
#define lc_isfinite(value) __builtin_isfinite(value)
#else
#define lc_isfinite(value) isfinite(value)
#endif

/* Stable identifiers, not a list of currently implemented profiles. */
typedef enum lc_numeric_profile {
    LC_NUMERIC_PROFILE_UNKNOWN = 0,
    LC_NUMERIC_PROFILE_BINARY64 = 1,
    LC_NUMERIC_PROFILE_BINARY32_TIME64 = 2,
    LC_NUMERIC_PROFILE_BINARY32 = 3,
    LC_NUMERIC_PROFILE_BINARY16 = 4
} lc_numeric_profile;

/* Query keys for lc_numeric_property(). Unknown keys return zero. */
typedef enum lc_numeric_property_field {
    LC_NUMERIC_PROPERTY_VERSION = 1,
    LC_NUMERIC_PROPERTY_PROFILE = 2,
    LC_NUMERIC_PROPERTY_REAL_BITS = 3,
    LC_NUMERIC_PROPERTY_TIME_BITS = 4,
    LC_NUMERIC_PROPERTY_REAL_MANT_DIG = 5,
    LC_NUMERIC_PROPERTY_TIME_MANT_DIG = 6,
    LC_NUMERIC_PROPERTY_REAL_MAX_EXP = 7,
    LC_NUMERIC_PROPERTY_TIME_MAX_EXP = 8,
    LC_NUMERIC_PROPERTY_WIDE_BITS = 9,
    LC_NUMERIC_PROPERTY_WIDE_MANT_DIG = 10,
    LC_NUMERIC_PROPERTY_WIDE_MAX_EXP = 11,
    LC_NUMERIC_PROPERTY_RADIX = 12,
    LC_NUMERIC_PROPERTY_ARITHMETIC_REVISION = 13
} lc_numeric_property_field;

#if LACUNA_REAL_BITS == 64
typedef double lc_real_t;
#define LC_REAL_MANT_DIG DBL_MANT_DIG
#define LC_REAL_MAX_EXP DBL_MAX_EXP
#define LC_REAL_MIN_EXP DBL_MIN_EXP
#define LC_REAL_EPSILON DBL_EPSILON
#define LC_REAL_MIN DBL_MIN
#define LC_REAL_MAX DBL_MAX
#define LC_REAL_C(value) value
#define lc_real_exp exp
#define lc_real_expm1 expm1
#define lc_real_log log
#define lc_real_log1p log1p
#define lc_real_pow pow
#define lc_real_fabs fabs
#define lc_real_fmin fmin
#define lc_real_fmax fmax
#define lc_real_sqrt sqrt
#define lc_real_sin sin
#define lc_real_cos cos
#define lc_real_tanh tanh
#define lc_real_floor floor
#define lc_real_ceil ceil
#define lc_real_nextafter nextafter
#define lc_real_copysign copysign
#elif LACUNA_REAL_BITS == 32
typedef float lc_real_t;
#define LC_REAL_MANT_DIG FLT_MANT_DIG
#define LC_REAL_MAX_EXP FLT_MAX_EXP
#define LC_REAL_MIN_EXP FLT_MIN_EXP
#define LC_REAL_EPSILON FLT_EPSILON
#define LC_REAL_MIN FLT_MIN
#define LC_REAL_MAX FLT_MAX
#define LC_REAL_C(value) value##f
#define lc_real_exp expf
#define lc_real_expm1 expm1f
#define lc_real_log logf
#define lc_real_log1p log1pf
#define lc_real_pow powf
#define lc_real_fabs fabsf
#define lc_real_fmin fminf
#define lc_real_fmax fmaxf
#define lc_real_sqrt sqrtf
#define lc_real_sin sinf
#define lc_real_cos cosf
#define lc_real_tanh tanhf
#define lc_real_floor floorf
#define lc_real_ceil ceilf
#define lc_real_nextafter nextafterf
#define lc_real_copysign copysignf
#else
#include "lacuna_half.h"
typedef lc_half_t lc_real_t;
#define LC_REAL_MANT_DIG 11
#define LC_REAL_MAX_EXP 16
#define LC_REAL_MIN_EXP (-13)
#define LC_REAL_C(value) ((lc_half_t)(value))
#define LC_REAL_EPSILON LC_REAL_C(0x1p-10)
#define LC_REAL_MIN LC_REAL_C(0x1p-14)
#define LC_REAL_MAX LC_REAL_C(65504.0)
#define lc_real_exp lc_half_exp
#define lc_real_expm1 lc_half_expm1
#define lc_real_log lc_half_log
#define lc_real_log1p lc_half_log1p
#define lc_real_pow lc_half_pow
#define lc_real_fabs lc_half_fabs
#define lc_real_fmin lc_half_fmin
#define lc_real_fmax lc_half_fmax
#define lc_real_sqrt lc_half_sqrt
#define lc_real_sin lc_half_sin
#define lc_real_cos lc_half_cos
#define lc_real_tanh lc_half_tanh
#define lc_real_floor lc_half_floor
#define lc_real_ceil lc_half_ceil
#define lc_real_nextafter lc_half_nextafter
#define lc_real_copysign lc_half_copysign
#endif

/* Ratios are rounded constants, without overflowing their half numerators. */
#if LACUNA_REAL_BITS == 16
#define LC_REAL_RATIO(numerator, denominator) \
    ((lc_real_t)((numerator) / (denominator)))
#else
#define LC_REAL_RATIO(numerator, denominator) \
    (LC_REAL_C(numerator) / LC_REAL_C(denominator))
#endif

#if LACUNA_TIME_BITS == 64
typedef double lc_time_t;
typedef double lc_profile_t;
#define LC_TIME_MANT_DIG DBL_MANT_DIG
#define LC_TIME_MAX_EXP DBL_MAX_EXP
#define LC_TIME_EPSILON DBL_EPSILON
#define LC_TIME_MIN DBL_MIN
#define LC_TIME_MAX DBL_MAX
#define LC_TIME_C(value) value
#define lc_time_fabs fabs
#define lc_time_fmin fmin
#define lc_time_fmax fmax
#define lc_time_floor floor
#define lc_time_ceil ceil
#define lc_time_nextafter nextafter
#elif LACUNA_TIME_BITS == 32
typedef float lc_time_t;
typedef float lc_profile_t;
#define LC_TIME_MANT_DIG FLT_MANT_DIG
#define LC_TIME_MAX_EXP FLT_MAX_EXP
#define LC_TIME_EPSILON FLT_EPSILON
#define LC_TIME_MIN FLT_MIN
#define LC_TIME_MAX FLT_MAX
#define LC_TIME_C(value) value##f
#define lc_time_fabs fabsf
#define lc_time_fmin fminf
#define lc_time_fmax fmaxf
#define lc_time_floor floorf
#define lc_time_ceil ceilf
#define lc_time_nextafter nextafterf
#else
typedef lc_half_t lc_time_t;
typedef lc_half_t lc_profile_t;
#define LC_TIME_MANT_DIG LC_REAL_MANT_DIG
#define LC_TIME_MAX_EXP LC_REAL_MAX_EXP
#define LC_TIME_EPSILON LC_REAL_EPSILON
#define LC_TIME_MIN LC_REAL_MIN
#define LC_TIME_MAX LC_REAL_MAX
#define LC_TIME_C(value) LC_REAL_C(value)
#define lc_time_fabs lc_half_fabs
#define lc_time_fmin lc_half_fmin
#define lc_time_fmax lc_half_fmax
#define lc_time_floor lc_half_floor
#define lc_time_ceil lc_half_ceil
#define lc_time_nextafter lc_half_nextafter
#endif

/* Only the default profile retains extended guard intermediates. */
#if LACUNA_REAL_BITS == 64
typedef long double lc_wide_t;
#define LC_WIDE_MANT_DIG LDBL_MANT_DIG
#define LC_WIDE_MAX_EXP LDBL_MAX_EXP
#define LC_WIDE_EPSILON LDBL_EPSILON
#define LC_WIDE_MIN LDBL_MIN
#define LC_WIDE_MAX LDBL_MAX
#define LC_WIDE_C(value) value##L
#define lc_wide_exp expl
#define lc_wide_fabs fabsl
#define lc_wide_fmax fmaxl
#define lc_wide_pow powl
#elif LACUNA_TIME_BITS == 64
typedef double lc_wide_t;
#define LC_WIDE_MANT_DIG DBL_MANT_DIG
#define LC_WIDE_MAX_EXP DBL_MAX_EXP
#define LC_WIDE_EPSILON DBL_EPSILON
#define LC_WIDE_MIN DBL_MIN
#define LC_WIDE_MAX DBL_MAX
#define LC_WIDE_C(value) value
#define lc_wide_exp exp
#define lc_wide_fabs fabs
#define lc_wide_fmax fmax
#define lc_wide_pow pow
#elif LACUNA_REAL_BITS == 32
typedef float lc_wide_t;
#define LC_WIDE_MANT_DIG FLT_MANT_DIG
#define LC_WIDE_MAX_EXP FLT_MAX_EXP
#define LC_WIDE_EPSILON FLT_EPSILON
#define LC_WIDE_MIN FLT_MIN
#define LC_WIDE_MAX FLT_MAX
#define LC_WIDE_C(value) value##f
#define lc_wide_exp expf
#define lc_wide_fabs fabsf
#define lc_wide_fmax fmaxf
#define lc_wide_pow powf
#else
typedef lc_half_t lc_wide_t;
#define LC_WIDE_MANT_DIG LC_REAL_MANT_DIG
#define LC_WIDE_MAX_EXP LC_REAL_MAX_EXP
#define LC_WIDE_EPSILON LC_REAL_EPSILON
#define LC_WIDE_MIN LC_REAL_MIN
#define LC_WIDE_MAX LC_REAL_MAX
#define LC_WIDE_C(value) LC_REAL_C(value)
#define lc_wide_exp lc_half_exp
#define lc_wide_fabs lc_half_fabs
#define lc_wide_fmax lc_half_fmax
#define lc_wide_pow lc_half_pow
#endif

#endif
