#include "lacuna_half.h"

#include <fenv.h>
#include <string.h>

typedef lc_half_t (*half_unary)(lc_half_t);

/* Integer-only test transport avoids a host FFI dependency on half arguments. */
void lc_half_test_row(uint32_t row, uint16_t *output) {
    static const half_unary functions[] = {
        lc_half_exp, lc_half_expm1, lc_half_log, lc_half_log1p,
        lc_half_sqrt, lc_half_sin, lc_half_cos, lc_half_tanh,
        lc_half_phi1, lc_half_phi1_deriv
    };
    static const uint16_t exponents[] = {
        0xbc00U, 0x4000U, 0x4200U, 0x4400U, 0x4500U, 0x4600U,
        0x4700U, 0x4800U, 0xc000U, 0xb266U, 0x3266U
    };
    uint32_t bits;
    for (bits = 0U; bits < 65536U; ++bits) {
        uint16_t payload = (uint16_t)bits;
        lc_half_t input;
        lc_half_t result;
        memcpy(&input, &payload, sizeof(input));
        if (row < 10U) result = functions[row](input);
        else if (row < 21U) {
            lc_half_t exponent;
            memcpy(&exponent, &exponents[row - 10U], sizeof(exponent));
            result = lc_half_pow(input, exponent);
        } else if (row == 21U) result = lc_half_floor(input);
        else if (row == 22U) result = lc_half_ceil(input);
        else if (row == 23U) result = lc_half_fabs(input);
        else {
            payload = 0x7e00U;
            memcpy(&result, &payload, sizeof(result));
        }
        memcpy(&output[bits], &result, sizeof(result));
    }
}

/* Mode changes belong to host tests, never to the execution runtime. */
int lc_half_test_environment(uint32_t mode) {
    static const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    int previous = fegetround();
    int result;
    if (mode >= 4U || previous == -1 || fesetround(modes[mode]) != 0) return -1;
    result = (int)lc_half_environment_valid();
    if (fesetround(previous) != 0) return -1;
    return result;
}

int lc_half_test_flush_subnormals(void) {
#if defined(__aarch64__)
    uint64_t previous;
    uint64_t changed;
    uint64_t observed;
    int result;
    __asm__ volatile("mrs %0, fpcr" : "=r"(previous));
    changed = previous | (UINT64_C(1) << 19U);
    __asm__ volatile("msr fpcr, %0" : : "r"(changed) : "memory");
    __asm__ volatile("mrs %0, fpcr" : "=r"(observed));
    result = (observed & (UINT64_C(1) << 19U)) != 0U
        ? (int)lc_half_environment_valid() : -1;
    __asm__ volatile("msr fpcr, %0" : : "r"(previous) : "memory");
    return result;
#else
    return -1;
#endif
}
