#include "lacuna_target.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

#ifdef __FAST_MATH__
#error "Target constant binding requires strict floating-point arithmetic"
#endif

uint32_t lc_target_abi_version(void) {
    return LC_TARGET_ABI_VERSION;
}

uint32_t lc_target_sizeof_expr_node(void) {
    return (uint32_t)sizeof(lc_target_expr_node);
}

uint32_t lc_target_expr_node_offset(uint32_t field) {
    switch (field) {
        case 0U:
            return (uint32_t)offsetof(lc_target_expr_node, op);
        case 1U:
            return (uint32_t)offsetof(lc_target_expr_node, lhs);
        case 2U:
            return (uint32_t)offsetof(lc_target_expr_node, rhs);
        case 3U:
            return (uint32_t)offsetof(lc_target_expr_node, binding);
        case 4U:
            return (uint32_t)offsetof(lc_target_expr_node, value);
        default:
            return UINT32_MAX;
    }
}

static int environment_supported(void) {
    volatile float f_min = FLT_MIN;
    volatile float f_two = 2.0f;
    volatile float f_half;
    volatile float f_small;
    volatile double d_min = DBL_MIN;
    volatile double d_two = 2.0;
    volatile double d_half;
    volatile double d_small;

    if (CHAR_BIT != 8 || sizeof(float) != 4U || sizeof(double) != 8U ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53 ||
        FLT_MAX_EXP != 128 || DBL_MAX_EXP != 1024 ||
        FLT_MIN_EXP != -125 || DBL_MIN_EXP != -1021 ||
        FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) {
        return 0;
    }

    /* Exercise arithmetic so flush-to-zero and denormal-input modes fail. */
    f_half = f_min / f_two;
    d_half = d_min / d_two;
    f_small = nextafterf(0.0f, 1.0f);
    d_small = nextafter(0.0, 1.0);
    if (!(f_half > 0.0f) || f_half * f_two != f_min ||
        !(d_half > 0.0) || d_half * d_two != d_min ||
        !(f_small > 0.0f) || !(f_small * f_two > f_small) ||
        !(d_small > 0.0) || !(d_small * d_two > d_small)) {
        return 0;
    }
    return 1;
}

static int arity(uint32_t op) {
    switch (op) {
        case LC_TARGET_CONST:
        case LC_TARGET_PARAM:
        case LC_TARGET_VAR:
            return 0;
        case LC_TARGET_NEG:
        case LC_TARGET_EXP:
        case LC_TARGET_LOG:
        case LC_TARGET_PHI1:
        case LC_TARGET_PHI1_DERIV:
        case LC_TARGET_SIN:
        case LC_TARGET_COS:
        case LC_TARGET_TANH:
            return 1;
        case LC_TARGET_ADD:
        case LC_TARGET_SUB:
        case LC_TARGET_MUL:
        case LC_TARGET_DIV:
        case LC_TARGET_POW:
        case LC_TARGET_MAX:
            return 2;
        default:
            return -1;
    }
}

static int primitive(uint32_t op) {
    return op <= LC_TARGET_DIV || op == LC_TARGET_MAX;
}

static uint8_t node_kind(
    const lc_target_expr_node *node,
    const uint8_t *parameter_constant,
    const uint8_t *kinds
) {
    int count = arity(node->op);
    if (node->op == LC_TARGET_VAR ||
        (node->op == LC_TARGET_PARAM &&
         parameter_constant[node->binding] == 0U)) {
        return LC_TARGET_DYNAMIC;
    }
    if ((count >= 1 && kinds[node->lhs] == LC_TARGET_DYNAMIC) ||
        (count == 2 && kinds[node->rhs] == LC_TARGET_DYNAMIC)) {
        return LC_TARGET_DYNAMIC;
    }
    if (!primitive(node->op) ||
        (count >= 1 && kinds[node->lhs] == LC_TARGET_UNSUPPORTED) ||
        (count == 2 && kinds[node->rhs] == LC_TARGET_UNSUPPORTED)) {
        return LC_TARGET_UNSUPPORTED;
    }
    return LC_TARGET_BOUND;
}

static lc_target_status round32(double value, float *result) {
    volatile float rounded;
    if (!isfinite(value)) {
        return LC_TARGET_NONFINITE;
    }
    /* Check the range before casting because out-of-range casts are not portable. */
    if (value > (double)FLT_MAX || value < -(double)FLT_MAX) {
        return LC_TARGET_NONFINITE;
    }
    rounded = (float)value;
    if (!isfinite(rounded)) {
        return LC_TARGET_NONFINITE;
    }
    if (value != 0.0 && rounded == 0.0f) {
        return LC_TARGET_UNDERFLOW;
    }
    *result = rounded;
    return LC_TARGET_OK;
}

static lc_target_status evaluate32(
    uint32_t op, float lhs, float rhs, float *result
) {
    volatile float a = lhs;
    volatile float b = rhs;
    volatile float value;
    switch (op) {
        case LC_TARGET_NEG:
            value = -a;
            break;
        case LC_TARGET_ADD:
            value = a + b;
            break;
        case LC_TARGET_SUB:
            value = a - b;
            break;
        case LC_TARGET_MUL:
            value = a * b;
            break;
        case LC_TARGET_DIV:
            if (b == 0.0f) {
                return LC_TARGET_DIVISION_BY_ZERO;
            }
            value = a / b;
            break;
        case LC_TARGET_MAX:
            value = fmaxf(a, b);
            break;
        default:
            return LC_TARGET_INVALID_ARGUMENT;
    }
    if (!isfinite(value)) {
        return LC_TARGET_NONFINITE;
    }
    if ((op == LC_TARGET_MUL || op == LC_TARGET_DIV) &&
        a != 0.0f && b != 0.0f && value == 0.0f) {
        return LC_TARGET_UNDERFLOW;
    }
    *result = value;
    return LC_TARGET_OK;
}

static lc_target_status evaluate64(
    uint32_t op, double lhs, double rhs, double *result
) {
    volatile double a = lhs;
    volatile double b = rhs;
    volatile double value;
    switch (op) {
        case LC_TARGET_NEG:
            value = -a;
            break;
        case LC_TARGET_ADD:
            value = a + b;
            break;
        case LC_TARGET_SUB:
            value = a - b;
            break;
        case LC_TARGET_MUL:
            value = a * b;
            break;
        case LC_TARGET_DIV:
            if (b == 0.0) {
                return LC_TARGET_DIVISION_BY_ZERO;
            }
            value = a / b;
            break;
        case LC_TARGET_MAX:
            value = fmax(a, b);
            break;
        default:
            return LC_TARGET_INVALID_ARGUMENT;
    }
    if (!isfinite(value)) {
        return LC_TARGET_NONFINITE;
    }
    if ((op == LC_TARGET_MUL || op == LC_TARGET_DIV) &&
        a != 0.0 && b != 0.0 && value == 0.0) {
        return LC_TARGET_UNDERFLOW;
    }
    *result = value;
    return LC_TARGET_OK;
}

static lc_target_status bind32(
    const lc_target_expr_node *nodes, uint32_t node_count,
    const double *parameters, const uint8_t *parameter_constant,
    double *values, uint8_t *kinds, uint32_t *failed_node
) {
    uint32_t index;
    for (index = 0U; index < node_count; ++index) {
        const lc_target_expr_node *node = &nodes[index];
        lc_target_status status = LC_TARGET_OK;
        float value = 0.0f;
        kinds[index] = node_kind(node, parameter_constant, kinds);
        if (kinds[index] == LC_TARGET_BOUND) {
            if (node->op == LC_TARGET_CONST) {
                status = round32(node->value, &value);
            } else if (node->op == LC_TARGET_PARAM) {
                status = round32(parameters[node->binding], &value);
            } else {
                float lhs = (float)values[node->lhs];
                float rhs = arity(node->op) == 2 ?
                    (float)values[node->rhs] : 0.0f;
                status = evaluate32(node->op, lhs, rhs, &value);
            }
        }
        if (status != LC_TARGET_OK) {
            *failed_node = index;
            return status;
        }
        values[index] = (double)value;
    }
    return LC_TARGET_OK;
}

static lc_target_status bind64(
    const lc_target_expr_node *nodes, uint32_t node_count,
    const double *parameters, const uint8_t *parameter_constant,
    double *values, uint8_t *kinds, uint32_t *failed_node
) {
    uint32_t index;
    for (index = 0U; index < node_count; ++index) {
        const lc_target_expr_node *node = &nodes[index];
        lc_target_status status = LC_TARGET_OK;
        double value = 0.0;
        kinds[index] = node_kind(node, parameter_constant, kinds);
        if (kinds[index] == LC_TARGET_BOUND) {
            if (node->op == LC_TARGET_CONST) {
                value = node->value;
            } else if (node->op == LC_TARGET_PARAM) {
                value = parameters[node->binding];
            } else {
                double rhs = arity(node->op) == 2 ? values[node->rhs] : 0.0;
                status = evaluate64(node->op, values[node->lhs], rhs, &value);
            }
        }
        if (status != LC_TARGET_OK) {
            *failed_node = index;
            return status;
        }
        values[index] = value;
    }
    return LC_TARGET_OK;
}

uint32_t lc_target_bind(
    const lc_target_expr_node *nodes,
    uint32_t node_count,
    const double *parameters,
    uint32_t parameter_count,
    const uint8_t *parameter_constant,
    uint32_t variable_count,
    uint32_t real_bits,
    double *out_values,
    uint8_t *out_kinds,
    uint32_t capacity,
    uint32_t *failed_node
) {
    uint32_t index;
    uint32_t local_failed_node = UINT32_MAX;
    if (failed_node == NULL) {
        failed_node = &local_failed_node;
    }
    *failed_node = UINT32_MAX;
    if ((real_bits != 32U && real_bits != 64U) || capacity < node_count ||
        (node_count > 0U &&
         (nodes == NULL || out_values == NULL || out_kinds == NULL)) ||
        (parameter_count > 0U &&
         (parameters == NULL || parameter_constant == NULL))) {
        return LC_TARGET_INVALID_ARGUMENT;
    }
    for (index = 0U; index < parameter_count; ++index) {
        if (parameter_constant[index] > 1U) {
            return LC_TARGET_INVALID_ARGUMENT;
        }
        if (!isfinite(parameters[index])) {
            return LC_TARGET_NONFINITE;
        }
    }
    for (index = 0U; index < node_count; ++index) {
        const lc_target_expr_node *node = &nodes[index];
        int count = arity(node->op);
        if (count < 0 || (count >= 1 && node->lhs >= index) ||
            (count == 2 && node->rhs >= index) ||
            (node->op == LC_TARGET_PARAM && node->binding >= parameter_count) ||
            (node->op == LC_TARGET_VAR && node->binding >= variable_count)) {
            *failed_node = index;
            return LC_TARGET_INVALID_ARGUMENT;
        }
        if (node->op == LC_TARGET_CONST && !isfinite(node->value)) {
            *failed_node = index;
            return LC_TARGET_NONFINITE;
        }
    }
    if (!environment_supported()) {
        return LC_TARGET_UNSUPPORTED_ENVIRONMENT;
    }
    if (real_bits == 32U) {
        return bind32(nodes, node_count, parameters, parameter_constant,
                      out_values, out_kinds, failed_node);
    }
    return bind64(nodes, node_count, parameters, parameter_constant,
                  out_values, out_kinds, failed_node);
}
