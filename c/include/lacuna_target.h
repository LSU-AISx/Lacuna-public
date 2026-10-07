#ifndef LACUNA_TARGET_H
#define LACUNA_TARGET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LC_TARGET_ABI_VERSION 1U

/* Host compiler transport, independent of the execution runtime ABI. */
typedef struct lc_target_expr_node {
    uint32_t op;
    uint32_t lhs;
    uint32_t rhs;
    uint32_t binding;
    double value;
} lc_target_expr_node;

typedef enum lc_target_status {
    LC_TARGET_OK = 0,
    LC_TARGET_INVALID_ARGUMENT = 1,
    LC_TARGET_UNSUPPORTED_ENVIRONMENT = 2,
    LC_TARGET_NONFINITE = 3,
    LC_TARGET_UNDERFLOW = 4,
    LC_TARGET_DIVISION_BY_ZERO = 5
} lc_target_status;

typedef enum lc_target_kind {
    LC_TARGET_DYNAMIC = 0,
    LC_TARGET_BOUND = 1,
    LC_TARGET_UNSUPPORTED = 2
} lc_target_kind;

/* Operation identifiers match the model-neutral expression DAG. */
typedef enum lc_target_expr_op {
    LC_TARGET_CONST = 0,
    LC_TARGET_PARAM = 1,
    LC_TARGET_VAR = 2,
    LC_TARGET_NEG = 3,
    LC_TARGET_ADD = 4,
    LC_TARGET_SUB = 5,
    LC_TARGET_MUL = 6,
    LC_TARGET_DIV = 7,
    LC_TARGET_POW = 8,
    LC_TARGET_EXP = 9,
    LC_TARGET_LOG = 10,
    LC_TARGET_PHI1 = 11,
    LC_TARGET_PHI1_DERIV = 12,
    LC_TARGET_SIN = 13,
    LC_TARGET_COS = 14,
    LC_TARGET_TANH = 15,
    LC_TARGET_MAX = 16
} lc_target_expr_op;

uint32_t lc_target_abi_version(void);
uint32_t lc_target_sizeof_expr_node(void);
/* Field order is op, lhs, rhs, binding, value. Unknown fields return UINT32_MAX. */
uint32_t lc_target_expr_node_offset(uint32_t field);

/*
 * Bind constant-only primitive subgraphs in binary32 or binary64 arithmetic.
 * Binary64 arrays are host transport, not a network conversion interface.
 * Transcendental functions remain unsupported and state-dependent nodes remain
 * dynamic. A parameter is constant only when its mask entry is one.
 *
 * All references must point backward. Parameter and output arrays must contain
 * the declared number of elements and must not overlap other argument arrays.
 * Malformed graphs and insufficient capacity are rejected before output-array writes.
 * Outputs are undefined after an arithmetic error. failed_node is optional and
 * receives UINT32_MAX when an error is not associated with an expression node.
 */
/* The fixed-width return carries an lc_target_status value. */
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
);

#ifdef __cplusplus
}
#endif

#endif
