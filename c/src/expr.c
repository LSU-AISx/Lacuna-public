#include "lacuna.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Evaluate compiled equations and locate threshold crossings in continuous time. */

static int lc_unary_ref_valid(const lc_expr_node *node, uint32_t index) {
    return node->lhs < index;
}

static int lc_binary_refs_valid(const lc_expr_node *node, uint32_t index) {
    return node->lhs < index && node->rhs < index;
}

static lc_real_t lc_phi1(lc_real_t value) {
#if LACUNA_REAL_BITS == 16
    return lc_half_phi1(value);
#else
    /* The series avoids cancellation near zero. */
    if (lc_real_fabs(value) < LC_REAL_C(1e-4)) {
        return LC_REAL_C(1.0) + value * (LC_REAL_C(0.5) + value * (LC_REAL_RATIO(1.0, 6.0) + value * (
            LC_REAL_RATIO(1.0, 24.0) + value * (LC_REAL_RATIO(1.0, 120.0) + value * (LC_REAL_RATIO(1.0, 720.0) +
            value / LC_REAL_C(5040.0)))
        )));
    }
    return lc_real_expm1(value) / value;
#endif
}

static lc_real_t lc_phi1_derivative(lc_real_t value) {
#if LACUNA_REAL_BITS == 16
    return lc_half_phi1_deriv(value);
#else
    /* A wider series interval avoids binary32 cancellation in the numerator. */
    if (lc_real_fabs(value) <
#if LACUNA_REAL_BITS <= 32
        LC_REAL_C(0.125)
#else
        LC_REAL_C(1e-4)
#endif
    ) {
        return LC_REAL_C(0.5) + value * (LC_REAL_RATIO(1.0, 3.0) + value * (LC_REAL_RATIO(1.0, 8.0) + value * (
            LC_REAL_RATIO(1.0, 30.0) + value * (LC_REAL_RATIO(1.0, 144.0) + value * (LC_REAL_RATIO(1.0, 840.0) +
            value / LC_REAL_C(5760.0)))
        )));
    }
    return (value * lc_real_exp(value) - lc_real_expm1(value)) / (value * value);
#endif
}

/* Evaluate a full DAG or a marked dependency closure in node order. */
static lc_status lc_expr_evaluate_active(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    const uint8_t *active
) {
    uint32_t index;
    if (nodes == NULL || workspace == NULL || node_count == 0 || workspace_count < node_count ||
        (parameter_count > 0 && parameters == NULL) ||
        (variable_count > 0 && variables == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0; index < node_count; ++index) {
        const lc_expr_node *node = &nodes[index];
        lc_real_t value = LC_REAL_C(0.0);
        if (active != NULL && active[index] == 0U) {
            continue;
        }
        switch ((lc_expr_op)node->op) {
            case LC_EXPR_CONST:
                value = node->value;
                break;
            case LC_EXPR_PARAM:
                if (node->binding >= parameter_count) {
                    return LC_INVALID_ARGUMENT;
                }
                value = parameters[node->binding];
                break;
            case LC_EXPR_VAR:
                if (node->binding >= variable_count) {
                    return LC_INVALID_ARGUMENT;
                }
                value = variables[node->binding];
                break;
            case LC_EXPR_NEG:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = -workspace[node->lhs];
                break;
            case LC_EXPR_ADD:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = workspace[node->lhs] + workspace[node->rhs];
                break;
            case LC_EXPR_SUB:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = workspace[node->lhs] - workspace[node->rhs];
                break;
            case LC_EXPR_MUL:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = workspace[node->lhs] * workspace[node->rhs];
                break;
            case LC_EXPR_DIV:
                if (!lc_binary_refs_valid(node, index) || workspace[node->rhs] == LC_REAL_C(0.0)) {
                    return LC_NUMERIC_ERROR;
                }
                value = workspace[node->lhs] / workspace[node->rhs];
                break;
            case LC_EXPR_POW:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_pow(workspace[node->lhs], workspace[node->rhs]);
                break;
            case LC_EXPR_EXP:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_exp(workspace[node->lhs]);
                break;
            case LC_EXPR_LOG:
                if (!lc_unary_ref_valid(node, index) || workspace[node->lhs] <= LC_REAL_C(0.0)) {
                    return LC_NUMERIC_ERROR;
                }
                value = lc_real_log(workspace[node->lhs]);
                break;
            case LC_EXPR_PHI1:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_phi1(workspace[node->lhs]);
                break;
            case LC_EXPR_PHI1_DERIV:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_phi1_derivative(workspace[node->lhs]);
                break;
            case LC_EXPR_SIN:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_sin(workspace[node->lhs]);
                break;
            case LC_EXPR_COS:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_cos(workspace[node->lhs]);
                break;
            case LC_EXPR_TANH:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_tanh(workspace[node->lhs]);
                break;
            case LC_EXPR_MAX:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                value = lc_real_fmax(workspace[node->lhs], workspace[node->rhs]);
                break;
            default:
                return LC_INVALID_ARGUMENT;
        }
        if (!lc_isfinite(value)) {
            return LC_NUMERIC_ERROR;
        }
        workspace[index] = value;
    }
    return LC_OK;
}

/* Evaluate every node into caller-owned workspace. */
lc_status lc_expr_evaluate(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_expr_evaluate_active(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count, NULL
    );
}

/* Evaluate named roots after deriving their transitive dependencies. */
lc_status lc_expr_evaluate_selected(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_real_t *variables,
    uint32_t variable_count,
    const uint32_t *roots,
    uint32_t root_count,
    lc_real_t *outputs,
    lc_real_t *workspace,
    uint32_t workspace_count,
    uint8_t *active,
    uint32_t active_count
) {
    uint32_t cursor;
    lc_status status;
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (nodes == NULL || roots == NULL || outputs == NULL || workspace == NULL ||
        active == NULL || node_count == 0U || root_count == 0U ||
        workspace_count < node_count || active_count < node_count ||
        (parameter_count > 0U && parameters == NULL) ||
        (variable_count > 0U && variables == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    memset(active, 0, node_count * sizeof(*active));
    /* Walk backward from each root to mark its dependency closure. */
    for (cursor = 0; cursor < root_count; ++cursor) {
        if (roots[cursor] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
        active[roots[cursor]] = 1U;
    }
    for (cursor = node_count; cursor > 0U; --cursor) {
        uint32_t index = cursor - 1U;
        const lc_expr_node *node;
        if (active[index] == 0U) {
            continue;
        }
        node = &nodes[index];
        switch ((lc_expr_op)node->op) {
            case LC_EXPR_CONST:
            case LC_EXPR_PARAM:
            case LC_EXPR_VAR:
                break;
            case LC_EXPR_NEG:
            case LC_EXPR_EXP:
            case LC_EXPR_LOG:
            case LC_EXPR_PHI1:
            case LC_EXPR_PHI1_DERIV:
            case LC_EXPR_SIN:
            case LC_EXPR_COS:
            case LC_EXPR_TANH:
                if (!lc_unary_ref_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                active[node->lhs] = 1U;
                break;
            case LC_EXPR_ADD:
            case LC_EXPR_SUB:
            case LC_EXPR_MUL:
            case LC_EXPR_DIV:
            case LC_EXPR_POW:
            case LC_EXPR_MAX:
                if (!lc_binary_refs_valid(node, index)) {
                    return LC_INVALID_ARGUMENT;
                }
                active[node->lhs] = 1U;
                active[node->rhs] = 1U;
                break;
            default:
                return LC_INVALID_ARGUMENT;
        }
    }
    status = lc_expr_evaluate_active(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count, active
    );
    if (status != LC_OK) {
        return status;
    }
    for (cursor = 0; cursor < root_count; ++cursor) {
        outputs[cursor] = workspace[roots[cursor]];
    }
    return LC_OK;
}

/* Advance analytical state by evaluating all propagation roots together. */
lc_status lc_expr_state_advance(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *roots,
    uint32_t state_count,
    lc_real_t *state,
    lc_time_t *t_last,
    lc_time_t t,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t index;
    lc_status status;
    lc_time_t delta;
    if (nodes == NULL || roots == NULL || state == NULL || t_last == NULL ||
        variables == NULL || workspace == NULL || state_count == 0 ||
        !lc_isfinite(*t_last) || !lc_isfinite(t)) {
        return LC_INVALID_ARGUMENT;
    }
    if (state_count == UINT32_MAX || variable_count != state_count + 1U) {
        return LC_INVALID_ARGUMENT;
    }
    if (t < *t_last) {
        return LC_TIME_REVERSED;
    }
    for (index = 0; index < state_count; ++index) {
        if (roots[index] >= node_count || !lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    if (t == *t_last) {
        return LC_OK;
    }
    delta = t - *t_last;
    if (!lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    variables[0] = (lc_real_t)delta;
    for (index = 0; index < state_count; ++index) {
        variables[index + 1U] = state[index];
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    for (index = 0; index < state_count; ++index) {
        state[index] = workspace[roots[index]];
    }
    *t_last = t;
    return LC_OK;
}

/* Apply a simultaneous reset or clamp map from the pre-map state. */
lc_status lc_expr_state_map(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *roots,
    uint32_t state_count,
    lc_real_t *state,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t index;
    lc_status status;
    if (nodes == NULL || roots == NULL || state == NULL || variables == NULL ||
        workspace == NULL || state_count == 0U || state_count > LC_ANALYTICAL_MAX_STATES ||
        variable_count != state_count + 1U) {
        return LC_INVALID_ARGUMENT;
    }
    variables[0] = LC_REAL_C(0.0);
    for (index = 0; index < state_count; ++index) {
        if (roots[index] >= node_count || !lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
        variables[index + 1U] = state[index];
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    for (index = 0; index < state_count; ++index) {
        lc_real_t next = workspace[roots[index]];
        if (!lc_isfinite(next)) {
            return LC_NUMERIC_ERROR;
        }
        state[index] = next;
    }
    return LC_OK;
}

/* Evaluate a weight-dependent deposit and add it to one state component. */
lc_status lc_expr_state_deposit(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    uint32_t root,
    lc_real_t weight,
    lc_real_t *state,
    uint32_t state_count,
    uint32_t target,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t index;
    lc_real_t next;
    lc_status status;
    if (nodes == NULL || state == NULL || variables == NULL || workspace == NULL ||
        state_count == 0 || target >= state_count || root >= node_count ||
        variable_count != 1U || !lc_isfinite(weight)) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    variables[0] = weight;
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    next = state[target] + workspace[root];
    if (!lc_isfinite(next)) {
        return LC_NUMERIC_ERROR;
    }
    state[target] = next;
    return LC_OK;
}

static int lc_step_config_valid(const lc_step_config *config) {
    return config != NULL && lc_isfinite(config->relative_tolerance) &&
        config->relative_tolerance > LC_REAL_C(0.0) &&
        lc_isfinite(config->absolute_tolerance) && config->absolute_tolerance > LC_REAL_C(0.0) &&
        lc_isfinite(config->initial_step) && config->initial_step > LC_REAL_C(0.0) &&
        lc_isfinite(config->minimum_step) && config->minimum_step > LC_REAL_C(0.0) &&
        lc_isfinite(config->maximum_step) &&
        config->maximum_step >= config->minimum_step &&
        lc_isfinite(config->event_tolerance) && config->event_tolerance > LC_REAL_C(0.0) &&
        config->maximum_steps > 0U && config->maximum_rhs_evaluations > 0U;
}

/* The stepped path uses an adaptive Dormand-Prince trajectory. */

static lc_status lc_step_rhs(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *rhs_roots,
    uint32_t state_count,
    uint32_t readout,
    lc_time_t t,
    const lc_real_t *state,
    uint32_t clamped,
    lc_real_t *derivative,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t index;
    lc_status status;
    if (nodes == NULL || rhs_roots == NULL || state == NULL || derivative == NULL ||
        variables == NULL || workspace == NULL || state_count == 0U ||
        state_count > LC_ANALYTICAL_MAX_STATES || readout >= state_count ||
        variable_count != state_count + 1U || !lc_isfinite(t) || clamped > 1U) {
        return LC_INVALID_ARGUMENT;
    }
    variables[0] = (lc_real_t)t;
#if LACUNA_REAL_BITS != LACUNA_TIME_BITS
    if (!lc_isfinite(variables[0]) ||
        (t != LC_TIME_C(0.0) && variables[0] == LC_REAL_C(0.0))) {
        for (index = 0U; index < node_count; ++index) {
            if (nodes[index].op == LC_EXPR_VAR && nodes[index].binding == 0U) {
                return LC_NUMERIC_ERROR;
            }
        }
        variables[0] = LC_REAL_C(0.0);
    }
#endif
    for (index = 0U; index < state_count; ++index) {
        if (rhs_roots[index] >= node_count || !lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
        variables[index + 1U] = state[index];
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    for (index = 0U; index < state_count; ++index) {
        derivative[index] = clamped != 0U && index == readout
            ? LC_REAL_C(0.0)
            : workspace[rhs_roots[index]];
        if (!lc_isfinite(derivative[index])) {
            return LC_NUMERIC_ERROR;
        }
    }
    return LC_OK;
}

static lc_real_t lc_step_polynomial_value(const lc_real_t coefficients[5], lc_real_t theta) {
    return coefficients[0] + theta * (coefficients[1] + theta * (
        coefficients[2] + theta * (coefficients[3] + theta * coefficients[4])
    ));
}

/* Evaluate the derivative of the dense fourth-degree interpolation polynomial. */
static lc_real_t lc_step_polynomial_derivative(
    const lc_real_t coefficients[5], lc_real_t theta
) {
    return coefficients[1] + theta * (LC_REAL_C(2.0) * coefficients[2] + theta * (
        LC_REAL_C(3.0) * coefficients[3] + theta * LC_REAL_C(4.0) * coefficients[4]
    ));
}

static void lc_step_sort(lc_real_t *values, uint32_t count) {
    uint32_t index;
    for (index = 0U; index < count; ++index) {
        uint32_t cursor;
        for (cursor = index + 1U; cursor < count; ++cursor) {
            if (values[cursor] < values[index]) {
                lc_real_t swap = values[index];
                values[index] = values[cursor];
                values[cursor] = swap;
            }
        }
    }
}

/* Add a distinct interpolation root that lies inside the unit step. */
static void lc_step_add_unit_root(
    lc_real_t *values, uint32_t *count, uint32_t capacity, lc_real_t root
) {
    uint32_t index;
    if (!(root > LC_REAL_C(0.0) && root < LC_REAL_C(1.0)) || *count >= capacity) {
        return;
    }
    for (index = 0U; index < *count; ++index) {
        if (
#if LACUNA_REAL_BITS <= 32
            values[index] == root
#else
            lc_real_fabs(values[index] - root) <= LC_REAL_C(64.0) * LC_REAL_EPSILON
#endif
        ) {
            return;
        }
    }
    values[(*count)++] = root;
}

/* Find extrema of the dense output polynomial within one accepted step. */
static uint32_t lc_step_dense_extrema(
    const lc_real_t coefficients[5],
    lc_real_t extrema[3],
    uint32_t *iterations
) {
    lc_real_t q[4] = {
        coefficients[1], LC_REAL_C(2.0) * coefficients[2],
        LC_REAL_C(3.0) * coefficients[3], LC_REAL_C(4.0) * coefficients[4]
    };
    lc_real_t partitions[4] = {LC_REAL_C(0.0), LC_REAL_C(1.0), LC_REAL_C(0.0), LC_REAL_C(0.0)};
    uint32_t partition_count = 2U;
    uint32_t extremum_count = 0U;
    uint32_t index;
    lc_real_t qa = LC_REAL_C(3.0) * q[3];
    lc_real_t qb = LC_REAL_C(2.0) * q[2];
    lc_real_t qc = q[1];
    lc_real_t scale = lc_real_fmax(LC_REAL_C(1.0), lc_real_fmax(lc_real_fabs(qa), lc_real_fmax(lc_real_fabs(qb), lc_real_fabs(qc))));
    lc_real_t floor =
#if LACUNA_REAL_BITS <= 32
        LC_REAL_C(0.0);
#else
        LC_REAL_C(64.0) * LC_REAL_EPSILON * scale;
#endif
#if LACUNA_REAL_BITS <= 32
    if (!lc_isfinite(scale)) {
        return UINT32_MAX;
    }
#endif

    if (lc_real_fabs(qa) <= floor) {
        if (lc_real_fabs(qb) > floor) {
            lc_step_add_unit_root(partitions, &partition_count, 4U, -qc / qb);
        }
    } else {
        lc_real_t discriminant = qb * qb - LC_REAL_C(4.0) * qa * qc;
#if LACUNA_REAL_BITS <= 32
        if (!lc_isfinite(discriminant)) {
            return UINT32_MAX;
        }
#endif
        if (discriminant >= LC_REAL_C(0.0)) {
            lc_real_t square_root = lc_real_sqrt(discriminant);
#if LACUNA_REAL_BITS <= 32
            lc_real_t stable = -LC_REAL_C(0.5) * (
                qb + lc_real_copysign(square_root, qb)
            );
            if (stable == LC_REAL_C(0.0)) {
                lc_step_add_unit_root(
                    partitions, &partition_count, 4U,
                    -qb / (LC_REAL_C(2.0) * qa)
                );
            } else {
                lc_step_add_unit_root(partitions, &partition_count, 4U, stable / qa);
                lc_step_add_unit_root(partitions, &partition_count, 4U, qc / stable);
            }
#else
            lc_step_add_unit_root(
                partitions, &partition_count, 4U,
                (-qb - square_root) / (LC_REAL_C(2.0) * qa)
            );
            lc_step_add_unit_root(
                partitions, &partition_count, 4U,
                (-qb + square_root) / (LC_REAL_C(2.0) * qa)
            );
#endif
        }
    }
    lc_step_sort(partitions, partition_count);
    for (index = 0U; index + 1U < partition_count; ++index) {
        lc_real_t low = partitions[index];
        lc_real_t high = partitions[index + 1U];
        lc_real_t f_low = q[0] + low * (q[1] + low * (q[2] + low * q[3]));
        lc_real_t f_high = q[0] + high * (q[1] + high * (q[2] + high * q[3]));
        uint32_t local_iterations = 0U;
        if (f_low == LC_REAL_C(0.0)) {
            lc_step_add_unit_root(extrema, &extremum_count, 3U, low);
        }
        if (f_high == LC_REAL_C(0.0)) {
            lc_step_add_unit_root(extrema, &extremum_count, 3U, high);
        }
        if ((f_low < LC_REAL_C(0.0) && f_high > LC_REAL_C(0.0)) ||
            (f_low > LC_REAL_C(0.0) && f_high < LC_REAL_C(0.0))) {
            while (
#if LACUNA_REAL_BITS <= 32
                   lc_real_nextafter(low, high) < high &&
#else
                   high - low > LC_REAL_C(64.0) * LC_REAL_EPSILON &&
#endif
                   local_iterations < LC_ROOT_MAX_ITERATIONS) {
                lc_real_t midpoint = low + LC_REAL_C(0.5) * (high - low);
                lc_real_t f_midpoint;
                if (midpoint == low || midpoint == high) {
                    break;
                }
                f_midpoint = q[0] + midpoint * (
                    q[1] + midpoint * (q[2] + midpoint * q[3])
                );
                local_iterations++;
                if ((f_low < LC_REAL_C(0.0) && f_midpoint < LC_REAL_C(0.0)) ||
                    (f_low > LC_REAL_C(0.0) && f_midpoint > LC_REAL_C(0.0))) {
                    low = midpoint;
                    f_low = f_midpoint;
                } else {
                    high = midpoint;
                }
            }
            lc_step_add_unit_root(
                extrema, &extremum_count, 3U, low + LC_REAL_C(0.5) * (high - low)
            );
        }
        *iterations += local_iterations;
    }
    lc_step_sort(extrema, extremum_count);
    return extremum_count;
}

/* Isolate the first rising crossing across monotone dense-output intervals. */
static int lc_step_find_rising_crossing(
    lc_real_t y0,
    lc_time_t h,
    const lc_real_t k[7],
    lc_real_t threshold,
    lc_time_t time_tolerance,
    lc_real_t *theta,
    uint32_t *iterations
) {
    lc_real_t model_h = (lc_real_t)h;
    lc_real_t coefficients[5];
    lc_real_t extrema[3] = {LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(0.0)};
    lc_real_t partition[5] = {LC_REAL_C(0.0), LC_REAL_C(1.0), LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(0.0)};
    uint32_t extrema_count;
    uint32_t count = 2U;
    uint32_t index;

    *iterations = 0U;
    coefficients[0] = y0 - threshold;
    coefficients[1] = model_h * k[0];
    coefficients[2] = model_h * (
        (-LC_REAL_RATIO(8048581381.0, 2820520608.0)) * k[0] +
        (LC_REAL_RATIO(131558114200.0, 32700410799.0)) * k[2] -
        (LC_REAL_RATIO(1754552775.0, 470086768.0)) * k[3] +
        (LC_REAL_RATIO(127303824393.0, 49829197408.0)) * k[4] -
        (LC_REAL_RATIO(282668133.0, 205662961.0)) * k[5] +
        (LC_REAL_RATIO(40617522.0, 29380423.0)) * k[6]
    );
    coefficients[3] = model_h * (
        (LC_REAL_RATIO(8663915743.0, 2820520608.0)) * k[0] -
        (LC_REAL_RATIO(68118460800.0, 10900136933.0)) * k[2] +
        (LC_REAL_RATIO(14199869525.0, 1410260304.0)) * k[3] -
        (LC_REAL_RATIO(318862633887.0, 49829197408.0)) * k[4] +
        (LC_REAL_RATIO(2019193451.0, 616988883.0)) * k[5] -
        (LC_REAL_RATIO(110615467.0, 29380423.0)) * k[6]
    );
    coefficients[4] = model_h * (
        (-LC_REAL_RATIO(12715105075.0, 11282082432.0)) * k[0] +
        (LC_REAL_RATIO(87487479700.0, 32700410799.0)) * k[2] -
        (LC_REAL_RATIO(10690763975.0, 1880347072.0)) * k[3] +
        (LC_REAL_RATIO(701980252875.0, 199316789632.0)) * k[4] -
        (LC_REAL_RATIO(1453857185.0, 822651844.0)) * k[5] +
        (LC_REAL_RATIO(69997945.0, 29380423.0)) * k[6]
    );
#if LACUNA_REAL_BITS <= 32
    for (index = 0U; index < 5U; ++index) {
        if (!lc_isfinite(coefficients[index])) {
            return -1;
        }
    }
#endif
    extrema_count = lc_step_dense_extrema(coefficients, extrema, iterations);
    if (extrema_count == UINT32_MAX) {
        return -1;
    }
    for (index = 0U; index < extrema_count; ++index) {
        partition[count++] = extrema[index];
    }
    lc_step_sort(partition, count);
    for (index = 0U; index + 1U < count; ++index) {
        lc_real_t low = partition[index];
        lc_real_t high = partition[index + 1U];
        lc_real_t g_low = lc_step_polynomial_value(coefficients, low);
        lc_real_t g_high = lc_step_polynomial_value(coefficients, high);
        if (!(g_low < LC_REAL_C(0.0) && g_high >= LC_REAL_C(0.0))) {
            continue;
        }
        if (g_high == LC_REAL_C(0.0)) {
            lc_real_t slope = lc_step_polynomial_derivative(coefficients, high);
            if (slope <=
#if LACUNA_REAL_BITS <= 32
                LC_REAL_C(0.0)
#else
                LC_REAL_C(64.0) * LC_REAL_EPSILON
#endif
            ) {
                continue;
            }
        }
        while ((high - low) * h > time_tolerance &&
               *iterations < LC_ROOT_MAX_ITERATIONS) {
            lc_real_t midpoint = low + LC_REAL_C(0.5) * (high - low);
            lc_real_t g_midpoint = lc_step_polynomial_value(coefficients, midpoint);
            if (midpoint == low || midpoint == high) {
                break;
            }
            (*iterations)++;
            if (g_midpoint >= LC_REAL_C(0.0)) {
                high = midpoint;
            } else {
                low = midpoint;
            }
        }
        *theta = low + LC_REAL_C(0.5) * (high - low);
        return 1;
    }
    return 0;
}

/* Integrate with adaptive error control and optional threshold detection. */
static lc_status lc_expr_step_integrate(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *rhs_roots,
    uint32_t state_count,
    uint32_t readout,
    lc_real_t threshold,
    const lc_step_config *config,
    lc_real_t *state,
    lc_time_t t_start,
    lc_time_t t_end,
    uint32_t clamped,
    uint32_t detect_crossing,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_step_result *result
) {
    lc_real_t y[LC_ANALYTICAL_MAX_STATES];
    lc_real_t trial[LC_ANALYTICAL_MAX_STATES];
    lc_real_t fourth[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k1[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k2[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k3[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k4[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k5[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k6[LC_ANALYTICAL_MAX_STATES];
    lc_real_t k7[LC_ANALYTICAL_MAX_STATES];
    lc_time_t current = t_start;
    lc_time_t step;
    uint32_t index;
    lc_status status;

    if (!lc_step_config_valid(config) || nodes == NULL || rhs_roots == NULL ||
        state == NULL || variables == NULL || workspace == NULL || result == NULL ||
        state_count == 0U || state_count > LC_ANALYTICAL_MAX_STATES ||
        readout >= state_count || variable_count != state_count + 1U ||
        !lc_isfinite(t_start) || !lc_isfinite(t_end) || t_end < t_start ||
        !lc_isfinite(threshold) || clamped > 1U || detect_crossing > 1U) {
        return LC_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->t_reached = t_start;
    result->t_crossing = NAN;
    for (index = 0U; index < state_count; ++index) {
        if (!lc_isfinite(state[index]) || rhs_roots[index] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
        y[index] = state[index];
    }
    if (detect_crossing != 0U && y[readout] >= threshold) {
        return LC_INVALID_ARGUMENT;
    }
    if (t_end == t_start) {
        return detect_crossing != 0U ? LC_NO_CROSSING : LC_OK;
    }
    step = lc_time_fmin(config->initial_step, lc_time_fmin(config->maximum_step, t_end - current));
#if LACUNA_REAL_BITS == 16
    /* Start on the next clock value if the preferred initial step is smaller. */
    {
        lc_time_t clock_step = lc_time_nextafter(current, t_end) - current;
        if (clock_step > config->maximum_step) {
            return LC_NUMERIC_ERROR;
        }
        step = lc_time_fmax(step, clock_step);
    }
#endif

    while (current < t_end) {
        lc_real_t model_step;
        lc_real_t error_norm = LC_REAL_C(0.0);
        lc_real_t factor;
        lc_time_t remaining = t_end - current;
        if (result->accepted_steps + result->rejected_steps >= config->maximum_steps) {
            return LC_STEP_LIMIT;
        }
        if (config->maximum_rhs_evaluations - result->rhs_evaluations < 7U) {
            return LC_RHS_EVALUATION_LIMIT;
        }
        step = lc_time_fmin(step, remaining);
#if LACUNA_REAL_BITS == 16
        /* Integrate the interval represented by the accepted endpoint. */
        {
            lc_time_t endpoint = lc_time_fmin(t_end, current + step);
            if (endpoint - current > config->maximum_step) {
                endpoint = lc_time_nextafter(endpoint, current);
            }
            step = endpoint - current;
        }
#endif
        model_step = (lc_real_t)step;
        if (!(step > LC_REAL_C(0.0)) || !lc_isfinite(step) ||
            !(current + step > current) || !lc_isfinite(model_step) ||
            !(model_step > LC_REAL_C(0.0))) {
            return LC_NUMERIC_ERROR;
        }

        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current, y, clamped, k1, variables, variable_count,
            workspace, workspace_count
        );
        if (status != LC_OK) return status;
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * (LC_REAL_RATIO(1.0, 5.0)) * k1[index];
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step * (LC_REAL_RATIO(1.0, 5.0)), trial, clamped, k2,
            variables, variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 2U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * ((LC_REAL_RATIO(3.0, 40.0)) * k1[index] +
                (LC_REAL_RATIO(9.0, 40.0)) * k2[index]);
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step * (LC_REAL_RATIO(3.0, 10.0)), trial, clamped, k3,
            variables, variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 3U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * ((LC_REAL_RATIO(44.0, 45.0)) * k1[index] -
                (LC_REAL_RATIO(56.0, 15.0)) * k2[index] + (LC_REAL_RATIO(32.0, 9.0)) * k3[index]);
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step * (LC_REAL_RATIO(4.0, 5.0)), trial, clamped, k4,
            variables, variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 4U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * ((LC_REAL_RATIO(19372.0, 6561.0)) * k1[index] -
                (LC_REAL_RATIO(25360.0, 2187.0)) * k2[index] + (LC_REAL_RATIO(64448.0, 6561.0)) * k3[index] -
                (LC_REAL_RATIO(212.0, 729.0)) * k4[index]);
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step * (LC_REAL_RATIO(8.0, 9.0)), trial, clamped, k5,
            variables, variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 5U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * ((LC_REAL_RATIO(9017.0, 3168.0)) * k1[index] -
                (LC_REAL_RATIO(355.0, 33.0)) * k2[index] + (LC_REAL_RATIO(46732.0, 5247.0)) * k3[index] +
                (LC_REAL_RATIO(49.0, 176.0)) * k4[index] - (LC_REAL_RATIO(5103.0, 18656.0)) * k5[index]);
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step, trial, clamped, k6, variables,
            variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 6U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        for (index = 0U; index < state_count; ++index)
            trial[index] = y[index] + model_step * ((LC_REAL_RATIO(35.0, 384.0)) * k1[index] +
                (LC_REAL_RATIO(500.0, 1113.0)) * k3[index] + (LC_REAL_RATIO(125.0, 192.0)) * k4[index] -
                (LC_REAL_RATIO(2187.0, 6784.0)) * k5[index] + (LC_REAL_RATIO(11.0, 84.0)) * k6[index]);
        status = lc_step_rhs(
            nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
            readout, current + step, trial, clamped, k7, variables,
            variable_count, workspace, workspace_count
        );
        if (status != LC_OK) {
#if LACUNA_REAL_BITS <= 32
            if (status == LC_NUMERIC_ERROR || status == LC_INVALID_ARGUMENT) {
                result->rhs_evaluations += 7U;
                goto numeric_trial_rejected;
            }
#endif
            return status;
        }
        result->rhs_evaluations += 7U;

        for (index = 0U; index < state_count; ++index) {
            lc_real_t scale;
            lc_real_t normalized;
            fourth[index] = y[index] + model_step * (
                (LC_REAL_RATIO(5179.0, 57600.0)) * k1[index] +
                (LC_REAL_RATIO(7571.0, 16695.0)) * k3[index] +
                (LC_REAL_RATIO(393.0, 640.0)) * k4[index] -
                (LC_REAL_RATIO(92097.0, 339200.0)) * k5[index] +
                (LC_REAL_RATIO(187.0, 2100.0)) * k6[index] + (LC_REAL_RATIO(1.0, 40.0)) * k7[index]
            );
            scale = config->absolute_tolerance + config->relative_tolerance *
                lc_real_fmax(lc_real_fabs(y[index]), lc_real_fabs(trial[index]));
            normalized = lc_real_fabs(trial[index] - fourth[index]) / scale;
            if (!lc_isfinite(normalized)) {
#if LACUNA_REAL_BITS <= 32
                goto numeric_trial_rejected;
#endif
                return LC_NUMERIC_ERROR;
            }
            error_norm = lc_real_fmax(error_norm, normalized);
        }
        result->error_norm = error_norm;
        result->last_step = step;
        if (error_norm <= LC_REAL_C(1.0)) {
            if (detect_crossing != 0U) {
                int crossing_found;
                lc_real_t crossing_theta;
                lc_real_t readout_derivatives[7] = {
                    k1[readout], k2[readout], k3[readout], k4[readout],
                    k5[readout], k6[readout], k7[readout]
                };
                uint32_t event_iterations;
                crossing_found = lc_step_find_rising_crossing(
                        y[readout], step, readout_derivatives, threshold,
                        config->event_tolerance, &crossing_theta,
                        &event_iterations);
                if (crossing_found < 0) {
                    return LC_ROOT_NONCONVERGENCE;
                }
                if (crossing_found != 0) {
                    result->accepted_steps++;
                    result->event_iterations += event_iterations;
                    result->t_crossing = current + crossing_theta * step;
#if LACUNA_REAL_BITS == 16
                    if (!lc_isfinite(result->t_crossing) ||
                        !(result->t_crossing > current) ||
                        result->t_crossing > current + step) {
                        return LC_NUMERIC_ERROR;
                    }
#endif
                    result->t_reached = result->t_crossing;
                    return LC_OK;
                }
                result->event_iterations += event_iterations;
            }
            for (index = 0U; index < state_count; ++index) {
                y[index] = trial[index];
            }
            current += step;
            result->accepted_steps++;
            result->t_reached = current;
            factor = error_norm == LC_REAL_C(0.0)
                ? LC_REAL_C(5.0)
                : lc_real_fmin(LC_REAL_C(5.0), lc_real_fmax(LC_REAL_C(0.2), LC_REAL_C(0.9) * lc_real_pow(error_norm, -LC_REAL_C(0.2))));
            step = lc_time_fmin(config->maximum_step, step * factor);
        } else {
            result->rejected_steps++;
            factor = lc_real_fmin(LC_REAL_C(0.5), lc_real_fmax(LC_REAL_C(0.1), LC_REAL_C(0.9) * lc_real_pow(error_norm, -LC_REAL_C(0.2))));
            step *= factor;
            if (step < config->minimum_step && remaining > config->minimum_step) {
                return LC_NUMERIC_ERROR;
            }
        }
#if LACUNA_REAL_BITS <= 32
        continue;
numeric_trial_rejected:
        /* A trial outside the model domain does not replace the accepted state. */
        result->rejected_steps++;
        result->last_step = step;
        result->error_norm = INFINITY;
        step *= LC_TIME_C(0.5);
        if (step < config->minimum_step) {
            return LC_NUMERIC_ERROR;
        }
#endif
    }
    if (detect_crossing != 0U) {
        return LC_NO_CROSSING;
    }
    for (index = 0U; index < state_count; ++index) {
        state[index] = y[index];
    }
    return LC_OK;
}

/* Advance a generic ODE state without searching for a spike. */
lc_status lc_expr_step_advance(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *rhs_roots,
    uint32_t state_count,
    uint32_t readout,
    const lc_step_config *config,
    lc_real_t *state,
    lc_time_t *t_last,
    lc_time_t t,
    uint32_t clamped,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_step_result *result
) {
    lc_status status;
    if (t_last == NULL || !lc_isfinite(*t_last) || t < *t_last) {
        return t_last != NULL && lc_isfinite(*t_last) && lc_isfinite(t)
            ? LC_TIME_REVERSED
            : LC_INVALID_ARGUMENT;
    }
    status = lc_expr_step_integrate(
        nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
        readout, LC_REAL_C(0.0), config, state, *t_last, t, clamped, 0U, variables,
        variable_count, workspace, workspace_count, result
    );
    if (status == LC_OK) {
        *t_last = t;
    }
    return status;
}

/* Predict the first rising crossing without mutating the input state. */
lc_status lc_expr_step_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *rhs_roots,
    uint32_t state_count,
    uint32_t readout,
    lc_real_t threshold,
    const lc_step_config *config,
    const lc_real_t *state,
    lc_time_t t_last,
    lc_time_t horizon,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_step_result *result
) {
    lc_real_t copy[LC_ANALYTICAL_MAX_STATES];
    uint32_t index;
    if (state == NULL || state_count == 0U ||
        state_count > LC_ANALYTICAL_MAX_STATES) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < state_count; ++index) {
        copy[index] = state[index];
    }
    return lc_expr_step_integrate(
        nodes, node_count, parameters, parameter_count, rhs_roots, state_count,
        readout, threshold, config, copy, t_last, horizon, 0U, 1U, variables,
        variable_count, workspace, workspace_count, result
    );
}

/* Invert the stable scalar affine trajectory with a logarithm. */
lc_status lc_expr_scalar_log_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_scalar_log_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    uint32_t readout,
    lc_time_t t_last,
    lc_time_t *t_spike,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_real_t decay;
    lc_real_t affine;
    lc_real_t threshold;
    lc_real_t asymptote;
    lc_real_t ratio;
    lc_time_t delta;
    lc_time_t next_time;
    uint32_t index;
    lc_status status;
    if (nodes == NULL || hint == NULL || state == NULL || t_spike == NULL ||
        variables == NULL || workspace == NULL || state_count == 0U ||
        state_count == UINT32_MAX || readout >= state_count ||
        variable_count != state_count + 1U || !lc_isfinite(t_last) ||
        hint->decay_root >= node_count || hint->affine_root >= node_count ||
        hint->threshold_root >= node_count) {
        return LC_INVALID_ARGUMENT;
    }
    variables[0] = LC_REAL_C(0.0);
    for (index = 0; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
        variables[index + 1U] = state[index];
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    decay = workspace[hint->decay_root];
    affine = workspace[hint->affine_root];
    threshold = workspace[hint->threshold_root];
    if (!(decay < LC_REAL_C(0.0)) || !lc_isfinite(affine) || !lc_isfinite(threshold) ||
        state[readout] >= threshold) {
        return LC_INVALID_ARGUMENT;
    }
    asymptote = -affine / decay;
    if (!lc_isfinite(asymptote)) {
        return LC_NUMERIC_ERROR;
    }
    if (asymptote <= threshold) {
        return LC_NO_CROSSING;
    }
    ratio = (threshold - asymptote) / (state[readout] - asymptote);
    if (!(ratio > LC_REAL_C(0.0) && ratio < LC_REAL_C(1.0))) {
        return LC_NUMERIC_ERROR;
    }
    delta = lc_real_log(ratio) / decay;
    next_time = t_last + delta;
    if (!lc_isfinite(delta) || delta <= LC_REAL_C(0.0) || !lc_isfinite(next_time) ||
        next_time <= t_last) {
        return LC_NUMERIC_ERROR;
    }
    *t_spike = next_time;
    return LC_OK;
}

typedef struct lc_alpha_root_context {
    const lc_expr_node *nodes;
    uint32_t node_count;
    const lc_real_t *parameters;
    uint32_t parameter_count;
    const lc_real_t *state;
    uint32_t state_count;
    lc_time_t t_last;
    lc_time_t fastest_time_constant;
    lc_real_t relative_tolerance;
    lc_real_t *variables;
    uint32_t variable_count;
    lc_real_t *workspace;
    uint32_t workspace_count;
    uint32_t iterations_used;
} lc_alpha_root_context;

typedef struct lc_bracket_solution {
    lc_time_t root;
    lc_time_t low;
    lc_time_t high;
    lc_real_t residual;
    lc_time_t tolerance;
    uint32_t iterations;
} lc_bracket_solution;

static int lc_same_sign(lc_real_t lhs, lc_real_t rhs) {
    return (lhs > LC_REAL_C(0.0) && rhs > LC_REAL_C(0.0)) || (lhs < LC_REAL_C(0.0) && rhs < LC_REAL_C(0.0));
}

static int lc_value_sign(lc_real_t value) {
    if (value > LC_REAL_C(0.0)) {
        return 1;
    }
    if (value < LC_REAL_C(0.0)) {
        return -1;
    }
    return 0;
}

/* Evaluate an alpha trajectory and its derivative at one elapsed time. */
static lc_status lc_alpha_evaluate(
    lc_alpha_root_context *context,
    lc_time_t delta,
    uint32_t value_root,
    uint32_t derivative_root,
    lc_real_t *value,
    lc_real_t *derivative
) {
    uint32_t index;
    lc_status status;
    if (!lc_isfinite(delta) || delta < LC_REAL_C(0.0) || value_root >= context->node_count ||
        derivative_root >= context->node_count || value == NULL || derivative == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    context->variables[0] = (lc_real_t)delta;
    for (index = 0; index < context->state_count; ++index) {
        context->variables[index + 1U] = context->state[index];
    }
    status = lc_expr_evaluate(
        context->nodes, context->node_count, context->parameters,
        context->parameter_count, context->variables, context->variable_count,
        context->workspace, context->workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    *value = context->workspace[value_root];
    *derivative = context->workspace[derivative_root];
    return LC_OK;
}

/* Classify a near-zero alpha crossing using a roundoff-aware tolerance. */
static lc_status lc_alpha_classified_crossing_value(
    lc_real_t threshold,
    lc_real_t asymptote,
    lc_real_t membrane_coefficient,
    lc_real_t synapse_constant,
    lc_real_t synapse_linear,
    lc_real_t membrane_rate,
    lc_real_t synapse_rate,
    lc_time_t delta,
    lc_real_t *value
) {
    lc_wide_t membrane_term;
    lc_wide_t synapse_constant_term;
    lc_wide_t synapse_linear_term;
    lc_wide_t total;
    lc_wide_t magnitude;
    if (value == NULL || !lc_isfinite(threshold) || !lc_isfinite(asymptote) ||
        !lc_isfinite(membrane_coefficient) || !lc_isfinite(synapse_constant) ||
        !lc_isfinite(synapse_linear) || !lc_isfinite(membrane_rate) ||
        !lc_isfinite(synapse_rate) || !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    membrane_term = (lc_wide_t)membrane_coefficient * lc_wide_exp(
        (lc_wide_t)membrane_rate * (lc_wide_t)delta
    );
    synapse_constant_term = (lc_wide_t)synapse_constant * lc_wide_exp(
        (lc_wide_t)synapse_rate * (lc_wide_t)delta
    );
    synapse_linear_term = (lc_wide_t)synapse_linear * (lc_wide_t)delta *
        lc_wide_exp((lc_wide_t)synapse_rate * (lc_wide_t)delta);
    total = (lc_wide_t)threshold - (lc_wide_t)asymptote - membrane_term -
        synapse_constant_term - synapse_linear_term;
    magnitude = lc_wide_fabs((lc_wide_t)threshold) +
        lc_wide_fabs((lc_wide_t)asymptote) + lc_wide_fabs(membrane_term) +
        lc_wide_fabs(synapse_constant_term) + lc_wide_fabs(synapse_linear_term);
    if (!lc_isfinite(total) || !lc_isfinite(magnitude)) {
        return LC_NUMERIC_ERROR;
    }
    /*
     * At a threshold tangency the analytical terms cancel. Their inputs are
     * doubles, so a sign at the accumulated roundoff floor cannot certify
     * which side of threshold the extremum occupies. Treat that narrow
     * interval as equality. Crossings with resolved separation
     * retain their sign and are still bracketed normally.
     */
    if (lc_wide_fabs(total) <= LC_WIDE_C(64.0) * (lc_wide_t)LC_REAL_EPSILON * magnitude) {
#if LACUNA_REAL_BITS <= 32
        return LC_ROOT_NONCONVERGENCE;
#else
        total = LC_WIDE_C(0.0);
#endif
    }
    if (lc_wide_fabs(total) > LC_REAL_MAX) {
        return LC_NUMERIC_ERROR;
    }
    *value = (lc_real_t)total;
    return lc_isfinite(*value) ? LC_OK : LC_NUMERIC_ERROR;
}

#if LACUNA_REAL_BITS <= 32
/* Root refinement cannot distinguish times that map to one trajectory input. */
static lc_time_t lc_trajectory_time_spacing(lc_time_t t_last, lc_time_t delta) {
    lc_time_t absolute_time = t_last + delta;
    lc_real_t model_delta = (lc_real_t)delta;
    lc_time_t clock_spacing = lc_time_nextafter(absolute_time, INFINITY) - absolute_time;
    lc_time_t model_spacing = (lc_time_t)(
        lc_real_nextafter(model_delta, INFINITY) - model_delta
    );
    return lc_time_fmax(clock_spacing, model_spacing);
}
#endif

static lc_time_t lc_root_tolerance(const lc_alpha_root_context *context, lc_time_t delta) {
    lc_time_t relative = context->relative_tolerance * context->fastest_time_constant;
#if LACUNA_REAL_BITS <= 32
    lc_time_t floor = lc_trajectory_time_spacing(context->t_last, delta);
#else
    lc_time_t absolute_time = lc_time_fabs(context->t_last + delta);
    lc_time_t floor = LC_REAL_C(8.0) * LC_TIME_EPSILON * lc_time_fmax(LC_REAL_C(1.0), absolute_time);
#endif
    return lc_time_fmax(relative, floor);
}

/* Refine a certified alpha crossing bracket with safeguarded iteration. */
static lc_status lc_solve_bracket(
    lc_alpha_root_context *context,
    uint32_t value_root,
    uint32_t derivative_root,
    lc_time_t low,
    lc_time_t high,
    lc_bracket_solution *solution
) {
    lc_real_t f_low;
    lc_real_t f_high;
    lc_real_t derivative;
    uint32_t iteration;
    lc_status status;
    if (solution == NULL || !lc_isfinite(low) || !lc_isfinite(high) || low < LC_REAL_C(0.0) ||
        high < low) {
        return LC_INVALID_ARGUMENT;
    }
    memset(solution, 0, sizeof(*solution));
    status = lc_alpha_evaluate(
        context, low, value_root, derivative_root, &f_low, &derivative
    );
    if (status != LC_OK) {
        return status;
    }
    status = lc_alpha_evaluate(
        context, high, value_root, derivative_root, &f_high, &derivative
    );
    if (status != LC_OK) {
        return status;
    }
    if (f_low == LC_REAL_C(0.0)) {
        solution->root = low;
        solution->low = low;
        solution->high = low;
        solution->residual = LC_REAL_C(0.0);
        solution->tolerance = lc_root_tolerance(context, low);
        return LC_OK;
    }
    if (f_high == LC_REAL_C(0.0)) {
        solution->root = high;
        solution->low = high;
        solution->high = high;
        solution->residual = LC_REAL_C(0.0);
        solution->tolerance = lc_root_tolerance(context, high);
        return LC_OK;
    }
    if (lc_same_sign(f_low, f_high)) {
        return LC_INVALID_ARGUMENT;
    }

    for (iteration = 0;
         iteration < LC_ROOT_MAX_ITERATIONS &&
         context->iterations_used < LC_ROOT_MAX_ITERATIONS;
         ++iteration) {
        lc_time_t width = high - low;
        lc_time_t midpoint = low + LC_REAL_C(0.5) * width;
        lc_time_t tolerance = lc_root_tolerance(context, midpoint);
        lc_time_t base;
        lc_real_t f_base;
        lc_time_t candidate;
        lc_real_t f_candidate;
        lc_real_t candidate_derivative;
        lc_real_t derivative_floor;
        int used_newton = 0;
        context->iterations_used++;
        if (width <= tolerance) {
            status = lc_alpha_evaluate(
                context, midpoint, value_root, derivative_root,
                &f_candidate, &candidate_derivative
            );
            if (status != LC_OK) {
                return status;
            }
            solution->root = midpoint;
            solution->low = low;
            solution->high = high;
            solution->residual = f_candidate;
            solution->tolerance = tolerance;
            solution->iterations = iteration + 1U;
            return LC_OK;
        }

        base = lc_real_fabs(f_low) <= lc_real_fabs(f_high) ? low : high;
        status = lc_alpha_evaluate(
            context, base, value_root, derivative_root, &f_base, &derivative
        );
        if (status != LC_OK) {
            return status;
        }
        derivative_floor = lc_real_sqrt(LC_REAL_EPSILON) * lc_real_fmax(LC_REAL_C(1.0), lc_real_fabs(f_base)) /
            context->fastest_time_constant;
        candidate = midpoint;
        if (lc_real_fabs(derivative) > derivative_floor) {
            lc_time_t newton = base - f_base / derivative;
            if (lc_isfinite(newton) && newton > low && newton < high) {
                candidate = newton;
                used_newton = 1;
            }
        }
        status = lc_alpha_evaluate(
            context, candidate, value_root, derivative_root,
            &f_candidate, &candidate_derivative
        );
        if (status != LC_OK) {
            return status;
        }
        if (f_candidate == LC_REAL_C(0.0)) {
            solution->root = candidate;
            solution->low = candidate;
            solution->high = candidate;
            solution->residual = LC_REAL_C(0.0);
            solution->tolerance = lc_root_tolerance(context, candidate);
            solution->iterations = iteration + 1U;
            return LC_OK;
        }

        if (used_newton) {
            lc_time_t tentative_width = lc_same_sign(f_candidate, f_low)
                ? high - candidate
                : candidate - low;
            if (tentative_width > LC_REAL_C(0.5) * width) {
                candidate = midpoint;
                status = lc_alpha_evaluate(
                    context, candidate, value_root, derivative_root,
                    &f_candidate, &candidate_derivative
                );
                if (status != LC_OK) {
                    return status;
                }
                if (f_candidate == LC_REAL_C(0.0)) {
                    solution->root = candidate;
                    solution->low = candidate;
                    solution->high = candidate;
                    solution->residual = LC_REAL_C(0.0);
                    solution->tolerance = lc_root_tolerance(context, candidate);
                    solution->iterations = iteration + 1U;
                    return LC_OK;
                }
            }
        }
        if (lc_same_sign(f_candidate, f_low)) {
            low = candidate;
            f_low = f_candidate;
        } else {
            high = candidate;
            f_high = f_candidate;
        }
    }

    solution->root = low + LC_REAL_C(0.5) * (high - low);
    solution->low = low;
    solution->high = high;
    solution->tolerance = lc_root_tolerance(context, solution->root);
    solution->iterations = iteration;
    status = lc_alpha_evaluate(
        context, solution->root, value_root, derivative_root,
        &solution->residual, &derivative
    );
    return status == LC_OK ? LC_ROOT_NONCONVERGENCE : status;
}

/* Insert a distinct extremum that lies within the prediction horizon. */
static int lc_add_extremum(
    lc_time_t *extrema,
    uint32_t *count,
    lc_time_t value,
    lc_time_t horizon,
    lc_time_t tolerance
) {
    if (value <= tolerance || value >= horizon - tolerance) {
        return 1;
    }
    if (*count > 0 && lc_time_fabs(extrema[*count - 1U] - value) <= tolerance) {
        return 1;
    }
    if (*count >= 2U) {
        return 0;
    }
    extrema[*count] = value;
    (*count)++;
    return 1;
}

/* Partition a two-mode trajectory at extrema and solve its first crossing. */
lc_status lc_expr_two_exp_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_two_exp_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    lc_time_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_alpha_root_context context;
    uint32_t roots[7];
    uint32_t index;
    lc_real_t g_zero;
    lc_real_t unused_derivative;
    lc_real_t limit;
    lc_real_t coefficient_one;
    lc_real_t coefficient_two;
    lc_real_t rate_one;
    lc_real_t rate_two;
    lc_time_t horizon;
    lc_time_t extremum = LC_REAL_C(0.0);
    int has_extremum = 0;
    lc_time_t partition[3];
    uint32_t partition_count = 0U;
    lc_status status;

    if (nodes == NULL || hint == NULL || state == NULL || result == NULL ||
        variables == NULL || workspace == NULL || state_count != 2U ||
        variable_count != state_count + 1U || !lc_isfinite(t_last) || t_last < LC_REAL_C(0.0) ||
        !lc_isfinite(hint->relative_tolerance) || hint->relative_tolerance <= LC_REAL_C(0.0) ||
        !lc_isfinite(hint->fastest_time_constant) || hint->fastest_time_constant <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    roots[0] = hint->g_root;
    roots[1] = hint->g_prime_root;
    roots[2] = hint->limit_root;
    roots[3] = hint->coefficient_one_root;
    roots[4] = hint->coefficient_two_root;
    roots[5] = hint->rate_one_root;
    roots[6] = hint->rate_two_root;
    for (index = 0; index < 7U; ++index) {
        if (roots[index] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (index = 0; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
    }

    memset(result, 0, sizeof(*result));
    memset(&context, 0, sizeof(context));
    context.nodes = nodes;
    context.node_count = node_count;
    context.parameters = parameters;
    context.parameter_count = parameter_count;
    context.state = state;
    context.state_count = state_count;
    context.t_last = t_last;
    context.fastest_time_constant = hint->fastest_time_constant;
    context.relative_tolerance = hint->relative_tolerance;
    context.variables = variables;
    context.variable_count = variable_count;
    context.workspace = workspace;
    context.workspace_count = workspace_count;

    status = lc_alpha_evaluate(
        &context, LC_REAL_C(0.0), hint->g_root, hint->g_prime_root,
        &g_zero, &unused_derivative
    );
    if (status != LC_OK) {
        return status;
    }
    limit = workspace[hint->limit_root];
    coefficient_one = workspace[hint->coefficient_one_root];
    coefficient_two = workspace[hint->coefficient_two_root];
    rate_one = workspace[hint->rate_one_root];
    rate_two = workspace[hint->rate_two_root];
    if (!lc_isfinite(limit) || !lc_isfinite(coefficient_one) ||
        !lc_isfinite(coefficient_two) || !lc_isfinite(rate_one) ||
        !lc_isfinite(rate_two) || rate_one >= LC_REAL_C(0.0) || rate_two >= LC_REAL_C(0.0) ||
        rate_one == rate_two) {
        return LC_UNSUPPORTED_MODEL;
    }
    if (g_zero <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }

#if LACUNA_REAL_BITS == 16
    /* Losing a nonzero derivative term cannot certify a monotone trajectory. */
    if ((coefficient_one != LC_REAL_C(0.0) &&
         rate_one * coefficient_one == LC_REAL_C(0.0)) ||
        (coefficient_two != LC_REAL_C(0.0) &&
         rate_two * coefficient_two == LC_REAL_C(0.0))) {
        return LC_ROOT_NONCONVERGENCE;
    }
#endif
    if (limit == LC_REAL_C(0.0)) {
        if (coefficient_one != LC_REAL_C(0.0) && coefficient_two != LC_REAL_C(0.0)) {
            lc_real_t ratio = -coefficient_two / coefficient_one;
#if LACUNA_REAL_BITS == 16
            if (!lc_isfinite(ratio) || ratio == LC_REAL_C(0.0))
                return LC_ROOT_NONCONVERGENCE;
#endif
            if (lc_isfinite(ratio) && ratio > LC_REAL_C(0.0)) {
                lc_time_t candidate = lc_real_log(ratio) / (rate_one - rate_two);
                if (lc_isfinite(candidate) && candidate > LC_REAL_C(0.0)) {
                    lc_real_t g_candidate;
                    lc_real_t g_prime_candidate;
                    status = lc_alpha_evaluate(
                        &context, candidate, hint->g_root, hint->g_prime_root,
                        &g_candidate, &g_prime_candidate
                    );
                    if (status != LC_OK) {
                        return status;
                    }
                    result->horizon = t_last + candidate;
                    if (g_prime_candidate < LC_REAL_C(0.0)) {
                        result->t_spike = t_last + candidate;
                        result->bracket_low = result->t_spike;
                        result->bracket_high = result->t_spike;
                        result->residual = g_candidate;
                        result->tolerance = lc_root_tolerance(&context, candidate);
                        return LC_OK;
                    }
                }
            }
        }
        return LC_NO_CROSSING;
    }

    horizon = lc_time_fmax(-LC_REAL_C(1.0) / rate_one, -LC_REAL_C(1.0) / rate_two);
    if (!lc_isfinite(horizon) || horizon <= LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    for (index = 0; index < 128U; ++index) {
        lc_real_t envelope = lc_real_fabs(coefficient_one) * lc_real_exp(rate_one * horizon) +
            lc_real_fabs(coefficient_two) * lc_real_exp(rate_two * horizon);
        if (!lc_isfinite(envelope)) {
            return LC_NUMERIC_ERROR;
        }
        if (envelope < lc_real_fabs(limit)) {
            break;
        }
        horizon *= LC_REAL_C(2.0);
        if (!lc_isfinite(horizon)) {
            return LC_NUMERIC_ERROR;
        }
    }
    if (index == 128U) {
        return LC_ROOT_NONCONVERGENCE;
    }
    result->horizon = t_last + horizon;
    if (!lc_isfinite(result->horizon)) {
        return LC_NUMERIC_ERROR;
    }

    if (rate_one * coefficient_one != LC_REAL_C(0.0) &&
        rate_two * coefficient_two != LC_REAL_C(0.0)) {
        lc_real_t ratio = -(rate_two * coefficient_two) /
            (rate_one * coefficient_one);
#if LACUNA_REAL_BITS == 16
        if (!lc_isfinite(ratio) || ratio == LC_REAL_C(0.0))
            return LC_ROOT_NONCONVERGENCE;
#endif
        if (lc_isfinite(ratio) && ratio > LC_REAL_C(0.0)) {
            lc_time_t candidate = lc_real_log(ratio) / (rate_one - rate_two);
            if (lc_isfinite(candidate)) {
                lc_time_t tolerance = lc_root_tolerance(&context, candidate);
                if (candidate > tolerance && candidate < horizon - tolerance) {
                    extremum = candidate;
                    has_extremum = 1;
                    result->extrema_count = 1U;
                }
            }
        }
    }

    partition[partition_count++] = LC_REAL_C(0.0);
    if (has_extremum) {
        partition[partition_count++] = extremum;
    }
    partition[partition_count++] = horizon;
    for (index = 0; index + 1U < partition_count; ++index) {
        lc_time_t low = partition[index];
        lc_time_t high = partition[index + 1U];
        lc_real_t g_low;
        lc_real_t g_high;
        lc_real_t g_prime_high;
        lc_bracket_solution crossing_solution;
        status = lc_alpha_evaluate(
            &context, low, hint->g_root, hint->g_prime_root,
            &g_low, &unused_derivative
        );
        if (status != LC_OK) {
            return status;
        }
        status = lc_alpha_evaluate(
            &context, high, hint->g_root, hint->g_prime_root,
            &g_high, &g_prime_high
        );
        if (status != LC_OK) {
            return status;
        }
        if (g_low > LC_REAL_C(0.0) && g_high < LC_REAL_C(0.0)) {
            status = lc_solve_bracket(
                &context, hint->g_root, hint->g_prime_root,
                low, high, &crossing_solution
            );
            result->iterations += crossing_solution.iterations;
            result->bracket_low = t_last + crossing_solution.low;
            result->bracket_high = t_last + crossing_solution.high;
            result->residual = crossing_solution.residual;
            result->tolerance = crossing_solution.tolerance;
            if (status != LC_OK) {
                return status;
            }
            result->t_spike = t_last + crossing_solution.root;
            if (!lc_isfinite(result->t_spike) || result->t_spike <= t_last) {
                return LC_NUMERIC_ERROR;
            }
            return LC_OK;
        }
        if (g_high == LC_REAL_C(0.0) && g_prime_high < LC_REAL_C(0.0) && high > LC_REAL_C(0.0)) {
            result->t_spike = t_last + high;
            result->bracket_low = result->t_spike;
            result->bracket_high = result->t_spike;
            result->residual = LC_REAL_C(0.0);
            result->tolerance = lc_root_tolerance(&context, high);
            return LC_OK;
        }
    }
    return LC_NO_CROSSING;
}

typedef struct lc_multi_exp_context {
    lc_time_t t_last;
    lc_real_t relative_tolerance;
    lc_time_t fastest_time_constant;
    uint32_t iteration_cap;
    uint32_t iterations_used;
} lc_multi_exp_context;

/* Set the elapsed-time tolerance with an absolute-time roundoff floor. */
static lc_time_t lc_multi_exp_tolerance(
    const lc_multi_exp_context *context,
    lc_time_t delta
) {
    lc_time_t relative = context->relative_tolerance * context->fastest_time_constant;
#if LACUNA_REAL_BITS <= 32
    lc_time_t floor = lc_trajectory_time_spacing(context->t_last, delta);
#else
    lc_time_t absolute_time = lc_time_fabs(context->t_last + delta);
    lc_time_t floor = LC_REAL_C(8.0) * LC_TIME_EPSILON * lc_time_fmax(LC_REAL_C(1.0), absolute_time);
#endif
    return lc_time_fmax(relative, floor);
}

/* Set a stricter tolerance for recursive root isolation. */
static lc_time_t lc_multi_exp_isolation_tolerance(
    const lc_multi_exp_context *context,
    lc_time_t delta
) {
#if LACUNA_REAL_BITS <= 32
    return lc_trajectory_time_spacing(context->t_last, delta);
#else
    lc_time_t absolute_time = lc_time_fabs(context->t_last + delta);
    return LC_REAL_C(16.0) * LC_TIME_EPSILON * lc_time_fmax(LC_REAL_C(1.0), absolute_time);
#endif
}

/* Evaluate a stable real exponential sum in its original scale. */
static lc_status lc_multi_exp_value(
    lc_real_t limit,
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t delta,
    lc_real_t *value
) {
    uint32_t index;
    lc_wide_t total;
    lc_wide_t magnitude;
    if (coefficients == NULL || rates == NULL || value == NULL ||
        !lc_isfinite(limit) || !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    total = (lc_wide_t)limit;
    magnitude = lc_wide_fabs(total);
    for (index = 0U; index < count; ++index) {
        lc_wide_t term = (lc_wide_t)coefficients[index] *
            lc_wide_exp((lc_wide_t)rates[index] * (lc_wide_t)delta);
        total += term;
        magnitude += lc_wide_fabs(term);
    }
#if LACUNA_REAL_BITS == 64
    if (lc_wide_fabs(total) <= LC_WIDE_C(32.0) * LC_REAL_EPSILON * magnitude) {
        total = LC_WIDE_C(0.0);
    }
#else
    if (!lc_isfinite(magnitude)) {
        return LC_ROOT_NONCONVERGENCE;
    }
#endif
    if (!lc_isfinite(total) || lc_wide_fabs(total) > LC_REAL_MAX) {
        return LC_NUMERIC_ERROR;
    }
    *value = (lc_real_t)total;
    return lc_isfinite(*value) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Remove a shared exponential scale before sign classification. */
static lc_status lc_multi_exp_normalized_value(
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t delta,
    lc_real_t *value
) {
    uint32_t index;
    lc_wide_t total;
    lc_wide_t magnitude;
    lc_real_t base;
    if (coefficients == NULL || rates == NULL || value == NULL || count == 0U ||
        !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    base = rates[count - 1U];
    total = (lc_wide_t)coefficients[count - 1U];
    magnitude = lc_wide_fabs(total);
    for (index = 0U; index + 1U < count; ++index) {
        lc_wide_t term = (lc_wide_t)coefficients[index] * lc_wide_exp(
            ((lc_wide_t)rates[index] - (lc_wide_t)base) * (lc_wide_t)delta
        );
        total += term;
        magnitude += lc_wide_fabs(term);
    }
#if LACUNA_REAL_BITS == 64
    if (lc_wide_fabs(total) <= LC_WIDE_C(32.0) * LC_REAL_EPSILON * magnitude) {
        total = LC_WIDE_C(0.0);
    }
#else
    if (!lc_isfinite(magnitude)) {
        return LC_ROOT_NONCONVERGENCE;
    }
#endif
    if (!lc_isfinite(total) || lc_wide_fabs(total) > LC_REAL_MAX) {
        return LC_NUMERIC_ERROR;
    }
    *value = (lc_real_t)total;
    return lc_isfinite(*value) ? LC_OK : LC_NUMERIC_ERROR;
}

#if LACUNA_REAL_BITS <= 32
/* An uncertain extremum cannot establish whether the trajectory crosses. */
static lc_status lc_multi_exp_check_extremum(
    lc_real_t limit,
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t delta,
    lc_real_t base
) {
    lc_wide_t total = (lc_wide_t)limit;
    lc_wide_t magnitude = lc_wide_fabs(total);
    uint32_t index;
    for (index = 0U; index < count; ++index) {
        lc_wide_t term = (lc_wide_t)coefficients[index] * lc_wide_exp(
            ((lc_wide_t)rates[index] - (lc_wide_t)base) * (lc_wide_t)delta
        );
        total += term;
        magnitude += lc_wide_fabs(term);
    }
    if (!lc_isfinite(total) || !lc_isfinite(magnitude) ||
        lc_wide_fabs(total) <= LC_WIDE_C(64.0) * LC_WIDE_EPSILON * magnitude) {
        return LC_ROOT_NONCONVERGENCE;
    }
    return LC_OK;
}
#endif

/* Add one isolated root without erasing distinct binary32 roots. */
static int lc_multi_exp_add_root(
    lc_multi_exp_context *context,
    lc_time_t *roots,
    uint32_t *root_count,
    uint32_t capacity,
    lc_time_t root,
    lc_time_t low,
    lc_time_t high
) {
#if LACUNA_REAL_BITS <= 32
    (void)context;
    if (root <= low || root >= high) {
        return 1;
    }
    if (*root_count > 0U && roots[*root_count - 1U] == root) {
        return 1;
    }
#else
    lc_time_t tolerance = lc_multi_exp_tolerance(context, root);
    if (root <= low + tolerance || root >= high - tolerance) {
        return 1;
    }
    if (*root_count > 0U && lc_time_fabs(roots[*root_count - 1U] - root) <= tolerance) {
        return 1;
    }
#endif
    if (*root_count >= capacity) {
        return 0;
    }
    roots[*root_count] = root;
    (*root_count)++;
    return 1;
}

/* Bisect an isolated root using normalized values to avoid underflow. */
static lc_status lc_multi_exp_bisect_normalized(
    lc_multi_exp_context *context,
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t low,
    lc_time_t high,
    lc_time_t *root
) {
    lc_real_t f_low;
    lc_real_t f_high;
    lc_status status;
    if (context == NULL || root == NULL || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    status = lc_multi_exp_normalized_value(
        coefficients, rates, count, low, &f_low
    );
    if (status != LC_OK) {
        return status;
    }
    status = lc_multi_exp_normalized_value(
        coefficients, rates, count, high, &f_high
    );
    if (status != LC_OK) {
        return status;
    }
    if (f_low == LC_REAL_C(0.0)) {
        *root = low;
        return LC_OK;
    }
    if (f_high == LC_REAL_C(0.0)) {
        *root = high;
        return LC_OK;
    }
    if (lc_same_sign(f_low, f_high)) {
        return LC_INVALID_ARGUMENT;
    }
    while (high - low > lc_multi_exp_isolation_tolerance(
            context, low + LC_REAL_C(0.5) * (high - low))) {
        lc_time_t midpoint;
        lc_real_t f_midpoint;
        if (context->iterations_used >= context->iteration_cap) {
            *root = low + LC_REAL_C(0.5) * (high - low);
            return LC_ROOT_NONCONVERGENCE;
        }
        context->iterations_used++;
        midpoint = low + LC_REAL_C(0.5) * (high - low);
        status = lc_multi_exp_normalized_value(
            coefficients, rates, count, midpoint, &f_midpoint
        );
        if (status != LC_OK) {
            return status;
        }
        if (f_midpoint == LC_REAL_C(0.0)) {
            *root = midpoint;
            return LC_OK;
        }
        if (lc_same_sign(f_low, f_midpoint)) {
            low = midpoint;
            f_low = f_midpoint;
        } else {
            high = midpoint;
            f_high = f_midpoint;
        }
    }
    *root = low + LC_REAL_C(0.5) * (high - low);
    return LC_OK;
}

/*
 * Isolate every root of a real exponential polynomial on a finite interval.
 * Factoring out the slowest exponential leaves a constant plus one fewer
 * exponentials. Rolle recursion therefore gives bounded monotone partitions.
 */
static lc_status lc_multi_exp_roots(
    lc_multi_exp_context *context,
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t low,
    lc_time_t high,
    lc_time_t *roots,
    uint32_t capacity,
    uint32_t *root_count
) {
    lc_real_t derivative_coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t derivative_rates[LC_ANALYTICAL_MAX_STATES];
    lc_time_t critical[LC_ANALYTICAL_MAX_STATES];
    uint32_t derivative_count = 0U;
    uint32_t critical_count = 0U;
    uint32_t index;
    lc_real_t base;
    lc_status status;
    if (context == NULL || coefficients == NULL || rates == NULL || roots == NULL ||
        root_count == NULL || count == 0U || count > LC_ANALYTICAL_MAX_STATES ||
        !lc_isfinite(low) || !lc_isfinite(high) || low < LC_REAL_C(0.0) || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    *root_count = 0U;
    if (count == 1U || high == low) {
        return LC_OK;
    }
    base = rates[count - 1U];
    for (index = 0U; index + 1U < count; ++index) {
        lc_wide_t coefficient = (lc_wide_t)coefficients[index] *
            ((lc_wide_t)rates[index] - (lc_wide_t)base);
#if LACUNA_REAL_BITS == 16
        if (coefficient == LC_WIDE_C(0.0) && coefficients[index] != LC_REAL_C(0.0) &&
            rates[index] != base) return LC_ROOT_NONCONVERGENCE;
#endif
        if (!lc_isfinite(coefficient) || lc_wide_fabs(coefficient) > LC_REAL_MAX) {
            return LC_NUMERIC_ERROR;
        }
        if (coefficient != LC_WIDE_C(0.0)) {
            derivative_coefficients[derivative_count] = (lc_real_t)coefficient;
            derivative_rates[derivative_count] = rates[index] - base;
            derivative_count++;
        }
    }
    if (derivative_count > 1U) {
        status = lc_multi_exp_roots(
            context, derivative_coefficients, derivative_rates, derivative_count,
            low, high, critical, LC_ANALYTICAL_MAX_STATES, &critical_count
        );
        if (status != LC_OK) {
            return status;
        }
    }
    for (index = 0U; index <= critical_count; ++index) {
        lc_time_t interval_low = index == 0U ? low : critical[index - 1U];
        lc_time_t interval_high = index == critical_count ? high : critical[index];
        lc_real_t f_low;
        lc_real_t f_high;
        status = lc_multi_exp_normalized_value(
            coefficients, rates, count, interval_low, &f_low
        );
        if (status != LC_OK) {
            return status;
        }
        status = lc_multi_exp_normalized_value(
            coefficients, rates, count, interval_high, &f_high
        );
        if (status != LC_OK) {
            return status;
        }
#if LACUNA_REAL_BITS <= 32
        if (index < critical_count) {
            status = lc_multi_exp_check_extremum(
                LC_REAL_C(0.0), coefficients, rates, count, interval_high,
                rates[count - 1U]
            );
            if (status != LC_OK) {
                return status;
            }
        }
#endif
        if (f_low == LC_REAL_C(0.0) && !lc_multi_exp_add_root(
                context, roots, root_count, capacity, interval_low, low, high)) {
            return LC_ROOT_NONCONVERGENCE;
        }
        if (f_high == LC_REAL_C(0.0) && !lc_multi_exp_add_root(
                context, roots, root_count, capacity, interval_high, low, high)) {
            return LC_ROOT_NONCONVERGENCE;
        }
        if (!lc_same_sign(f_low, f_high) && f_low != LC_REAL_C(0.0) && f_high != LC_REAL_C(0.0)) {
            lc_time_t root;
            status = lc_multi_exp_bisect_normalized(
                context, coefficients, rates, count,
                interval_low, interval_high, &root
            );
            if (status != LC_OK) {
                return status;
            }
            if (!lc_multi_exp_add_root(
                    context, roots, root_count, capacity, root, low, high)) {
                return LC_ROOT_NONCONVERGENCE;
            }
        }
    }
    return LC_OK;
}

/* Refine a rising trajectory crossing in one monotone interval. */
static lc_status lc_multi_exp_bisect_crossing(
    lc_multi_exp_context *context,
    lc_real_t limit,
    const lc_real_t *coefficients,
    const lc_real_t *rates,
    uint32_t count,
    lc_time_t low,
    lc_time_t high,
    lc_bracket_solution *solution
) {
    lc_real_t f_low;
    lc_real_t f_high;
    lc_status status;
    if (context == NULL || solution == NULL || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    memset(solution, 0, sizeof(*solution));
    status = lc_multi_exp_value(limit, coefficients, rates, count, low, &f_low);
    if (status != LC_OK) {
        return status;
    }
    status = lc_multi_exp_value(limit, coefficients, rates, count, high, &f_high);
    if (status != LC_OK) {
        return status;
    }
    if (!(f_low > LC_REAL_C(0.0) && f_high <= LC_REAL_C(0.0))) {
        return LC_INVALID_ARGUMENT;
    }
    while (high - low > lc_multi_exp_tolerance(context, low + LC_REAL_C(0.5) * (high - low))) {
        lc_time_t midpoint;
        lc_real_t f_midpoint;
        if (context->iterations_used >= context->iteration_cap) {
            solution->root = low + LC_REAL_C(0.5) * (high - low);
            solution->low = low;
            solution->high = high;
            solution->tolerance = lc_multi_exp_tolerance(context, solution->root);
            lc_multi_exp_value(
                limit, coefficients, rates, count, solution->root,
                &solution->residual
            );
            return LC_ROOT_NONCONVERGENCE;
        }
        context->iterations_used++;
        solution->iterations++;
        midpoint = low + LC_REAL_C(0.5) * (high - low);
        status = lc_multi_exp_value(
            limit, coefficients, rates, count, midpoint, &f_midpoint
        );
        if (status != LC_OK) {
            return status;
        }
        if (f_midpoint > LC_REAL_C(0.0)) {
            low = midpoint;
            f_low = f_midpoint;
        } else {
            high = midpoint;
            f_high = f_midpoint;
        }
    }
    solution->root = low + LC_REAL_C(0.5) * (high - low);
    solution->low = low;
    solution->high = high;
    solution->tolerance = lc_multi_exp_tolerance(context, solution->root);
    status = lc_multi_exp_value(
        limit, coefficients, rates, count, solution->root, &solution->residual
    );
    return status;
}

/* Recursively isolate extrema of a bounded real exponential trajectory. */
lc_status lc_expr_multi_exp_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_multi_exp_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    lc_time_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_real_t coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t rates[LC_ANALYTICAL_MAX_STATES];
    lc_real_t compact_coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t compact_rates[LC_ANALYTICAL_MAX_STATES];
    lc_real_t derivative_coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t derivative_rates[LC_ANALYTICAL_MAX_STATES];
    lc_time_t extrema[LC_ANALYTICAL_MAX_STATES];
    lc_multi_exp_context context;
    lc_real_t limit;
    lc_real_t g_zero;
    lc_time_t horizon;
    uint32_t compact_count = 0U;
    uint32_t derivative_count = 0U;
    uint32_t extrema_count = 0U;
    uint32_t index;
    lc_status status;

    if (nodes == NULL || hint == NULL || state == NULL || result == NULL ||
        variables == NULL || workspace == NULL || state_count == 0U ||
        state_count > LC_ANALYTICAL_MAX_STATES ||
        variable_count != state_count + 1U || hint->mode_count == 0U ||
        hint->mode_count > LC_ANALYTICAL_MAX_STATES ||
        hint->iteration_cap == 0U || hint->iteration_cap > 4096U ||
        !lc_isfinite(t_last) || t_last < LC_REAL_C(0.0) ||
        !lc_isfinite(hint->relative_tolerance) || hint->relative_tolerance <= LC_REAL_C(0.0) ||
        !lc_isfinite(hint->fastest_time_constant) ||
        hint->fastest_time_constant <= LC_REAL_C(0.0) || hint->limit_root >= node_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
        variables[index + 1U] = state[index];
    }
    variables[0] = LC_REAL_C(0.0);
    for (index = 0U; index < hint->mode_count; ++index) {
        if (hint->coefficient_roots[index] >= node_count ||
            hint->rate_roots[index] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    limit = workspace[hint->limit_root];
    if (!lc_isfinite(limit)) {
        return LC_NUMERIC_ERROR;
    }
    for (index = 0U; index < hint->mode_count; ++index) {
        uint32_t cursor;
        coefficients[index] = workspace[hint->coefficient_roots[index]];
        rates[index] = workspace[hint->rate_roots[index]];
        if (!lc_isfinite(coefficients[index]) || !lc_isfinite(rates[index]) ||
            rates[index] >= LC_REAL_C(0.0)) {
            return LC_UNSUPPORTED_MODEL;
        }
        cursor = index;
        while (cursor > 0U && rates[cursor] < rates[cursor - 1U]) {
            lc_real_t swap;
            swap = rates[cursor - 1U];
            rates[cursor - 1U] = rates[cursor];
            rates[cursor] = swap;
            swap = coefficients[cursor - 1U];
            coefficients[cursor - 1U] = coefficients[cursor];
            coefficients[cursor] = swap;
            cursor--;
        }
    }
    for (index = 0U; index < hint->mode_count; ++index) {
        if (compact_count > 0U && rates[index] == compact_rates[compact_count - 1U]) {
            lc_wide_t combined = (lc_wide_t)compact_coefficients[compact_count - 1U] +
                (lc_wide_t)coefficients[index];
            if (!lc_isfinite(combined) || lc_wide_fabs(combined) > LC_REAL_MAX) {
                return LC_NUMERIC_ERROR;
            }
            compact_coefficients[compact_count - 1U] = (lc_real_t)combined;
        } else {
            compact_rates[compact_count] = rates[index];
            compact_coefficients[compact_count] = coefficients[index];
            compact_count++;
        }
    }
    {
        uint32_t destination = 0U;
        for (index = 0U; index < compact_count; ++index) {
            if (compact_coefficients[index] != LC_REAL_C(0.0)) {
                compact_rates[destination] = compact_rates[index];
                compact_coefficients[destination] = compact_coefficients[index];
                destination++;
            }
        }
        compact_count = destination;
    }
    memset(result, 0, sizeof(*result));
    memset(&context, 0, sizeof(context));
    context.t_last = t_last;
    context.relative_tolerance = hint->relative_tolerance;
    context.fastest_time_constant = hint->fastest_time_constant;
    context.iteration_cap = hint->iteration_cap;
    status = lc_multi_exp_value(
        limit, compact_coefficients, compact_rates, compact_count, LC_REAL_C(0.0), &g_zero
    );
    if (status != LC_OK) {
        return status;
    }
    if (g_zero <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    if (compact_count == 0U) {
        return LC_NO_CROSSING;
    }
    horizon = -LC_REAL_C(1.0) / compact_rates[compact_count - 1U];
    if (!lc_isfinite(horizon) || horizon <= LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    for (index = 0U; index < 128U; ++index) {
        lc_wide_t envelope = LC_WIDE_C(0.0);
        uint32_t cursor;
        if (limit != LC_REAL_C(0.0)) {
            for (cursor = 0U; cursor < compact_count; ++cursor) {
                envelope += lc_wide_fabs((lc_wide_t)compact_coefficients[cursor]) * lc_wide_exp(
                    (lc_wide_t)compact_rates[cursor] * (lc_wide_t)horizon
                );
            }
            if (envelope < lc_wide_fabs((lc_wide_t)limit)) {
                break;
            }
        } else if (compact_count == 1U) {
            break;
        } else {
            lc_real_t slow_rate = compact_rates[compact_count - 1U];
            for (cursor = 0U; cursor + 1U < compact_count; ++cursor) {
                envelope += lc_wide_fabs((lc_wide_t)compact_coefficients[cursor]) * lc_wide_exp(
                    ((lc_wide_t)compact_rates[cursor] - (lc_wide_t)slow_rate) *
                    (lc_wide_t)horizon
                );
            }
            if (envelope < lc_wide_fabs(
                    (lc_wide_t)compact_coefficients[compact_count - 1U])) {
                break;
            }
        }
        horizon *= LC_REAL_C(2.0);
        if (!lc_isfinite(horizon)) {
            return LC_NUMERIC_ERROR;
        }
    }
    if (index == 128U) {
        return LC_ROOT_NONCONVERGENCE;
    }
    result->horizon = t_last + horizon;
    if (!lc_isfinite(result->horizon)) {
        return LC_NUMERIC_ERROR;
    }
    for (index = 0U; index < compact_count; ++index) {
        lc_wide_t derivative = (lc_wide_t)compact_coefficients[index] *
            (lc_wide_t)compact_rates[index];
#if LACUNA_REAL_BITS == 16
        if (derivative == LC_WIDE_C(0.0) &&
            compact_coefficients[index] != LC_REAL_C(0.0))
            return LC_ROOT_NONCONVERGENCE;
#endif
        if (!lc_isfinite(derivative) || lc_wide_fabs(derivative) > LC_REAL_MAX) {
            return LC_NUMERIC_ERROR;
        }
        if (derivative != LC_WIDE_C(0.0)) {
            derivative_coefficients[derivative_count] = (lc_real_t)derivative;
            derivative_rates[derivative_count] = compact_rates[index];
            derivative_count++;
        }
    }
    if (derivative_count > 1U) {
        status = lc_multi_exp_roots(
            &context, derivative_coefficients, derivative_rates, derivative_count,
            LC_REAL_C(0.0), horizon, extrema, LC_ANALYTICAL_MAX_STATES, &extrema_count
        );
        if (status != LC_OK) {
            result->iterations = context.iterations_used;
            return status;
        }
    }
    result->extrema_count = extrema_count;
    for (index = 0U; index <= extrema_count; ++index) {
        lc_time_t low = index == 0U ? LC_REAL_C(0.0) : extrema[index - 1U];
        lc_time_t high = index == extrema_count ? horizon : extrema[index];
        lc_real_t g_low;
        lc_real_t g_high;
        status = lc_multi_exp_value(
            limit, compact_coefficients, compact_rates, compact_count, low, &g_low
        );
        if (status != LC_OK) {
            return status;
        }
        status = lc_multi_exp_value(
            limit, compact_coefficients, compact_rates, compact_count, high, &g_high
        );
        if (status != LC_OK) {
            return status;
        }
#if LACUNA_REAL_BITS <= 32
        if (index < extrema_count) {
            status = lc_multi_exp_check_extremum(
                limit, compact_coefficients, compact_rates, compact_count,
                high, LC_REAL_C(0.0)
            );
            if (status != LC_OK) {
                return status;
            }
        }
#endif
        if (g_low > LC_REAL_C(0.0) && g_high <= LC_REAL_C(0.0)) {
            int rising_crossing = g_high < LC_REAL_C(0.0);
            lc_bracket_solution crossing;
            if (!rising_crossing) {
                lc_real_t derivative_high;
                status = lc_multi_exp_value(
                    LC_REAL_C(0.0), derivative_coefficients, derivative_rates,
                    derivative_count, high, &derivative_high
                );
                if (status != LC_OK) {
                    return status;
                }
                rising_crossing = derivative_high < LC_REAL_C(0.0);
            }
            if (!rising_crossing) {
                continue;
            }
            status = lc_multi_exp_bisect_crossing(
                &context, limit, compact_coefficients, compact_rates, compact_count,
                low, high, &crossing
            );
            result->iterations = context.iterations_used;
            result->bracket_low = t_last + crossing.low;
            result->bracket_high = t_last + crossing.high;
            result->residual = crossing.residual;
            result->tolerance = crossing.tolerance;
            if (status != LC_OK) {
                return status;
            }
            result->t_spike = t_last + crossing.root;
            if (!lc_isfinite(result->t_spike) || result->t_spike <= t_last) {
                return LC_NUMERIC_ERROR;
            }
            return LC_OK;
        }
    }
    result->iterations = context.iterations_used;
    return LC_NO_CROSSING;
}

#define LC_EXP_POLY_MAX_BLOCKS (LC_ANALYTICAL_MAX_STATES + 1U)
#define LC_EXP_POLY_MAX_COEFFICIENTS (LC_ANALYTICAL_MAX_STATES + 1U)

typedef struct lc_exp_poly_function {
    uint32_t block_count;
    lc_real_t rates[LC_EXP_POLY_MAX_BLOCKS];
    uint32_t degrees[LC_EXP_POLY_MAX_BLOCKS];
    lc_real_t coefficients[LC_EXP_POLY_MAX_BLOCKS][LC_EXP_POLY_MAX_COEFFICIENTS];
} lc_exp_poly_function;

/* Merge one polynomial block with an existing equal-rate block. */
static int lc_exp_poly_add_block(
    lc_exp_poly_function *function,
    lc_real_t rate,
    const lc_real_t *coefficients,
    uint32_t count
) {
    uint32_t block;
    uint32_t index;
    uint32_t degree;
    if (function == NULL || coefficients == NULL || count == 0U ||
        count > LC_EXP_POLY_MAX_COEFFICIENTS || !lc_isfinite(rate)) {
        return 0;
    }
    degree = count - 1U;
    while (degree > 0U && coefficients[degree] == LC_REAL_C(0.0)) {
        degree--;
    }
    if (degree == 0U && coefficients[0] == LC_REAL_C(0.0)) {
        return 1;
    }
    for (block = 0U; block < function->block_count; ++block) {
        if (function->rates[block] == rate) {
            uint32_t resulting_degree = function->degrees[block] > degree
                ? function->degrees[block]
                : degree;
            for (index = 0U; index <= degree; ++index) {
                lc_wide_t combined =
                    (lc_wide_t)function->coefficients[block][index] +
                    (lc_wide_t)coefficients[index];
                if (!lc_isfinite(combined) || lc_wide_fabs(combined) > LC_REAL_MAX) {
                    return 0;
                }
                function->coefficients[block][index] = (lc_real_t)combined;
            }
            while (resulting_degree > 0U &&
                   function->coefficients[block][resulting_degree] == LC_REAL_C(0.0)) {
                resulting_degree--;
            }
            function->degrees[block] = resulting_degree;
            return 1;
        }
    }
    if (function->block_count >= LC_EXP_POLY_MAX_BLOCKS) {
        return 0;
    }
    block = function->block_count++;
    function->rates[block] = rate;
    function->degrees[block] = degree;
    for (index = 0U; index <= degree; ++index) {
        function->coefficients[block][index] = coefficients[index];
    }
    return 1;
}

/* Evaluate one polynomial coefficient block with Horner's method. */
static lc_status lc_exp_poly_polynomial(
    const lc_exp_poly_function *function,
    uint32_t block,
    lc_time_t delta,
    lc_wide_t *value,
    lc_wide_t *magnitude
) {
    uint32_t degree;
    lc_wide_t polynomial;
    lc_wide_t bound;
    if (function == NULL || value == NULL || magnitude == NULL ||
        block >= function->block_count || !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    degree = function->degrees[block];
    polynomial = (lc_wide_t)function->coefficients[block][degree];
    bound = lc_wide_fabs(polynomial);
    while (degree > 0U) {
        degree--;
        polynomial = polynomial * (lc_wide_t)delta +
            (lc_wide_t)function->coefficients[block][degree];
        bound = bound * (lc_wide_t)delta +
            lc_wide_fabs((lc_wide_t)function->coefficients[block][degree]);
    }
    if (!lc_isfinite(polynomial) || !lc_isfinite(bound)) {
        return LC_NUMERIC_ERROR;
    }
    *value = polynomial;
    *magnitude = bound;
    return LC_OK;
}

/* Evaluate the complete exponential-polynomial trajectory. */
static lc_status lc_exp_poly_value(
    const lc_exp_poly_function *function,
    lc_time_t delta,
    lc_real_t *value
) {
    uint32_t block;
    lc_real_t base;
    lc_wide_t total = LC_WIDE_C(0.0);
    lc_wide_t magnitude = LC_WIDE_C(0.0);
    if (function == NULL || value == NULL || !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    if (function->block_count == 0U) {
        *value = LC_REAL_C(0.0);
        return LC_OK;
    }
    base = function->rates[0];
    for (block = 1U; block < function->block_count; ++block) {
        if (function->rates[block] > base) {
            base = function->rates[block];
        }
    }
    for (block = 0U; block < function->block_count; ++block) {
        lc_wide_t polynomial;
        lc_wide_t polynomial_bound;
        lc_wide_t scale;
        lc_status status = lc_exp_poly_polynomial(
            function, block, delta, &polynomial, &polynomial_bound
        );
        if (status != LC_OK) {
            return status;
        }
        scale = lc_wide_exp(
            ((lc_wide_t)function->rates[block] - (lc_wide_t)base) *
            (lc_wide_t)delta
        );
        total += polynomial * scale;
        magnitude += polynomial_bound * scale;
    }
#if LACUNA_REAL_BITS == 64
    if (lc_wide_fabs(total) <= LC_WIDE_C(32.0) * LC_REAL_EPSILON * magnitude) {
        total = LC_WIDE_C(0.0);
    }
#else
    if (!lc_isfinite(magnitude)) {
        return LC_ROOT_NONCONVERGENCE;
    }
#endif
    if (!lc_isfinite(total) || lc_wide_fabs(total) > LC_REAL_MAX) {
        return LC_NUMERIC_ERROR;
    }
    *value = (lc_real_t)total;
    return lc_isfinite(*value) ? LC_OK : LC_NUMERIC_ERROR;
}

#if LACUNA_REAL_BITS <= 32
static lc_status lc_exp_poly_check_extremum(
    const lc_exp_poly_function *function, lc_time_t delta
) {
    lc_real_t base = function->rates[0];
    lc_wide_t total = LC_WIDE_C(0.0);
    lc_wide_t magnitude = LC_WIDE_C(0.0);
    uint32_t block;
    for (block = 1U; block < function->block_count; ++block) {
        if (function->rates[block] > base) {
            base = function->rates[block];
        }
    }
    for (block = 0U; block < function->block_count; ++block) {
        lc_wide_t polynomial;
        lc_wide_t bound;
        lc_wide_t scale;
        lc_status status = lc_exp_poly_polynomial(
            function, block, delta, &polynomial, &bound
        );
        if (status != LC_OK) {
            return status;
        }
        scale = lc_wide_exp(
            ((lc_wide_t)function->rates[block] - (lc_wide_t)base) * (lc_wide_t)delta
        );
        total += polynomial * scale;
        magnitude += bound * scale;
    }
    if (!lc_isfinite(total) || !lc_isfinite(magnitude) ||
        lc_wide_fabs(total) <= LC_WIDE_C(64.0) * LC_WIDE_EPSILON * magnitude) {
        return LC_ROOT_NONCONVERGENCE;
    }
    return LC_OK;
}
#endif

/* Apply (D - base_rate), the generalized-Rolle reduction operator. */
static lc_status lc_exp_poly_reduce(
    const lc_exp_poly_function *source,
    lc_real_t base_rate,
    lc_exp_poly_function *destination
) {
    uint32_t block;
    if (source == NULL || destination == NULL || !lc_isfinite(base_rate)) {
        return LC_INVALID_ARGUMENT;
    }
    memset(destination, 0, sizeof(*destination));
    for (block = 0U; block < source->block_count; ++block) {
        lc_real_t coefficients[LC_EXP_POLY_MAX_COEFFICIENTS] = {LC_REAL_C(0.0)};
        uint32_t degree = source->degrees[block];
        uint32_t index;
        for (index = 0U; index <= degree; ++index) {
            lc_wide_t reduced =
                ((lc_wide_t)source->rates[block] - (lc_wide_t)base_rate) *
                (lc_wide_t)source->coefficients[block][index];
#if LACUNA_REAL_BITS == 16
            if (reduced == LC_WIDE_C(0.0) && source->rates[block] != base_rate &&
                source->coefficients[block][index] != LC_REAL_C(0.0))
                return LC_ROOT_NONCONVERGENCE;
#endif
            if (index < degree) {
                reduced += (lc_wide_t)(index + 1U) *
                    (lc_wide_t)source->coefficients[block][index + 1U];
            }
            if (!lc_isfinite(reduced) || lc_wide_fabs(reduced) > LC_REAL_MAX) {
                return LC_NUMERIC_ERROR;
            }
            coefficients[index] = (lc_real_t)reduced;
        }
        if (!lc_exp_poly_add_block(
                destination, source->rates[block], coefficients, degree + 1U)) {
            return LC_NUMERIC_ERROR;
        }
    }
    return LC_OK;
}

static uint32_t lc_exp_poly_term_count(const lc_exp_poly_function *function) {
    uint32_t block;
    uint32_t count = 0U;
    if (function == NULL) {
        return 0U;
    }
    for (block = 0U; block < function->block_count; ++block) {
        count += function->degrees[block] + 1U;
    }
    return count;
}

/* Add one distinct root to a bounded exponential-polynomial root list. */
static int lc_exp_poly_add_root(
    lc_multi_exp_context *context,
    lc_time_t *roots,
    uint32_t *root_count,
    uint32_t capacity,
    lc_time_t root,
    lc_time_t low,
    lc_time_t high
) {
#if LACUNA_REAL_BITS <= 32
    (void)context;
    if (root <= low || root >= high) {
        return 1;
    }
    if (*root_count > 0U && roots[*root_count - 1U] == root) {
        return 1;
    }
#else
    lc_time_t tolerance = lc_multi_exp_isolation_tolerance(context, root);
    if (root <= low + tolerance || root >= high - tolerance) {
        return 1;
    }
    if (*root_count > 0U && lc_time_fabs(roots[*root_count - 1U] - root) <= tolerance) {
        return 1;
    }
#endif
    if (*root_count >= capacity) {
        return 0;
    }
    roots[(*root_count)++] = root;
    return 1;
}

/* Refine a root isolated by the generalized Rolle recursion. */
static lc_status lc_exp_poly_bisect_isolated(
    lc_multi_exp_context *context,
    const lc_exp_poly_function *function,
    lc_time_t low,
    lc_time_t high,
    lc_time_t *root
) {
    lc_real_t f_low;
    lc_real_t f_high;
    lc_status status;
    if (context == NULL || function == NULL || root == NULL || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    status = lc_exp_poly_value(function, low, &f_low);
    if (status != LC_OK) {
        return status;
    }
    status = lc_exp_poly_value(function, high, &f_high);
    if (status != LC_OK) {
        return status;
    }
    if (f_low == LC_REAL_C(0.0)) {
        *root = low;
        return LC_OK;
    }
    if (f_high == LC_REAL_C(0.0)) {
        *root = high;
        return LC_OK;
    }
    if (lc_same_sign(f_low, f_high)) {
        return LC_INVALID_ARGUMENT;
    }
    while (high - low > lc_multi_exp_isolation_tolerance(
            context, low + LC_REAL_C(0.5) * (high - low))) {
        lc_time_t midpoint;
        lc_real_t f_midpoint;
        if (context->iterations_used >= context->iteration_cap) {
            *root = low + LC_REAL_C(0.5) * (high - low);
            return LC_ROOT_NONCONVERGENCE;
        }
        context->iterations_used++;
        midpoint = low + LC_REAL_C(0.5) * (high - low);
        status = lc_exp_poly_value(function, midpoint, &f_midpoint);
        if (status != LC_OK) {
            return status;
        }
        if (f_midpoint == LC_REAL_C(0.0)) {
            *root = midpoint;
            return LC_OK;
        }
        if (lc_same_sign(f_low, f_midpoint)) {
            low = midpoint;
            f_low = f_midpoint;
        } else {
            high = midpoint;
            f_high = f_midpoint;
        }
    }
    *root = low + LC_REAL_C(0.5) * (high - low);
    return LC_OK;
}

/*
 * Generalized Rolle recursion: for H=(D-r)F, the derivative of lc_real_exp(-rt)F
 * has the sign of H. Choosing r from one block lowers the total polynomial
 * coefficient count by one, so the recursion is finite and bounded.
 */
static lc_status lc_exp_poly_roots(
    lc_multi_exp_context *context,
    const lc_exp_poly_function *function,
    lc_time_t low,
    lc_time_t high,
    lc_time_t *roots,
    uint32_t capacity,
    uint32_t *root_count
) {
    lc_exp_poly_function reduced;
    lc_time_t critical[LC_ANALYTICAL_MAX_STATES];
    uint32_t critical_count = 0U;
    uint32_t index;
    uint32_t term_count;
    lc_real_t base_rate;
    lc_status status;
    if (context == NULL || function == NULL || roots == NULL || root_count == NULL ||
        !lc_isfinite(low) || !lc_isfinite(high) || low < LC_REAL_C(0.0) || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    *root_count = 0U;
    term_count = lc_exp_poly_term_count(function);
    if (term_count <= 1U || high == low) {
        return LC_OK;
    }
    base_rate = function->rates[function->block_count - 1U];
    status = lc_exp_poly_reduce(function, base_rate, &reduced);
    if (status != LC_OK) {
        return status;
    }
    if (lc_exp_poly_term_count(&reduced) >= term_count) {
        return LC_NUMERIC_ERROR;
    }
    if (lc_exp_poly_term_count(&reduced) > 1U) {
        status = lc_exp_poly_roots(
            context, &reduced, low, high, critical,
            LC_ANALYTICAL_MAX_STATES, &critical_count
        );
        if (status != LC_OK) {
            return status;
        }
    }
    for (index = 0U; index <= critical_count; ++index) {
        lc_time_t interval_low = index == 0U ? low : critical[index - 1U];
        lc_time_t interval_high = index == critical_count ? high : critical[index];
        lc_real_t f_low;
        lc_real_t f_high;
        status = lc_exp_poly_value(function, interval_low, &f_low);
        if (status != LC_OK) {
            return status;
        }
        status = lc_exp_poly_value(function, interval_high, &f_high);
        if (status != LC_OK) {
            return status;
        }
#if LACUNA_REAL_BITS <= 32
        if (index < critical_count) {
            status = lc_exp_poly_check_extremum(function, interval_high);
            if (status != LC_OK) {
                return status;
            }
        }
#endif
        if (f_low == LC_REAL_C(0.0) && !lc_exp_poly_add_root(
                context, roots, root_count, capacity,
                interval_low, low, high)) {
            return LC_ROOT_NONCONVERGENCE;
        }
        if (f_high == LC_REAL_C(0.0) && !lc_exp_poly_add_root(
                context, roots, root_count, capacity,
                interval_high, low, high)) {
            return LC_ROOT_NONCONVERGENCE;
        }
        if (f_low != LC_REAL_C(0.0) && f_high != LC_REAL_C(0.0) && !lc_same_sign(f_low, f_high)) {
            lc_time_t root;
            status = lc_exp_poly_bisect_isolated(
                context, function, interval_low, interval_high, &root
            );
            if (status != LC_OK) {
                return status;
            }
            if (!lc_exp_poly_add_root(
                    context, roots, root_count, capacity, root, low, high)) {
                return LC_ROOT_NONCONVERGENCE;
            }
        }
    }
    return LC_OK;
}

/* Bound the remaining modal envelope over a finite interval. */
static lc_status lc_exp_poly_envelope(
    const lc_exp_poly_function *function,
    lc_real_t excluded_rate,
    int exclude_rate,
    lc_time_t delta,
    lc_wide_t *envelope
) {
    uint32_t block;
    lc_wide_t total = LC_WIDE_C(0.0);
    if (function == NULL || envelope == NULL || !lc_isfinite(delta) || delta < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    for (block = 0U; block < function->block_count; ++block) {
        lc_wide_t polynomial;
        lc_wide_t bound;
        lc_wide_t term;
        lc_status status;
        if (exclude_rate && function->rates[block] == excluded_rate) {
            continue;
        }
        status = lc_exp_poly_polynomial(
            function, block, delta, &polynomial, &bound
        );
        if (status != LC_OK) {
            return status;
        }
        term = bound * lc_wide_exp(
            (lc_wide_t)function->rates[block] * (lc_wide_t)delta
        );
        total += term;
    }
    if (!lc_isfinite(total)) {
        return LC_NUMERIC_ERROR;
    }
    *envelope = total;
    return LC_OK;
}

/* Refine a rising crossing after extrema partition the horizon. */
static lc_status lc_exp_poly_bisect_crossing(
    lc_multi_exp_context *context,
    const lc_exp_poly_function *function,
    lc_time_t low,
    lc_time_t high,
    lc_bracket_solution *solution
) {
    lc_real_t f_low;
    lc_real_t f_high;
    lc_status status;
    if (context == NULL || function == NULL || solution == NULL || high < low) {
        return LC_INVALID_ARGUMENT;
    }
    memset(solution, 0, sizeof(*solution));
    status = lc_exp_poly_value(function, low, &f_low);
    if (status != LC_OK) {
        return status;
    }
    status = lc_exp_poly_value(function, high, &f_high);
    if (status != LC_OK) {
        return status;
    }
    if (!(f_low > LC_REAL_C(0.0) && f_high < LC_REAL_C(0.0))) {
        return LC_INVALID_ARGUMENT;
    }
    while (high - low > lc_multi_exp_tolerance(
            context, low + LC_REAL_C(0.5) * (high - low))) {
        lc_time_t midpoint;
        lc_real_t f_midpoint;
        if (context->iterations_used >= context->iteration_cap) {
            solution->root = low + LC_REAL_C(0.5) * (high - low);
            solution->low = low;
            solution->high = high;
            solution->tolerance = lc_multi_exp_tolerance(context, solution->root);
            lc_exp_poly_value(function, solution->root, &solution->residual);
            return LC_ROOT_NONCONVERGENCE;
        }
        context->iterations_used++;
        solution->iterations++;
        midpoint = low + LC_REAL_C(0.5) * (high - low);
        status = lc_exp_poly_value(function, midpoint, &f_midpoint);
        if (status != LC_OK) {
            return status;
        }
        if (f_midpoint > LC_REAL_C(0.0)) {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    solution->root = low + LC_REAL_C(0.5) * (high - low);
    solution->low = low;
    solution->high = high;
    solution->tolerance = lc_multi_exp_tolerance(context, solution->root);
    return lc_exp_poly_value(function, solution->root, &solution->residual);
}

/* Find the first crossing of a bounded stable exponential polynomial. */
lc_status lc_expr_exp_poly_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_exp_poly_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    lc_time_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_exp_poly_function function;
    lc_exp_poly_function derivative;
    lc_multi_exp_context context;
    lc_time_t roots[LC_ANALYTICAL_MAX_STATES];
    lc_time_t extrema[LC_ANALYTICAL_MAX_STATES];
    lc_real_t limit;
    lc_real_t g_zero;
    lc_time_t horizon;
    uint32_t root_count = 0U;
    uint32_t extrema_count = 0U;
    uint32_t block;
    uint32_t index;
    lc_status status;
    if (nodes == NULL || hint == NULL || state == NULL || result == NULL ||
        variables == NULL || workspace == NULL || state_count == 0U ||
        state_count > LC_ANALYTICAL_MAX_STATES ||
        variable_count != state_count + 1U || hint->block_count == 0U ||
        hint->block_count > LC_ANALYTICAL_MAX_STATES ||
        hint->coefficient_count == 0U ||
        hint->coefficient_count > LC_ANALYTICAL_MAX_STATES ||
        hint->iteration_cap == 0U || hint->iteration_cap > 4096U ||
        !lc_isfinite(t_last) || t_last < LC_REAL_C(0.0) ||
        !lc_isfinite(hint->relative_tolerance) || hint->relative_tolerance <= LC_REAL_C(0.0) ||
        !lc_isfinite(hint->fastest_time_constant) ||
        hint->fastest_time_constant <= LC_REAL_C(0.0) || hint->limit_root >= node_count ||
        hint->coefficient_offsets[0] != 0U ||
        hint->coefficient_offsets[hint->block_count] != hint->coefficient_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
        variables[index + 1U] = state[index];
    }
    variables[0] = LC_REAL_C(0.0);
    for (block = 0U; block < hint->block_count; ++block) {
        if (hint->rate_roots[block] >= node_count ||
            hint->coefficient_offsets[block] >=
                hint->coefficient_offsets[block + 1U] ||
            hint->coefficient_offsets[block + 1U] > hint->coefficient_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (index = 0U; index < hint->coefficient_count; ++index) {
        if (hint->coefficient_roots[index] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    status = lc_expr_evaluate(
        nodes, node_count, parameters, parameter_count, variables, variable_count,
        workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    memset(&function, 0, sizeof(function));
    limit = workspace[hint->limit_root];
    if (!lc_isfinite(limit)) {
        return LC_NUMERIC_ERROR;
    }
    if (limit != LC_REAL_C(0.0) && !lc_exp_poly_add_block(&function, LC_REAL_C(0.0), &limit, 1U)) {
        return LC_NUMERIC_ERROR;
    }
    for (block = 0U; block < hint->block_count; ++block) {
        lc_real_t coefficients[LC_EXP_POLY_MAX_COEFFICIENTS] = {LC_REAL_C(0.0)};
        lc_real_t rate = workspace[hint->rate_roots[block]];
        uint32_t begin = hint->coefficient_offsets[block];
        uint32_t end = hint->coefficient_offsets[block + 1U];
        if (!lc_isfinite(rate) || rate >= LC_REAL_C(0.0)) {
            return LC_UNSUPPORTED_MODEL;
        }
        for (index = begin; index < end; ++index) {
            coefficients[index - begin] = workspace[hint->coefficient_roots[index]];
            if (!lc_isfinite(coefficients[index - begin])) {
                return LC_NUMERIC_ERROR;
            }
        }
        if (!lc_exp_poly_add_block(
                &function, rate, coefficients, end - begin)) {
            return LC_NUMERIC_ERROR;
        }
    }
    memset(result, 0, sizeof(*result));
    memset(&context, 0, sizeof(context));
    context.t_last = t_last;
    context.relative_tolerance = hint->relative_tolerance;
    context.fastest_time_constant = hint->fastest_time_constant;
    context.iteration_cap = hint->iteration_cap;
    status = lc_exp_poly_value(&function, LC_REAL_C(0.0), &g_zero);
    if (status != LC_OK) {
        return status;
    }
    if (g_zero <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    if (function.block_count == (limit == LC_REAL_C(0.0) ? 0U : 1U)) {
        return LC_NO_CROSSING;
    }
    horizon = hint->fastest_time_constant;
    for (block = 0U; block < function.block_count; ++block) {
        if (function.rates[block] < LC_REAL_C(0.0)) {
            horizon = lc_time_fmax(horizon, -LC_REAL_C(1.0) / function.rates[block]);
            /* Past d/|r| every |c_k| t^k lc_real_exp(rt), k<=d, is nonincreasing. */
            horizon = lc_time_fmax(
                horizon,
                -(lc_real_t)function.degrees[block] / function.rates[block]
            );
        }
    }
    if (limit == LC_REAL_C(0.0) && function.block_count > 1U) {
        uint32_t dominant = 0U;
        for (block = 1U; block < function.block_count; ++block) {
            if (function.rates[block] > function.rates[dominant]) {
                dominant = block;
            }
        }
        for (block = 0U; block < function.block_count; ++block) {
            if (block != dominant &&
                function.degrees[block] > function.degrees[dominant]) {
                horizon = lc_time_fmax(
                    horizon,
                    (lc_real_t)(
                        function.degrees[block] - function.degrees[dominant]
                    ) / (function.rates[dominant] - function.rates[block])
                );
            }
        }
    }
    for (index = 0U; index < 128U; ++index) {
        lc_wide_t envelope;
        if (limit != LC_REAL_C(0.0)) {
            status = lc_exp_poly_envelope(
                &function, LC_REAL_C(0.0), 1, horizon, &envelope
            );
            if (status != LC_OK) {
                return status;
            }
            if (envelope < lc_wide_fabs((lc_wide_t)limit)) {
                break;
            }
        } else {
            uint32_t dominant = 0U;
            uint32_t degree;
            lc_wide_t leading;
            lc_wide_t lower = LC_WIDE_C(0.0);
            lc_wide_t normalized_envelope = LC_WIDE_C(0.0);
            lc_wide_t separation;
            lc_wide_t uncertainty;
            lc_wide_t comparison_scale;
            for (block = 1U; block < function.block_count; ++block) {
                if (function.rates[block] > function.rates[dominant]) {
                    dominant = block;
                }
            }
            degree = function.degrees[dominant];
            leading = lc_wide_fabs(
                (lc_wide_t)function.coefficients[dominant][degree]
            ) * lc_wide_pow((lc_wide_t)horizon, (lc_wide_t)degree);
            while (degree > 0U) {
                degree--;
                lower = lower * (lc_wide_t)horizon + lc_wide_fabs(
                    (lc_wide_t)function.coefficients[dominant][degree]
                );
            }
            for (block = 0U; block < function.block_count; ++block) {
                lc_wide_t polynomial;
                lc_wide_t bound;
                if (block == dominant) {
                    continue;
                }
                status = lc_exp_poly_polynomial(
                    &function, block, horizon, &polynomial, &bound
                );
                if (status != LC_OK) {
                    return status;
                }
                normalized_envelope += bound * lc_wide_exp(
                    ((lc_wide_t)function.rates[block] -
                     (lc_wide_t)function.rates[dominant]) *
                    (lc_wide_t)horizon
                );
            }
            if (!lc_isfinite(leading) || !lc_isfinite(lower) ||
                !lc_isfinite(normalized_envelope)) {
                return LC_NUMERIC_ERROR;
            }
            /*
             * Past the ratio peaks above, both bounds only separate further.
             * The coefficients entered this calculation as doubles.  Require
             * separation beyond their rounding uncertainty so x87 extended
             * intermediates cannot turn a mathematical boundary equality into
             * a platform-dependent strict inequality.
            */
            separation = leading - lower;
            comparison_scale = lc_wide_fmax(
                leading, lc_wide_fmax(lower, normalized_envelope)
            );
            uncertainty = (LC_WIDE_C(96.0) * (lc_wide_t)LC_REAL_EPSILON) *
                comparison_scale;
            if (separation > uncertainty &&
                normalized_envelope + uncertainty < separation) {
                break;
            }
        }
        horizon *= LC_REAL_C(2.0);
        if (!lc_isfinite(horizon)) {
            return LC_NUMERIC_ERROR;
        }
    }
    if (index == 128U) {
        return LC_ROOT_NONCONVERGENCE;
    }
    result->horizon = t_last + horizon;
    if (!lc_isfinite(result->horizon)) {
        return LC_NUMERIC_ERROR;
    }
    status = lc_exp_poly_reduce(&function, LC_REAL_C(0.0), &derivative);
    if (status != LC_OK) {
        return status;
    }
    if (lc_exp_poly_term_count(&derivative) > 1U) {
        status = lc_exp_poly_roots(
            &context, &derivative, LC_REAL_C(0.0), horizon, extrema,
            LC_ANALYTICAL_MAX_STATES, &extrema_count
        );
        if (status != LC_OK) {
            result->iterations = context.iterations_used;
            return status;
        }
    }
    result->extrema_count = extrema_count;
    status = lc_exp_poly_roots(
        &context, &function, LC_REAL_C(0.0), horizon, roots,
        LC_ANALYTICAL_MAX_STATES, &root_count
    );
    if (status != LC_OK) {
        result->iterations = context.iterations_used;
        return status;
    }
    for (index = 0U; index < root_count; ++index) {
        lc_time_t low = index == 0U ? LC_REAL_C(0.0) :
            roots[index - 1U] + LC_REAL_C(0.5) * (roots[index] - roots[index - 1U]);
        lc_time_t high = index + 1U == root_count ? horizon :
            roots[index] + LC_REAL_C(0.5) * (roots[index + 1U] - roots[index]);
        lc_real_t f_low;
        lc_real_t f_high;
        lc_bracket_solution crossing;
        status = lc_exp_poly_value(&function, low, &f_low);
        if (status != LC_OK) {
            return status;
        }
        status = lc_exp_poly_value(&function, high, &f_high);
        if (status != LC_OK) {
            return status;
        }
        if (!(f_low > LC_REAL_C(0.0) && f_high < LC_REAL_C(0.0))) {
            continue;
        }
        status = lc_exp_poly_bisect_crossing(
            &context, &function, low, high, &crossing
        );
        result->iterations = context.iterations_used;
        result->bracket_low = t_last + crossing.low;
        result->bracket_high = t_last + crossing.high;
        result->residual = crossing.residual;
        result->tolerance = crossing.tolerance;
        if (status != LC_OK) {
            return status;
        }
        result->t_spike = t_last + crossing.root;
        if (!lc_isfinite(result->t_spike) || result->t_spike <= t_last) {
            return LC_NUMERIC_ERROR;
        }
        return LC_OK;
    }
    result->iterations = context.iterations_used;
    return LC_NO_CROSSING;
}

/* Adapt the legacy alpha interface to the general exponential-polynomial solver. */
lc_status lc_expr_alpha_predict(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_root_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    lc_time_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_alpha_root_context context;
    uint32_t roots[11];
    uint32_t index;
    lc_real_t g_zero;
    lc_real_t unused_derivative;
    lc_real_t asymptote;
    lc_real_t membrane_coefficient;
    lc_real_t synapse_constant;
    lc_real_t synapse_linear;
    lc_real_t membrane_rate;
    lc_real_t synapse_rate;
    lc_real_t threshold;
    lc_real_t extremum_a;
    lc_real_t extremum_d;
    lc_real_t extremum_e;
    lc_real_t extremum_p;
    lc_time_t turn = LC_REAL_C(0.0);
    int has_turn = 0;
    lc_time_t horizon;
    lc_time_t extrema[2];
    uint32_t extrema_count = 0;
    lc_time_t partition[3];
    uint32_t partition_count;
    lc_status status;

    if (nodes == NULL || hint == NULL || state == NULL || result == NULL ||
        variables == NULL || workspace == NULL || state_count != 3U ||
        variable_count != 4U || !lc_isfinite(t_last) || t_last < LC_REAL_C(0.0) ||
        !lc_isfinite(hint->relative_tolerance) || hint->relative_tolerance <= LC_REAL_C(0.0) ||
        !lc_isfinite(hint->fastest_time_constant) || hint->fastest_time_constant <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    roots[0] = hint->g_root;
    roots[1] = hint->g_prime_root;
    roots[2] = hint->extremum_root;
    roots[3] = hint->extremum_prime_root;
    roots[4] = hint->asymptote_root;
    roots[5] = hint->membrane_coefficient_root;
    roots[6] = hint->synapse_constant_root;
    roots[7] = hint->synapse_linear_root;
    roots[8] = hint->membrane_rate_root;
    roots[9] = hint->synapse_rate_root;
    roots[10] = hint->threshold_root;
    for (index = 0; index < 11U; ++index) {
        if (roots[index] >= node_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (index = 0; index < state_count; ++index) {
        if (!lc_isfinite(state[index])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    memset(result, 0, sizeof(*result));
    memset(&context, 0, sizeof(context));
    context.nodes = nodes;
    context.node_count = node_count;
    context.parameters = parameters;
    context.parameter_count = parameter_count;
    context.state = state;
    context.state_count = state_count;
    context.t_last = t_last;
    context.fastest_time_constant = hint->fastest_time_constant;
    context.relative_tolerance = hint->relative_tolerance;
    context.variables = variables;
    context.variable_count = variable_count;
    context.workspace = workspace;
    context.workspace_count = workspace_count;

    status = lc_alpha_evaluate(
        &context, LC_REAL_C(0.0), hint->g_root, hint->g_prime_root,
        &g_zero, &unused_derivative
    );
    if (status != LC_OK) {
        return status;
    }
    asymptote = workspace[hint->asymptote_root];
    membrane_coefficient = workspace[hint->membrane_coefficient_root];
    synapse_constant = workspace[hint->synapse_constant_root];
    synapse_linear = workspace[hint->synapse_linear_root];
    membrane_rate = workspace[hint->membrane_rate_root];
    synapse_rate = workspace[hint->synapse_rate_root];
    threshold = workspace[hint->threshold_root];
    if (!lc_isfinite(asymptote) || !lc_isfinite(membrane_coefficient) ||
        !lc_isfinite(synapse_constant) || !lc_isfinite(synapse_linear) ||
        !lc_isfinite(membrane_rate) || !lc_isfinite(synapse_rate) ||
        !lc_isfinite(threshold) || membrane_rate >= LC_REAL_C(0.0) || synapse_rate >= LC_REAL_C(0.0) ||
        membrane_rate == synapse_rate) {
        return LC_UNSUPPORTED_MODEL;
    }
    if (g_zero <= LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }

    extremum_a = membrane_rate * membrane_coefficient;
    extremum_d = synapse_linear + synapse_rate * synapse_constant;
    extremum_e = synapse_rate * synapse_linear;
    extremum_p = membrane_rate - synapse_rate;
#if LACUNA_REAL_BITS == 16
    if ((extremum_a == LC_REAL_C(0.0) && membrane_coefficient != LC_REAL_C(0.0)) ||
        (synapse_rate * synapse_constant == LC_REAL_C(0.0) &&
         synapse_constant != LC_REAL_C(0.0)) ||
        (extremum_e == LC_REAL_C(0.0) && synapse_linear != LC_REAL_C(0.0)) ||
        (extremum_a != LC_REAL_C(0.0) && extremum_p != LC_REAL_C(0.0) &&
         extremum_a * extremum_p == LC_REAL_C(0.0))) {
        return LC_ROOT_NONCONVERGENCE;
    }
#endif
    if (!lc_isfinite(extremum_a) || !lc_isfinite(extremum_d) ||
        !lc_isfinite(extremum_e) || !lc_isfinite(extremum_p) ||
        !lc_isfinite(extremum_a * extremum_p)) {
        return LC_NUMERIC_ERROR;
    }
    if (extremum_a * extremum_p != LC_REAL_C(0.0)) {
        lc_real_t ratio = -extremum_e / (extremum_a * extremum_p);
#if LACUNA_REAL_BITS == 16
        if (!lc_isfinite(ratio) ||
            (ratio == LC_REAL_C(0.0) && extremum_e != LC_REAL_C(0.0)))
            return LC_ROOT_NONCONVERGENCE;
#endif
        if (lc_isfinite(ratio) && ratio > LC_REAL_C(0.0)) {
            lc_time_t candidate = lc_real_log(ratio) / extremum_p;
            if (lc_isfinite(candidate) && candidate > LC_REAL_C(0.0)) {
                turn = candidate;
                has_turn = 1;
            }
        }
    }

    horizon = lc_time_fmax(-LC_REAL_C(1.0) / membrane_rate, -LC_REAL_C(1.0) / synapse_rate);
    if (!lc_isfinite(horizon) || horizon <= LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    if (asymptote < threshold) {
        lc_real_t margin = threshold - asymptote;
        for (index = 0; index < 128U; ++index) {
            lc_real_t ea = lc_real_exp(membrane_rate * horizon);
            lc_real_t eq = lc_real_exp(synapse_rate * horizon);
            lc_real_t envelope = lc_real_fabs(membrane_coefficient) * ea +
                (lc_real_fabs(synapse_constant) + lc_real_fabs(synapse_linear) * horizon) * eq;
            if (!lc_isfinite(envelope)) {
                return LC_NUMERIC_ERROR;
            }
            if (envelope < margin) {
                break;
            }
            horizon *= LC_REAL_C(2.0);
            if (!lc_isfinite(horizon)) {
                return LC_NUMERIC_ERROR;
            }
        }
        if (index == 128U) {
            return LC_ROOT_NONCONVERGENCE;
        }
    } else if (asymptote > threshold) {
        for (index = 0; index < 128U; ++index) {
            lc_real_t g_horizon;
            status = lc_alpha_evaluate(
                &context, horizon, hint->g_root, hint->g_prime_root,
                &g_horizon, &unused_derivative
            );
            if (status != LC_OK) {
                return status;
            }
            if (g_horizon < LC_REAL_C(0.0)) {
                break;
            }
            horizon *= LC_REAL_C(2.0);
            if (!lc_isfinite(horizon)) {
                return LC_NUMERIC_ERROR;
            }
        }
        if (index == 128U) {
            return LC_ROOT_NONCONVERGENCE;
        }
    } else {
        int tail_sign;
        if (has_turn && turn > horizon) {
            horizon = turn;
        }
        if (extremum_p > LC_REAL_C(0.0) && extremum_a != LC_REAL_C(0.0)) {
            tail_sign = lc_value_sign(extremum_a);
        } else if (extremum_e != LC_REAL_C(0.0)) {
            tail_sign = lc_value_sign(extremum_e);
        } else if (extremum_d != LC_REAL_C(0.0)) {
            tail_sign = lc_value_sign(extremum_d);
        } else {
            tail_sign = lc_value_sign(extremum_a);
        }
        if (tail_sign != 0) {
            for (index = 0; index < 128U; ++index) {
                lc_real_t f_horizon;
                status = lc_alpha_evaluate(
                    &context, horizon, hint->extremum_root,
                    hint->extremum_prime_root, &f_horizon, &unused_derivative
                );
                if (status != LC_OK) {
                    return status;
                }
                if (lc_value_sign(f_horizon) == tail_sign) {
                    break;
                }
                horizon *= LC_REAL_C(2.0);
                if (!lc_isfinite(horizon)) {
                    return LC_NUMERIC_ERROR;
                }
            }
            if (index == 128U) {
                return LC_ROOT_NONCONVERGENCE;
            }
        }
    }
    result->horizon = t_last + horizon;
    if (!lc_isfinite(result->horizon)) {
        return LC_NUMERIC_ERROR;
    }

    partition[0] = LC_REAL_C(0.0);
    partition_count = 1U;
    if (has_turn && turn < horizon) {
        partition[partition_count++] = turn;
    }
    partition[partition_count++] = horizon;
    for (index = 0; index + 1U < partition_count; ++index) {
        lc_time_t low = partition[index];
        lc_time_t high = partition[index + 1U];
        lc_real_t f_low;
        lc_real_t f_high;
        lc_bracket_solution extremum_solution;
        status = lc_alpha_evaluate(
            &context, low, hint->extremum_root, hint->extremum_prime_root,
            &f_low, &unused_derivative
        );
        if (status != LC_OK) {
            return status;
        }
        status = lc_alpha_evaluate(
            &context, high, hint->extremum_root, hint->extremum_prime_root,
            &f_high, &unused_derivative
        );
        if (status != LC_OK) {
            return status;
        }
        if (f_low == LC_REAL_C(0.0) && !lc_add_extremum(
                extrema, &extrema_count, low, horizon,
                lc_root_tolerance(&context, low))) {
            return LC_NUMERIC_ERROR;
        }
        if (!lc_same_sign(f_low, f_high) && f_low != LC_REAL_C(0.0) && f_high != LC_REAL_C(0.0)) {
            status = lc_solve_bracket(
                &context, hint->extremum_root, hint->extremum_prime_root,
                low, high, &extremum_solution
            );
            result->iterations += extremum_solution.iterations;
            if (status != LC_OK) {
                result->bracket_low = t_last + extremum_solution.low;
                result->bracket_high = t_last + extremum_solution.high;
                result->residual = extremum_solution.residual;
                result->tolerance = extremum_solution.tolerance;
                return status;
            }
            if (!lc_add_extremum(
                    extrema, &extrema_count, extremum_solution.root, horizon,
                    extremum_solution.tolerance)) {
                return LC_NUMERIC_ERROR;
            }
        }
        if (f_high == LC_REAL_C(0.0) && !lc_add_extremum(
                extrema, &extrema_count, high, horizon,
                lc_root_tolerance(&context, high))) {
            return LC_NUMERIC_ERROR;
        }
    }
    result->extrema_count = extrema_count;

    {
        lc_time_t left = LC_REAL_C(0.0);
        lc_real_t g_left = g_zero;
        for (index = 0; index <= extrema_count; ++index) {
            lc_time_t right = index < extrema_count ? extrema[index] : horizon;
            lc_real_t g_right;
            lc_real_t g_prime_right;
            lc_real_t classified_left = g_left;
            lc_real_t classified_right;
            status = lc_alpha_evaluate(
                &context, right, hint->g_root, hint->g_prime_root,
                &g_right, &g_prime_right
            );
            if (status != LC_OK) {
                return status;
            }
            status = lc_alpha_classified_crossing_value(
                threshold, asymptote, membrane_coefficient, synapse_constant,
                synapse_linear, membrane_rate, synapse_rate, right,
                &classified_right
            );
            if (status != LC_OK) {
                return status;
            }
            if (classified_left > LC_REAL_C(0.0) && classified_right < LC_REAL_C(0.0)) {
                lc_bracket_solution crossing_solution;
                status = lc_solve_bracket(
                    &context, hint->g_root, hint->g_prime_root,
                    left, right, &crossing_solution
                );
                result->iterations += crossing_solution.iterations;
                result->bracket_low = t_last + crossing_solution.low;
                result->bracket_high = t_last + crossing_solution.high;
                result->residual = crossing_solution.residual;
                result->tolerance = crossing_solution.tolerance;
                if (status != LC_OK) {
                    return status;
                }
                result->t_spike = t_last + crossing_solution.root;
                if (!lc_isfinite(result->t_spike) || result->t_spike <= t_last) {
                    return LC_NUMERIC_ERROR;
                }
                return LC_OK;
            }
            if (classified_right == LC_REAL_C(0.0) && index == extrema_count &&
                g_prime_right < LC_REAL_C(0.0) && right > LC_REAL_C(0.0)) {
                result->t_spike = t_last + right;
                result->bracket_low = result->t_spike;
                result->bracket_high = result->t_spike;
                result->residual = g_right;
                result->tolerance = lc_root_tolerance(&context, right);
                return LC_OK;
            }
            left = right;
            g_left = classified_right;
        }
    }
    return LC_NO_CROSSING;
}
