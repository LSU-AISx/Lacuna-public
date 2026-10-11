#include "lacuna.h"
#include "network_internal.h"
#include "step_internal.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Compile mixed graphs and execute their events in deterministic time order. */

static int lc_allocation_size_overflows(uint64_t count, size_t element_size) {
    return element_size != 0U && count > (uint64_t)(SIZE_MAX / element_size);
}

/* Host profiling remains separate from model and simulation-clock precision. */
static lc_profile_t lc_monotonic_seconds(void) {
#if LACUNA_ENABLE_PROFILING
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return LC_REAL_C(0.0);
    }
    return (lc_profile_t)value.tv_sec +
        (lc_profile_t)value.tv_nsec * LC_TIME_C(1.0e-9);
#else
    return LC_REAL_C(0.0);
#endif
}

typedef struct lc_event {
    lc_time_t t;
    uint64_t seq;
    uint64_t generation;
    uint32_t index;
    uint32_t subject;
    uint32_t target;
    uint32_t auxiliary;
    lc_real_t value;
    uint8_t phase;
    uint8_t kind;
} lc_event;

/* The heap stores pending work and assigns a stable sequence to tied events. */
typedef struct lc_heap {
    lc_event *items;
    uint64_t size;
    uint64_t logical_size;
    uint64_t capacity;
    uint64_t next_seq;
    uint64_t peak;
} lc_heap;

typedef struct lc_delivery_sort_item {
    lc_time_t delay;
    uint32_t edge;
} lc_delivery_sort_item;

typedef struct lc_learning_sort_item {
    uint32_t program;
    uint32_t slot;
} lc_learning_sort_item;

/* Learning batches group slots that execute the same equation program. */

static int lc_learning_sort_before(const void *left, const void *right) {
    const lc_learning_sort_item *left_item = left;
    const lc_learning_sort_item *right_item = right;
    if (left_item->program < right_item->program) return -1;
    if (left_item->program > right_item->program) return 1;
    if (left_item->slot < right_item->slot) return -1;
    if (left_item->slot > right_item->slot) return 1;
    return 0;
}

typedef struct lc_expression_program_ref {
    const lc_expr_node *source;
    lc_expr_node *owned;
    uint32_t count;
    uint32_t main_representative;
    uint32_t deposit_representative;
    lc_expr_node *normal_specialized;
    lc_expr_node *clamped_specialized;
    lc_expr_node *reset_specialized;
    lc_expr_node *crossing_specialized;
    lc_expr_node *deposit_specialized;
} lc_expression_program_ref;

static int lc_expression_program_ref_before(const void *left, const void *right) {
    const lc_expression_program_ref *left_ref = left;
    const lc_expression_program_ref *right_ref = right;
    uintptr_t left_pointer = (uintptr_t)left_ref->source;
    uintptr_t right_pointer = (uintptr_t)right_ref->source;
    if (left_pointer < right_pointer) {
        return -1;
    }
    if (left_pointer > right_pointer) {
        return 1;
    }
    if (left_ref->count < right_ref->count) {
        return -1;
    }
    if (left_ref->count > right_ref->count) {
        return 1;
    }
    return 0;
}

static lc_expression_program_ref *lc_expression_program_find(
    lc_expression_program_ref *programs,
    uint32_t program_count,
    const lc_expr_node *source,
    uint32_t count
) {
    lc_expression_program_ref key;
    key.source = source;
    key.owned = NULL;
    key.count = count;
    return bsearch(
        &key, programs, program_count, sizeof(lc_expression_program_ref),
        lc_expression_program_ref_before
    );
}

#define LC_EVAL_PLAN_MAX_ROOTS (1U + 3U * LC_ANALYTICAL_MAX_STATES)

/* Replace nodes outside selected root closures with inert constants. */
static lc_status lc_expr_specialize_roots(
    const lc_expr_node *source,
    uint32_t node_count,
    const uint32_t *roots,
    uint32_t root_count,
    lc_expr_node *destination
) {
    uint8_t *active;
    uint32_t index;
    uint32_t cursor;
    if (source == NULL || node_count == 0U || destination == NULL ||
        (root_count > 0U && roots == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    active = calloc(node_count, sizeof(uint8_t));
    if (active == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    for (index = 0U; index < root_count; ++index) {
        if (roots[index] >= node_count) {
            free(active);
            return LC_INVALID_ARGUMENT;
        }
        active[roots[index]] = 1U;
    }
    for (cursor = node_count; cursor > 0U; --cursor) {
        const lc_expr_node *node;
        index = cursor - 1U;
        if (active[index] == 0U) {
            continue;
        }
        node = &source[index];
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
                if (node->lhs >= index) {
                    free(active);
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
                if (node->lhs >= index || node->rhs >= index) {
                    free(active);
                    return LC_INVALID_ARGUMENT;
                }
                active[node->lhs] = 1U;
                active[node->rhs] = 1U;
                break;
            default:
                free(active);
                return LC_INVALID_ARGUMENT;
        }
    }
    for (index = 0U; index < node_count; ++index) {
        if (active[index] != 0U) {
            destination[index] = source[index];
        } else {
            memset(&destination[index], 0, sizeof(lc_expr_node));
            destination[index].op = LC_EXPR_CONST;
        }
    }
    free(active);
    return LC_OK;
}

/* Collect every expression root required by one crossing capability. */
static int lc_mixed_crossing_roots(
    const lc_mixed_node *descriptor,
    uint32_t roots[LC_EVAL_PLAN_MAX_ROOTS],
    uint32_t *root_count
) {
    uint32_t count = 0U;
    uint32_t index;
    if (descriptor == NULL || roots == NULL || root_count == NULL) {
        return 0;
    }
#define LC_APPEND_CROSSING_ROOT(root_value)                                      \
    do {                                                                         \
        if (count >= LC_EVAL_PLAN_MAX_ROOTS) return 0;                           \
        roots[count++] = (root_value);                                            \
    } while (0)
    switch ((lc_crossing_kind)descriptor->crossing_kind) {
        case LC_CROSSING_REACTIVE:
            break;
        case LC_CROSSING_INTEGRATED_HAZARD:
            for (index = 0U; index < descriptor->state_count; ++index) {
                LC_APPEND_CROSSING_ROOT(descriptor->normal_roots[index]);
            }
            if (descriptor->hazard.trajectory_mode_count > 0U) {
                LC_APPEND_CROSSING_ROOT(
                    descriptor->hazard.trajectory_limit_root
                );
                for (index = 0U;
                     index < descriptor->hazard.trajectory_mode_count;
                     ++index) {
                    LC_APPEND_CROSSING_ROOT(
                        descriptor->hazard.trajectory_coefficient_roots[index]
                    );
                    LC_APPEND_CROSSING_ROOT(
                        descriptor->hazard.trajectory_rate_roots[index]
                    );
                }
            }
            break;
        case LC_CROSSING_SCALAR_LOG:
            LC_APPEND_CROSSING_ROOT(descriptor->scalar_log_hint.decay_root);
            LC_APPEND_CROSSING_ROOT(descriptor->scalar_log_hint.affine_root);
            LC_APPEND_CROSSING_ROOT(descriptor->scalar_log_hint.threshold_root);
            break;
        case LC_CROSSING_ALPHA_REAL:
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.g_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.g_prime_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.extremum_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.extremum_prime_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.asymptote_root);
            LC_APPEND_CROSSING_ROOT(
                descriptor->root_hint.membrane_coefficient_root
            );
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.synapse_constant_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.synapse_linear_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.membrane_rate_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.synapse_rate_root);
            LC_APPEND_CROSSING_ROOT(descriptor->root_hint.threshold_root);
            break;
        case LC_CROSSING_TWO_REAL_EXP:
            LC_APPEND_CROSSING_ROOT(descriptor->two_exp_hint.g_root);
            LC_APPEND_CROSSING_ROOT(descriptor->two_exp_hint.g_prime_root);
            LC_APPEND_CROSSING_ROOT(descriptor->two_exp_hint.limit_root);
            LC_APPEND_CROSSING_ROOT(
                descriptor->two_exp_hint.coefficient_one_root
            );
            LC_APPEND_CROSSING_ROOT(
                descriptor->two_exp_hint.coefficient_two_root
            );
            LC_APPEND_CROSSING_ROOT(descriptor->two_exp_hint.rate_one_root);
            LC_APPEND_CROSSING_ROOT(descriptor->two_exp_hint.rate_two_root);
            break;
        case LC_CROSSING_MULTI_REAL_EXP:
            if (descriptor->multi_exp_hint.mode_count >
                LC_ANALYTICAL_MAX_STATES) {
                return 0;
            }
            LC_APPEND_CROSSING_ROOT(descriptor->multi_exp_hint.limit_root);
            for (index = 0U; index < descriptor->multi_exp_hint.mode_count;
                 ++index) {
                LC_APPEND_CROSSING_ROOT(
                    descriptor->multi_exp_hint.coefficient_roots[index]
                );
                LC_APPEND_CROSSING_ROOT(
                    descriptor->multi_exp_hint.rate_roots[index]
                );
            }
            break;
        case LC_CROSSING_REPEATED_REAL_MODE:
        case LC_CROSSING_MULTI_EXP_POLY:
            if (descriptor->exp_poly_hint.block_count >
                    LC_ANALYTICAL_MAX_STATES ||
                descriptor->exp_poly_hint.coefficient_count >
                    LC_ANALYTICAL_MAX_STATES) {
                return 0;
            }
            LC_APPEND_CROSSING_ROOT(descriptor->exp_poly_hint.limit_root);
            for (index = 0U; index < descriptor->exp_poly_hint.block_count;
                 ++index) {
                LC_APPEND_CROSSING_ROOT(
                    descriptor->exp_poly_hint.rate_roots[index]
                );
            }
            for (index = 0U;
                 index < descriptor->exp_poly_hint.coefficient_count;
                 ++index) {
                LC_APPEND_CROSSING_ROOT(
                    descriptor->exp_poly_hint.coefficient_roots[index]
                );
            }
            break;
        case LC_CROSSING_NUMERICAL:
        default:
            return 0;
    }
#undef LC_APPEND_CROSSING_ROOT
    *root_count = count;
    return 1;
}

/* Compare root lists used to share specialized expression programs. */
static int lc_root_lists_equal(
    const uint32_t *left,
    uint32_t left_count,
    const uint32_t *right,
    uint32_t right_count
) {
    return left_count == right_count &&
        (left_count == 0U ||
         memcmp(left, right, left_count * sizeof(uint32_t)) == 0);
}

/* Capture queue or output exhaustion at the active event. */
static void lc_set_resource_error(
    lc_network_error *error,
    uint32_t resource,
    uint64_t capacity,
    uint64_t occupancy,
    uint64_t peak,
    const lc_event *event
) {
    if (error == NULL) {
        return;
    }
    error->resource = resource;
    error->capacity = capacity;
    error->occupancy = occupancy;
    error->peak = peak;
    error->has_event = event != NULL;
    if (event != NULL) {
        error->event_kind = event->kind;
        error->event_phase = event->phase;
        if (event->kind == LC_EVENT_INPUT_SPIKE ||
            event->kind == LC_EVENT_DRIVE_UPDATE ||
            event->kind == LC_EVENT_OUTPUT_SPIKE) {
            error->event_index = event->subject;
            error->node = event->index;
        } else if (event->kind == LC_EVENT_DELIVERY) {
            error->event_index = event->subject > 0U
                                     ? event->auxiliary
                                     : event->index;
            error->node = event->target;
        } else {
            error->event_index = event->index;
            error->node = event->index;
        }
        error->t = event->t;
    } else {
        error->event_kind = LC_EVENT_NONE;
        error->event_phase = LC_PHASE_NONE;
    }
}

/* Translate decoder retention exhaustion into a network diagnostic. */
static void lc_set_decoder_resource_error(
    lc_network_error *error,
    const lc_decoder_run *decoders,
    const lc_event *event
) {
    uint64_t count = 0U;
    uint64_t capacity = 0U;
    if (lc_decoder_run_event_usage(decoders, &count, &capacity) != LC_OK) {
        count = 0U;
        capacity = 0U;
    }
    lc_set_resource_error(
        error, LC_RESOURCE_DECODER_OUTPUT, capacity, count, count, event
    );
}

typedef struct lc_node_runtime {
    uint64_t generation;
    uint64_t refractory_generation;
    int clamped;
    int affine_disabled;
    lc_real_t hazard_remaining;
    uint64_t hazard_draw_index;
    int hazard_initialized;
    lc_step_cache *step_cache;
    uint64_t cache_generation;
} lc_node_runtime;

/* Only numerical nodes allocate histories. Release when destroying runtime. */
static void lc_runtime_clear_caches(lc_node_runtime *runtime, uint32_t count) {
    uint32_t node;
    if (runtime == NULL) return;
    for (node = 0U; node < count; ++node) {
        lc_step_cache_release(runtime[node].step_cache);
        free(runtime[node].step_cache);
        runtime[node].step_cache = NULL;
    }
}

/* Run initialization/reset discards all logical state, not reusable buffers.
 * Plans own their constants and recheck parameter bindings before use. */
static void lc_runtime_reset_caches(lc_node_runtime *runtime, uint32_t count) {
    uint32_t node;
    for (node = 0U; node < count; ++node) {
        lc_step_cache *cache = runtime[node].step_cache;
        lc_step_cache_reset(cache);
        memset(&runtime[node], 0, sizeof(runtime[node]));
        runtime[node].step_cache = cache;
    }
}

static int lc_runtime_sample_cache(const lc_node_runtime *runtime,
                                  lc_time_t t, lc_real_t *state,
                                  uint32_t state_count) {
    return !runtime->clamped && runtime->cache_generation == runtime->generation &&
        lc_step_cache_sample(runtime->step_cache, t, state, state_count);
}

/*
 * Operation-specific expression views for the equation-derived executor.
 * Each view has the same length and node indexes as the source program.
 * Required nodes are copied without modification. Irrelevant nodes are
 * replaced with LC_EXPR_CONST. This preserves the exact operation sequence
 * of every value that can reach an operation root and avoids remapping any
 * state or crossing roots.
 */
/* Confirm that a node may use direct scalar affine arithmetic. */
static int lc_mixed_scalar_affine_enabled(
    const lc_mixed_node *descriptor,
    const lc_node_runtime *runtime
) {
    return descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE &&
        runtime->affine_disabled == 0;
}

static uint64_t lc_hazard_splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

/* Draw the next unit exponential threshold from a node-local stream. */
static lc_status lc_hazard_draw(
    lc_node_runtime *runtime,
    uint64_t seed,
    uint32_t node
) {
    uint64_t bits;
    lc_real_t uniform;
    if (runtime == NULL || runtime->hazard_draw_index == UINT64_MAX) {
        return LC_NUMERIC_ERROR;
    }
    bits = lc_hazard_splitmix64(
        seed ^ ((uint64_t)node * UINT64_C(0xd1b54a32d192ed03)) ^
        (runtime->hazard_draw_index * UINT64_C(0x94d049bb133111eb))
    );
#if LACUNA_REAL_BITS == 16
    /* Binary16 half-bin points stay strictly inside (0, 1). */
    uniform = ((lc_real_t)(bits >> 54U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1.0p-10);
#elif LACUNA_REAL_BITS == 32
    /* Half-bin points remain strictly inside (0, 1) in binary32. */
    uniform = ((lc_real_t)(bits >> 41U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1.0p-23);
#else
    uniform = ((lc_real_t)(bits >> 11U) + LC_REAL_C(0.5)) * LC_REAL_C(0x1.0p-53);
#endif
    runtime->hazard_remaining = -lc_real_log(uniform);
    runtime->hazard_draw_index++;
    runtime->hazard_initialized = 1;
    return lc_isfinite(runtime->hazard_remaining) &&
        runtime->hazard_remaining > LC_REAL_C(0.0) ? LC_OK : LC_NUMERIC_ERROR;
}

typedef struct lc_hazard_integral_context {
    const lc_mixed_node *descriptor;
    const lc_expr_node *program_nodes;
    const lc_real_t *parameters;
    const lc_real_t *state;
    lc_real_t *variables;
    lc_real_t *workspace;
    uint32_t workspace_count;
    uint32_t trajectory_mode_count;
    lc_real_t trajectory_limit;
    lc_real_t trajectory_coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t trajectory_rates[LC_ANALYTICAL_MAX_STATES];
} lc_hazard_integral_context;

/* Evaluate modal trajectory coefficients used by hazard quadrature. */
static lc_status lc_hazard_context_prepare(
    lc_hazard_integral_context *context,
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_real_t *parameters,
    const lc_real_t *state,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    const lc_hazard_config *hazard;
#if LACUNA_REAL_BITS != 16
    lc_wide_t magnitude;
    lc_wide_t physical_scale;
#endif
    uint32_t index;
    lc_status status;
    if (context == NULL || descriptor == NULL || program_nodes == NULL ||
        state == NULL || variables == NULL || workspace == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    memset(context, 0, sizeof(*context));
    context->descriptor = descriptor;
    context->program_nodes = program_nodes;
    context->parameters = parameters;
    context->state = state;
    context->variables = variables;
    context->workspace = workspace;
    context->workspace_count = workspace_count;
    hazard = &descriptor->hazard;
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE ||
        hazard->trajectory_mode_count == 0U) {
        return LC_OK;
    }
    variables[0] = LC_REAL_C(0.0);
    for (index = 0U; index < descriptor->state_count; ++index) {
        variables[index + 1U] = state[index];
    }
    status = lc_expr_evaluate(
        program_nodes, descriptor->program_node_count, parameters,
        descriptor->parameter_count, variables, descriptor->state_count + 1U,
        workspace, workspace_count
    );
    if (status != LC_OK) return status;
    context->trajectory_limit = workspace[hazard->trajectory_limit_root];
    if (!lc_isfinite(context->trajectory_limit)) return LC_NUMERIC_ERROR;
#if LACUNA_REAL_BITS != 16
    magnitude = lc_wide_fabs((lc_wide_t)context->trajectory_limit);
#endif
    for (index = 0U; index < hazard->trajectory_mode_count; ++index) {
        lc_real_t coefficient =
            workspace[hazard->trajectory_coefficient_roots[index]];
        lc_real_t rate = workspace[hazard->trajectory_rate_roots[index]];
        if (!lc_isfinite(coefficient) || !lc_isfinite(rate) || rate >= LC_REAL_C(0.0)) {
            return LC_NUMERIC_ERROR;
        }
        context->trajectory_coefficients[index] = coefficient;
        context->trajectory_rates[index] = rate;
#if LACUNA_REAL_BITS != 16
        magnitude += lc_wide_fabs((lc_wide_t)coefficient);
#endif
    }
#if LACUNA_REAL_BITS != 16
    physical_scale = lc_wide_fmax(
        LC_WIDE_C(1.0),
        lc_wide_fmax(
            lc_wide_fabs((lc_wide_t)state[descriptor->readout]),
            lc_wide_fabs((lc_wide_t)context->trajectory_limit)
        )
    );
    /* Large cancelling modal terms lose more voltage accuracy than the
       configured hazard integration permits.  In that uncommon regime the
       stable expression DAG remains the authoritative evaluator. */
    if (magnitude <= LC_WIDE_C(1.0e5) * physical_scale) {
        context->trajectory_mode_count = hazard->trajectory_mode_count;
    }
#endif
    /* Binary16 has no validated modal-cancellation bound here. Retain the
       authoritative expression trajectory instead of widening its guards. */
    return LC_OK;
}

/* A mixed-profile clock interval must remain representable as model input. */
static int lc_hazard_real_delta(lc_time_t delta, lc_real_t *value) {
    if (!lc_isfinite(delta) || delta < LC_TIME_C(0.0)) return 0;
    *value = (lc_real_t)delta;
    return lc_isfinite(*value) &&
        (delta == LC_TIME_C(0.0) || *value > LC_REAL_C(0.0));
}

#if LACUNA_REAL_BITS < 64
/* Only a certified constant trajectory justifies an unbounded-time shortcut. */
static int lc_hazard_constant_trajectory(const lc_hazard_integral_context *context) {
    const lc_mixed_node *descriptor = context->descriptor;
    uint32_t index;
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE) {
        /* Rounded -b/a need not be stationary under exp/expm1 propagation. */
        return descriptor->scalar_affine_drive == LC_REAL_C(0.0) &&
            context->state[0] == LC_REAL_C(0.0);
    }
    if (context->trajectory_mode_count == 0U ||
        context->state[descriptor->readout] != context->trajectory_limit) return 0;
    for (index = 0U; index < context->trajectory_mode_count; ++index)
        if (context->trajectory_coefficients[index] != LC_REAL_C(0.0)) return 0;
    return 1;
}

/* Hazard voltage must use exactly the scalar runtime's primitive sequence. */
static lc_status lc_hazard_scalar_voltage(
    const lc_mixed_node *descriptor,
    lc_real_t initial,
    lc_time_t duration,
    lc_real_t *voltage
) {
    lc_scalar_lif_model model;
    lc_scalar_state state;
    lc_status status;
    model.a = descriptor->scalar_affine_decay;
    model.b = descriptor->scalar_affine_drive;
    model.threshold = descriptor->threshold;
    model.reset = descriptor->scalar_affine_reset;
    model.refractory = descriptor->refractory;
    model.polarity = descriptor->polarity;
    state.value = initial;
    state.t_last = LC_TIME_C(0.0);
    status = lc_scalar_advance(&model, &state, duration);
    if (status == LC_OK) *voltage = state.value;
    return status;
}
#endif

/* Integrated hazard advances along the selected deterministic trajectory. */

static lc_status lc_hazard_rate_at(
    const lc_hazard_integral_context *context,
    lc_time_t delta,
    lc_real_t *rate
) {
    const lc_mixed_node *descriptor;
    lc_real_t voltage;
    lc_real_t log_rate;
    lc_real_t real_delta;
    uint32_t index;
    lc_status status;
    if (context == NULL || rate == NULL || !lc_isfinite(delta) || delta < LC_TIME_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    descriptor = context->descriptor;
#if LACUNA_REAL_BITS < 64
    if (lc_hazard_constant_trajectory(context)) {
        voltage = context->state[descriptor->readout];
    } else
#endif
    {
    if (!lc_hazard_real_delta(delta, &real_delta)) return LC_NUMERIC_ERROR;
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE) {
#if LACUNA_REAL_BITS < 64
        status = lc_hazard_scalar_voltage(descriptor, context->state[0], delta, &voltage);
        if (status != LC_OK) return status;
#else
        lc_real_t decay = descriptor->scalar_affine_decay;
        lc_real_t asymptote = -descriptor->scalar_affine_drive / decay;
        voltage = asymptote + (context->state[0] - asymptote) *
            lc_real_exp(decay * real_delta);
#endif
    } else if (context->trajectory_mode_count > 0U) {
#if LACUNA_REAL_BITS < 64
        lc_real_t modal_voltage = context->trajectory_limit;
        for (index = 0U; index < context->trajectory_mode_count; ++index) {
            modal_voltage += context->trajectory_coefficients[index] *
                lc_real_exp(context->trajectory_rates[index] * real_delta);
        }
        voltage = modal_voltage;
#else
        lc_wide_t modal_voltage = (lc_wide_t)context->trajectory_limit;
        for (index = 0U; index < context->trajectory_mode_count; ++index) {
            modal_voltage +=
                (lc_wide_t)context->trajectory_coefficients[index] *
                (lc_wide_t)lc_real_exp(context->trajectory_rates[index] * real_delta);
        }
        voltage = (lc_real_t)modal_voltage;
#endif
    } else {
        context->variables[0] = real_delta;
        for (index = 0U; index < descriptor->state_count; ++index) {
            context->variables[index + 1U] = context->state[index];
        }
        status = lc_expr_evaluate(
            context->program_nodes, descriptor->program_node_count,
            context->parameters, descriptor->parameter_count,
            context->variables, descriptor->state_count + 1U,
            context->workspace, context->workspace_count
        );
        if (status != LC_OK) {
            return status;
        }
        voltage = context->workspace[
            descriptor->normal_roots[descriptor->readout]
        ];
    }
    }
    log_rate = descriptor->hazard.log_scale +
        descriptor->hazard.voltage_gain * voltage;
    if (!lc_isfinite(log_rate)) {
        return LC_NUMERIC_ERROR;
    }
#if LACUNA_REAL_BITS < 64
    /* Keep representable subnormal rates; overflow is an explicit failure. */
    *rate = lc_real_exp(log_rate);
#else
    if (log_rate >= lc_real_log(LC_REAL_MAX)) {
        *rate = LC_REAL_MAX;
    } else if (log_rate <= lc_real_log(LC_REAL_MIN)) {
        *rate = LC_REAL_C(0.0);
    } else {
        *rate = lc_real_exp(log_rate);
    }
#endif
    return lc_isfinite(*rate) && *rate >= LC_REAL_C(0.0) ? LC_OK : LC_NUMERIC_ERROR;
}

static lc_real_t lc_hazard_simpson(lc_time_t left, lc_time_t right, lc_real_t f_left,
                                lc_real_t f_mid, lc_real_t f_right) {
    lc_wide_t value = ((lc_wide_t)right - (lc_wide_t)left) *
        ((lc_wide_t)f_left + LC_WIDE_C(4.0) * (lc_wide_t)f_mid +
         (lc_wide_t)f_right) / LC_WIDE_C(6.0);
#if LACUNA_REAL_BITS == 16
    /* Do not mistake an overflowing weighted sum for a resolved integral. */
    return (lc_real_t)value;
#else
    return value >= (lc_wide_t)LC_REAL_MAX ? LC_REAL_MAX : (lc_real_t)value;
#endif
}

/* Refine adaptive Simpson quadrature until its local error is bounded. */
static lc_status lc_hazard_integral_recursive(
    const lc_hazard_integral_context *context,
    lc_time_t left,
    lc_time_t right,
    lc_real_t f_left,
    lc_real_t f_mid,
    lc_real_t f_right,
    lc_real_t whole,
    lc_real_t tolerance,
    uint32_t depth,
    lc_real_t *result
) {
    lc_time_t mid = left + LC_TIME_C(0.5) * (right - left);
    lc_time_t left_mid = left + LC_TIME_C(0.5) * (mid - left);
    lc_time_t right_mid = mid + LC_TIME_C(0.5) * (right - mid);
    lc_real_t f_left_mid;
    lc_real_t f_right_mid;
    lc_real_t left_value;
    lc_real_t right_value;
    lc_real_t combined;
    lc_real_t error;
    lc_status status;
#if LACUNA_REAL_BITS < 64
    if (!(left < left_mid && left_mid < mid && mid < right_mid && right_mid < right))
        return LC_ROOT_NONCONVERGENCE;
#endif
    status = lc_hazard_rate_at(context, left_mid, &f_left_mid);
    if (status != LC_OK) return status;
    status = lc_hazard_rate_at(context, right_mid, &f_right_mid);
    if (status != LC_OK) return status;
    left_value = lc_hazard_simpson(
        left, mid, f_left, f_left_mid, f_mid
    );
    right_value = lc_hazard_simpson(
        mid, right, f_mid, f_right_mid, f_right
    );
#if LACUNA_REAL_BITS == 16
    combined = left_value + right_value;
    if (!lc_isfinite(left_value) || !lc_isfinite(right_value) || !lc_isfinite(combined))
        return LC_NUMERIC_ERROR;
#else
    combined = left_value >= LC_REAL_MAX - right_value
        ? LC_REAL_MAX : left_value + right_value;
    if (combined == LC_REAL_MAX || whole == LC_REAL_MAX) {
        *result = LC_REAL_MAX;
        return LC_OK;
    }
#endif
    error = lc_real_fabs(combined - whole);
    if (error <= LC_REAL_C(15.0) * tolerance) {
        *result = combined + (combined - whole) / LC_REAL_C(15.0);
        return lc_isfinite(*result) && *result >= LC_REAL_C(0.0) ? LC_OK : LC_NUMERIC_ERROR;
    }
    if (depth == 0U) {
        return LC_ROOT_NONCONVERGENCE;
    }
    status = lc_hazard_integral_recursive(
        context, left, mid, f_left, f_left_mid, f_mid, left_value,
        LC_REAL_C(0.5) * tolerance, depth - 1U, result
    );
    if (status != LC_OK) return status;
    {
        lc_real_t right_result;
        status = lc_hazard_integral_recursive(
            context, mid, right, f_mid, f_right_mid, f_right, right_value,
            LC_REAL_C(0.5) * tolerance, depth - 1U, &right_result
        );
        if (status != LC_OK) return status;
#if LACUNA_REAL_BITS == 16
        *result += right_result;
        if (!lc_isfinite(*result)) return LC_NUMERIC_ERROR;
#else
        *result = *result >= LC_REAL_MAX - right_result
            ? LC_REAL_MAX : *result + right_result;
#endif
    }
    return LC_OK;
}

/* Integrate a prepared hazard trajectory over one elapsed interval. */
static lc_status lc_hazard_integral_prepared(
    const lc_hazard_integral_context *context,
    lc_time_t duration,
    lc_real_t *result
) {
    const lc_mixed_node *descriptor;
#if LACUNA_REAL_BITS == 64
    const lc_real_t *state;
#endif
    lc_real_t f_left;
    lc_real_t f_mid;
    lc_real_t f_right;
    lc_real_t whole;
    lc_real_t tolerance;
    lc_status status;
    if (context == NULL || duration < LC_TIME_C(0.0) || !lc_isfinite(duration) ||
        result == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    descriptor = context->descriptor;
#if LACUNA_REAL_BITS == 64
    state = context->state;
#endif
    if (duration == LC_TIME_C(0.0)) {
        *result = LC_REAL_C(0.0);
        return LC_OK;
    }
#if LACUNA_REAL_BITS < 64
    if (lc_hazard_constant_trajectory(context)) {
        lc_real_t rate;
        lc_wide_t integral;
        status = lc_hazard_rate_at(context, LC_TIME_C(0.0), &rate);
        if (status != LC_OK) return status;
        integral = (lc_wide_t)duration * (lc_wide_t)rate;
#if LACUNA_REAL_BITS == 16
        if (!lc_isfinite(integral)) return LC_NUMERIC_ERROR;
#endif
        *result = integral >= (lc_wide_t)LC_REAL_MAX
            ? LC_REAL_MAX : (lc_real_t)integral;
        return lc_isfinite(*result) ? LC_OK : LC_NUMERIC_ERROR;
    }
#else
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE) {
        lc_real_t asymptote = -descriptor->scalar_affine_drive /
            descriptor->scalar_affine_decay;
        lc_real_t scale = lc_real_fmax(LC_REAL_C(1.0), lc_real_fmax(lc_real_fabs(state[0]), lc_real_fabs(asymptote)));
        if (lc_real_fabs(state[0] - asymptote) <= LC_REAL_C(16.0) * LC_REAL_EPSILON * scale) {
            lc_real_t log_rate = descriptor->hazard.log_scale +
                descriptor->hazard.voltage_gain * state[0];
            lc_wide_t integral;
            if (log_rate <= lc_real_log(LC_REAL_MIN)) {
                *result = LC_REAL_C(0.0);
                return LC_OK;
            }
            integral = (lc_wide_t)duration *
                (lc_wide_t)(log_rate >= lc_real_log(LC_REAL_MAX) ? LC_REAL_MAX : lc_real_exp(log_rate));
            *result = integral >= (lc_wide_t)LC_REAL_MAX
                ? LC_REAL_MAX : (lc_real_t)integral;
            return LC_OK;
        }
    }
#endif
    status = lc_hazard_rate_at(context, LC_TIME_C(0.0), &f_left);
    if (status != LC_OK) return status;
    status = lc_hazard_rate_at(context, LC_TIME_C(0.5) * duration, &f_mid);
    if (status != LC_OK) return status;
    status = lc_hazard_rate_at(context, duration, &f_right);
    if (status != LC_OK) return status;
    whole = lc_hazard_simpson(LC_TIME_C(0.0), duration, f_left, f_mid, f_right);
#if LACUNA_REAL_BITS == 16
    if (!lc_isfinite(whole)) return LC_NUMERIC_ERROR;
#else
    if (whole == LC_REAL_MAX) {
        *result = LC_REAL_MAX;
        return LC_OK;
    }
#endif
    tolerance = descriptor->hazard.absolute_tolerance +
        descriptor->hazard.relative_tolerance * lc_real_fabs(whole);
    return lc_hazard_integral_recursive(
        context, LC_TIME_C(0.0), duration, f_left, f_mid, f_right, whole, tolerance,
        descriptor->hazard.maximum_quadrature_depth, result
    );
}

/* Prepare and integrate intrinsic hazard without mutating node state. */
static lc_status lc_hazard_integral(
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_real_t *parameters,
    const lc_real_t *state,
    lc_time_t duration,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_real_t *result
) {
    lc_hazard_integral_context context;
    lc_status status;
    if (duration < LC_TIME_C(0.0) || !lc_isfinite(duration) || result == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    if (duration == LC_TIME_C(0.0)) {
        *result = LC_REAL_C(0.0);
        return LC_OK;
    }
    status = lc_hazard_context_prepare(
        &context, descriptor, program_nodes, parameters, state, variables,
        workspace, workspace_count
    );
    if (status != LC_OK) return status;
    return lc_hazard_integral_prepared(&context, duration, result);
}

/* Consume integrated hazard as deterministic time advances. */
static lc_status lc_hazard_advance_local(
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_real_t *parameters,
    lc_real_t *state,
    lc_time_t duration,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_time_t local_time = LC_TIME_C(0.0);
    lc_real_t real_duration;
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE) {
#if LACUNA_REAL_BITS < 64
        return lc_hazard_scalar_voltage(descriptor, state[0], duration, &state[0]);
#else
        lc_real_t decay = descriptor->scalar_affine_decay;
        lc_real_t asymptote = -descriptor->scalar_affine_drive / decay;
        if (!lc_hazard_real_delta(duration, &real_duration)) return LC_NUMERIC_ERROR;
        state[0] = asymptote + (state[0] - asymptote) *
            lc_real_exp(decay * real_duration);
        return lc_isfinite(state[0]) ? LC_OK : LC_NUMERIC_ERROR;
#endif
    }
    if (!lc_hazard_real_delta(duration, &real_duration)) return LC_NUMERIC_ERROR;
    return lc_expr_state_advance(
        program_nodes, descriptor->program_node_count, parameters,
        descriptor->parameter_count, descriptor->normal_roots,
        descriptor->state_count, state, &local_time, duration, variables,
        descriptor->state_count + 1U, workspace, workspace_count
    );
}

/* Invert accumulated hazard to find the next stochastic spike time. */
static lc_status lc_hazard_predict_duration(
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_real_t *parameters,
    const lc_real_t *initial_state,
    lc_real_t target,
    lc_time_t horizon,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_time_t *duration
) {
    lc_real_t local_state[LC_ANALYTICAL_MAX_STATES];
    lc_time_t elapsed = LC_TIME_C(0.0);
    lc_real_t remaining = target;
    const lc_time_t maximum_segment = LC_TIME_C(512.0);
#if LACUNA_REAL_BITS < 64
    uint32_t segment_count = 0U;
#endif
    uint32_t local;
    if (descriptor == NULL || initial_state == NULL || duration == NULL ||
        !lc_isfinite(target) || target <= LC_REAL_C(0.0) || !lc_isfinite(horizon) ||
        horizon < LC_TIME_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    for (local = 0U; local < descriptor->state_count; ++local) {
        local_state[local] = initial_state[local];
    }
#if LACUNA_REAL_BITS < 64
    if (horizon > LC_TIME_C(0.0)) {
        lc_hazard_integral_context context;
        lc_status status = lc_hazard_context_prepare(
            &context, descriptor, program_nodes, parameters, local_state,
            variables, workspace, workspace_count
        );
        if (status != LC_OK) return status;
        if (lc_hazard_constant_trajectory(&context)) {
            lc_real_t rate;
            lc_time_t waiting;
            status = lc_hazard_rate_at(&context, LC_TIME_C(0.0), &rate);
            if (status != LC_OK) return status;
            if (rate == LC_REAL_C(0.0)) return LC_NO_CROSSING;
            waiting = (lc_time_t)target / (lc_time_t)rate;
            if (waiting > horizon) return LC_NO_CROSSING;
            if (!lc_isfinite(waiting) || waiting <= LC_TIME_C(0.0)) return LC_NUMERIC_ERROR;
            *duration = waiting;
            return LC_OK;
        }
    }
#else
    if (horizon > LC_REAL_C(0.0)) {
        lc_real_t probe_state[LC_ANALYTICAL_MAX_STATES];
        lc_time_t probe = lc_time_fmin(LC_REAL_C(1.0), horizon);
        lc_real_t rate;
        int unchanged = 1;
        lc_status status;
        lc_hazard_integral_context context;
        for (local = 0U; local < descriptor->state_count; ++local) {
            probe_state[local] = local_state[local];
        }
        status = lc_hazard_advance_local(
            descriptor, program_nodes, parameters, probe_state, probe,
            variables, workspace, workspace_count
        );
        if (status != LC_OK) return status;
        for (local = 0U; local < descriptor->state_count; ++local) {
            if (probe_state[local] != local_state[local]) {
                unchanged = 0;
                break;
            }
        }
        if (unchanged) {
            status = lc_hazard_context_prepare(
                &context, descriptor, program_nodes, parameters, local_state,
                variables, workspace, workspace_count
            );
            if (status != LC_OK) return status;
            status = lc_hazard_rate_at(&context, LC_REAL_C(0.0), &rate);
            if (status != LC_OK) return status;
            if (rate == LC_REAL_C(0.0) || target / rate > horizon) {
                return LC_NO_CROSSING;
            }
            *duration = target / rate;
            return lc_isfinite(*duration) ? LC_OK : LC_NUMERIC_ERROR;
        }
    }
#endif
    while (elapsed < horizon) {
        lc_time_t segment = lc_time_fmin(maximum_segment, horizon - elapsed);
        lc_real_t consumed;
        lc_hazard_integral_context context;
        lc_status status = lc_hazard_context_prepare(
            &context, descriptor, program_nodes, parameters, local_state,
            variables, workspace, workspace_count
        );
#if LACUNA_REAL_BITS < 64
        if (segment_count++ >= descriptor->hazard.maximum_root_iterations)
            return LC_ROOT_NONCONVERGENCE;
        if (!(segment > LC_TIME_C(0.0)) || !(elapsed + segment > elapsed))
            return LC_NUMERIC_ERROR;
#endif
        if (status != LC_OK) return status;
        status = lc_hazard_integral_prepared(&context, segment, &consumed);
        if (status != LC_OK) return status;
        if (consumed >= remaining) {
            lc_time_t lower = LC_TIME_C(0.0);
            lc_time_t upper = segment;
            lc_time_t candidate = segment * (
#if LACUNA_REAL_BITS < 64
                (lc_time_t)remaining / (lc_time_t)consumed
#else
                remaining / consumed
#endif
            );
            uint32_t iteration;
            candidate = lc_time_fmax(lower, lc_time_fmin(upper, candidate));
            for (iteration = 0U;
                 iteration < descriptor->hazard.maximum_root_iterations;
                 ++iteration) {
                lc_real_t value;
                lc_real_t residual;
                lc_real_t rate;
                lc_time_t next;
                if (upper - lower <= descriptor->hazard.time_tolerance *
                        (LC_TIME_C(1.0) + lc_time_fabs(elapsed + candidate))) {
                    *duration = elapsed + LC_TIME_C(0.5) * (lower + upper);
                    return LC_OK;
                }
                status = lc_hazard_integral_prepared(
                    &context, candidate, &value
                );
                if (status != LC_OK) return status;
                residual = value - remaining;
                if (lc_real_fabs(residual) <= descriptor->hazard.absolute_tolerance +
                        descriptor->hazard.relative_tolerance * remaining) {
                    *duration = elapsed + candidate;
                    return LC_OK;
                }
                if (residual >= LC_REAL_C(0.0)) upper = candidate;
                else lower = candidate;
                status = lc_hazard_rate_at(&context, candidate, &rate);
                if (status != LC_OK) return status;
                next = rate > LC_REAL_C(0.0) ? candidate -
#if LACUNA_REAL_BITS < 64
                    (lc_time_t)residual / (lc_time_t)rate
#else
                    residual / rate
#endif
                    : (lc_time_t)NAN;
                candidate = lc_isfinite(next) && next > lower && next < upper
                    ? next : lower + LC_TIME_C(0.5) * (upper - lower);
            }
            return LC_ROOT_NONCONVERGENCE;
        }
        remaining -= consumed;
        {
#if LACUNA_REAL_BITS == 64
            lc_real_t before[LC_ANALYTICAL_MAX_STATES];
            int unchanged = 1;
#endif
            lc_real_t rate;
#if LACUNA_REAL_BITS == 64
            for (local = 0U; local < descriptor->state_count; ++local) {
                before[local] = local_state[local];
            }
#endif
            status = lc_hazard_advance_local(
                descriptor, program_nodes, parameters, local_state, segment,
                variables, workspace, workspace_count
            );
            if (status != LC_OK) return status;
            elapsed += segment;
#if LACUNA_REAL_BITS == 64
            for (local = 0U; local < descriptor->state_count; ++local) {
                if (local_state[local] != before[local]) {
                    unchanged = 0;
                    break;
                }
            }
            if (!unchanged) continue;
#endif
            {
                lc_hazard_integral_context context;
                status = lc_hazard_context_prepare(
                    &context, descriptor, program_nodes, parameters,
                    local_state, variables, workspace, workspace_count
                );
                if (status != LC_OK) return status;
#if LACUNA_REAL_BITS < 64
                if (!lc_hazard_constant_trajectory(&context)) continue;
#endif
                status = lc_hazard_rate_at(&context, LC_TIME_C(0.0), &rate);
            }
            if (status != LC_OK) return status;
#if LACUNA_REAL_BITS < 64
            {
                lc_time_t waiting;
                if (rate == LC_REAL_C(0.0)) return LC_NO_CROSSING;
                waiting = (lc_time_t)remaining / (lc_time_t)rate;
                if (waiting > horizon - elapsed) return LC_NO_CROSSING;
                *duration = elapsed + waiting;
                return lc_isfinite(*duration) && waiting > LC_TIME_C(0.0) &&
                    *duration > elapsed ? LC_OK : LC_NUMERIC_ERROR;
            }
#else
            if (rate == LC_REAL_C(0.0) || remaining / rate > horizon - elapsed) {
                return LC_NO_CROSSING;
            }
            *duration = elapsed + remaining / rate;
            return lc_isfinite(*duration) ? LC_OK : LC_NUMERIC_ERROR;
#endif
        }
    }
    return LC_NO_CROSSING;
}

/* Advance a qualified scalar node with stable closed-form arithmetic. */
static lc_status lc_mixed_scalar_affine_advance(
    const lc_mixed_node *descriptor,
    lc_node_runtime *runtime,
    lc_real_t *state,
    lc_time_t *t_last,
    lc_time_t t
) {
    lc_scalar_lif_model model;
    lc_scalar_state scalar_state;
    lc_status status;
    if (t < *t_last) {
        return LC_TIME_REVERSED;
    }
    if (runtime->clamped) {
        *state = descriptor->scalar_affine_reset;
        *t_last = t;
        return LC_OK;
    }
    model.a = descriptor->scalar_affine_decay;
    model.b = descriptor->scalar_affine_drive;
    model.threshold = descriptor->threshold;
    model.reset = descriptor->scalar_affine_reset;
    model.refractory = descriptor->refractory;
    model.polarity = descriptor->polarity;
    scalar_state.value = *state;
    scalar_state.t_last = *t_last;
    status = lc_scalar_advance(&model, &scalar_state, t);
    if (status == LC_OK) {
        *state = scalar_state.value;
        *t_last = scalar_state.t_last;
    }
    return status;
}

typedef struct lc_output_sink {
    lc_output_spike *buffer;
    uint64_t capacity;
    uint64_t *count;
    lc_decoder_run *decoders;
} lc_output_sink;

/* Append a spike and stream it to decoders before further fan-out. */
static lc_status lc_emit_output_spike(
    lc_time_t t,
    uint32_t node,
    lc_output_sink *sink,
    lc_run_stats *stats,
    lc_network_error *error
) {
    lc_output_spike spike;
    lc_event event;
    lc_status status;
    if (sink == NULL || sink->count == NULL || stats == NULL) {
        return LC_OUTPUT_OVERFLOW;
    }
    memset(&event, 0, sizeof(event));
    event.t = t;
    event.phase = LC_PHASE_PREDICTION;
    event.kind = LC_EVENT_OUTPUT_SPIKE;
    event.index = node;
    event.subject = (uint32_t)(*sink->count > UINT32_MAX
                                   ? UINT32_MAX
                                   : *sink->count);
    /* A zero-capacity sink is the count-only recording mode.  The event is
       still delivered to any decoder and included in run statistics, but no
       raw spike payload is retained. */
    if (sink->capacity > 0U && *sink->count >= sink->capacity) {
        lc_set_resource_error(
            error, LC_RESOURCE_OUTPUT, sink->capacity, *sink->count,
            *sink->count, &event
        );
        return LC_OUTPUT_OVERFLOW;
    }
    spike.t = t;
    spike.node = node;
    if (sink->decoders != NULL) {
        status = lc_decoder_run_consume(sink->decoders, &spike);
        if (status != LC_OK) {
            if (status == LC_DECODER_OUTPUT_OVERFLOW) {
                lc_set_decoder_resource_error(error, sink->decoders, &event);
            }
            return status;
        }
    }
    if (sink->capacity > 0U) {
        sink->buffer[*sink->count] = spike;
        (*sink->count)++;
    }
    stats->output_spikes++;
    return LC_OK;
}

typedef struct lc_learning_observer_state {
    lc_real_t slow_voltage;
    lc_real_t fast_activity;
    lc_real_t slow_activity;
    lc_real_t sensitivity;
    lc_time_t t_activity;
} lc_learning_observer_state;

typedef struct lc_learning_readout_trajectory {
    uint32_t mode_count;
    lc_real_t limit;
    lc_real_t coefficients[LC_ANALYTICAL_MAX_STATES];
    lc_real_t rates[LC_ANALYTICAL_MAX_STATES];
} lc_learning_readout_trajectory;

struct lc_mixed_run {
    lc_compiled_graph *graph;
    lc_real_t *state;
    lc_time_t *t_last;
    lc_mixed_node *active_nodes;
    lc_real_t *active_parameters;
    lc_node_runtime *runtime;
    lc_real_t *state_deposits;
    lc_real_t *program_deposits;
    uint8_t *affected;
    uint8_t *deposit_affected;
    uint8_t *timestamp_reset;
    uint8_t *fired;
    uint32_t *affected_nodes;
    uint32_t *fired_nodes;
    uint32_t *timestamp_reset_nodes;
    lc_real_t *workspace;
    lc_plasticity_state *plasticity;
    lc_learning_observer_state *learning_observers;
    lc_real_t *learning_weight_deltas;
    uint8_t *learning_weight_touched;
    uint32_t *learning_weight_masters;
    uint32_t learning_weight_master_count;
    lc_modulation_event *pending_modulations;
    uint32_t pending_modulation_count;
    lc_real_t variables[LC_LEARNING_MAX_TRACES + LC_LEARNING_BASE_VARIABLE_COUNT];
    lc_heap heap;
    uint64_t heap_storage_capacity;
    lc_run_config incremental_config;
    lc_time_t frontier;
    uint64_t trace_sequence;
    uint64_t next_input_subject;
    int incremental_active;
    int incremental_failed;
    int ready;
};

/* Add a node once to a sparse same-time work set. */
static void lc_sparse_node_add(
    uint8_t *selected,
    uint32_t *nodes,
    uint32_t *count,
    uint32_t node
) {
    if (selected[node] == 0U) {
        selected[node] = 1U;
        nodes[*count] = node;
        (*count)++;
    }
}

static int lc_uint32_before(const void *left, const void *right) {
    uint32_t left_value = *(const uint32_t *)left;
    uint32_t right_value = *(const uint32_t *)right;
    if (left_value < right_value) return -1;
    if (left_value > right_value) return 1;
    return 0;
}

/* Apply trace kind and node filters before constructing a record. */
static int lc_trace_selected(
    const lc_trace_config *trace,
    uint32_t kind,
    uint32_t node
) {
    if (trace == NULL || kind >= LC_TRACE_KIND_COUNT ||
        (trace->kind_mask & (UINT64_C(1) << kind)) == 0U) {
        return 0;
    }
    return trace->node_mask == NULL || trace->node_mask[node] != 0U;
}

/* Validate caller-owned trace storage and selection masks. */
static int lc_trace_config_valid(
    const lc_trace_config *trace,
    uint32_t node_count
) {
    uint64_t valid_mask = (UINT64_C(1) << LC_TRACE_KIND_COUNT) - UINT64_C(1);
    int has_buffer;
    int has_consumer;
    if (trace == NULL) {
        return 1;
    }
    has_buffer = trace->records != NULL && trace->capacity > 0U;
    has_consumer = trace->consumer != NULL;
    if (trace->count == NULL || trace->capture_state > 1U ||
        (trace->kind_mask & ~valid_mask) != 0U ||
        ((trace->node_mask == NULL) != (trace->node_mask_count == 0U)) ||
        (trace->node_mask != NULL && trace->node_mask_count != node_count) ||
        (trace->kind_mask != 0U && has_buffer == has_consumer) ||
        (trace->records == NULL && trace->capacity != 0U) ||
        (trace->records != NULL && trace->capacity == 0U)) {
        return 0;
    }
    return 1;
}

/* Validate ordered inspection requests and output capacity. */
static lc_status lc_inspection_config_status(
    const lc_state_inspection_config *inspections,
    uint32_t node_count,
    const lc_time_t *t_last,
    lc_time_t t_end,
    int include_end
) {
    uint32_t index;
    lc_time_t previous = LC_REAL_C(0.0);
    if (inspections == NULL) {
        return LC_OK;
    }
    if (inspections->count == NULL ||
        (inspections->request_count == 0U &&
         (inspections->requests != NULL || inspections->results != NULL ||
          inspections->capacity != 0U)) ||
        (inspections->request_count > 0U &&
         (inspections->requests == NULL || inspections->results == NULL))) {
        return LC_INVALID_ARGUMENT;
    }
    if ((uint64_t)inspections->request_count > inspections->capacity) {
        return LC_INSPECTION_OVERFLOW;
    }
    for (index = 0; index < inspections->request_count; ++index) {
        const lc_state_inspection_request *request = &inspections->requests[index];
        if (request->node >= node_count || !lc_isfinite(request->t) ||
            request->t < LC_REAL_C(0.0) || request->t > t_end ||
            (!include_end && request->t == t_end) ||
            request->t < t_last[request->node] ||
            (index > 0U && request->t < previous)) {
            return LC_INVALID_ARGUMENT;
        }
        previous = request->t;
    }
    return LC_OK;
}

/* Report whether this trace record needs a state snapshot. */
static int lc_trace_wants_state(
    const lc_trace_config *trace,
    uint32_t kind,
    uint32_t node
) {
    return lc_trace_selected(trace, kind, node) && trace->capture_state != 0U;
}

/* Copy one node's bounded state vector into a trace field. */
static void lc_trace_copy_state(
    lc_real_t values[LC_ANALYTICAL_MAX_STATES],
    const lc_real_t *state,
    uint32_t state_count
) {
    uint32_t index;
    for (index = 0; index < state_count; ++index) {
        values[index] = state[index];
    }
}

/* Emit one causal record to a buffer, consumer, or both. */
static lc_status lc_trace_emit(
    lc_trace_config *trace,
    uint32_t kind,
    uint32_t phase,
    lc_time_t t,
    uint32_t node,
    uint32_t subject,
    uint64_t generation,
    lc_real_t value,
    const lc_real_t *before,
    const lc_real_t *after,
    uint32_t state_count,
    lc_network_error *error
) {
    lc_trace_record record;
    lc_status status;
    if (!lc_trace_selected(trace, kind, node)) {
        return LC_OK;
    }
    if (trace->count == NULL || *trace->count == UINT64_MAX ||
        state_count > LC_ANALYTICAL_MAX_STATES ||
        ((before == NULL) != (after == NULL))) {
        return LC_INVALID_ARGUMENT;
    }
    memset(&record, 0, sizeof(record));
    record.t = t;
    if (trace->sequence_base > UINT64_MAX - *trace->count) {
        return LC_NUMERIC_ERROR;
    }
    record.sequence = trace->sequence_base + *trace->count;
    record.generation = generation;
    record.kind = kind;
    record.phase = phase;
    record.node = node;
    record.subject = subject;
    record.value = value;
    if (trace->capture_state != 0U && before != NULL) {
        record.state_count = state_count;
        lc_trace_copy_state(record.before, before, state_count);
        lc_trace_copy_state(record.after, after, state_count);
    }
    if (trace->consumer != NULL) {
        status = trace->consumer(&record, trace->consumer_context);
        if (status != LC_OK) {
            return status;
        }
    } else {
        if (*trace->count >= trace->capacity) {
            lc_set_resource_error(
                error, LC_RESOURCE_TRACE, trace->capacity, *trace->count,
                *trace->count, NULL
            );
            if (error != NULL) {
                error->has_event = 1U;
                error->event_index = kind;
                error->event_phase = phase;
                error->node = node;
                error->t = t;
            }
            return LC_TRACE_OVERFLOW;
        }
        trace->records[*trace->count] = record;
    }
    (*trace->count)++;
    return LC_OK;
}

/* Inspections observe settled state and never alter the event frontier. */

static lc_status lc_mixed_capture_inspection(
    lc_mixed_run *run,
    lc_state_inspection_config *inspections,
    uint32_t request_index,
    lc_network_error *error
) {
    const lc_state_inspection_request *request;
    const lc_mixed_node *descriptor;
    lc_state_inspection_result result;
    lc_real_t sample[LC_ANALYTICAL_MAX_STATES];
    lc_time_t sample_time;
    lc_status status;
    if (run == NULL || inspections == NULL ||
        request_index >= inspections->request_count) {
        return LC_INSPECTION_OVERFLOW;
    }
    if (*inspections->count >= inspections->capacity) {
        const lc_state_inspection_request *overflowed =
            &inspections->requests[request_index];
        lc_set_resource_error(
            error, LC_RESOURCE_INSPECTION, inspections->capacity,
            *inspections->count, *inspections->count, NULL
        );
        if (error != NULL) {
            error->has_event = 1U;
            error->event_index = request_index;
            error->node = overflowed->node;
            error->t = overflowed->t;
        }
        return LC_INSPECTION_OVERFLOW;
    }
    request = &inspections->requests[request_index];
    descriptor = &run->active_nodes[request->node];
    sample_time = run->t_last[request->node];
    lc_trace_copy_state(
        sample, &run->state[descriptor->state_offset], descriptor->state_count
    );
    if (lc_mixed_scalar_affine_enabled(
            descriptor, &run->runtime[request->node])) {
        status = lc_mixed_scalar_affine_advance(
            descriptor, &run->runtime[request->node], sample, &sample_time,
            request->t
        );
    } else if (descriptor->dispatch == LC_STEPPED) {
        lc_step_result step_result;
        if (request->t == sample_time || lc_runtime_sample_cache(
                &run->runtime[request->node], request->t, sample,
                descriptor->state_count)) {
            status = LC_OK;
        } else status = lc_expr_step_advance(
            descriptor->program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &run->active_parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count, descriptor->normal_roots,
            descriptor->state_count, descriptor->readout,
            &descriptor->step_config, sample, &sample_time, request->t,
            run->runtime[request->node].clamped != 0, run->variables,
            descriptor->state_count + 1U, run->workspace,
            run->graph->workspace_count, &step_result
        );
    } else {
        status = lc_expr_state_advance(
            descriptor->program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &run->active_parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count,
            run->runtime[request->node].clamped
                ? descriptor->clamped_roots
                : descriptor->normal_roots,
            descriptor->state_count, sample, &sample_time, request->t,
            run->variables, descriptor->state_count + 1U, run->workspace,
            run->graph->workspace_count
        );
    }
    if (status != LC_OK) {
        return status;
    }
    memset(&result, 0, sizeof(result));
    result.t = request->t;
    result.generation = run->runtime[request->node].generation;
    result.node = request->node;
    result.state_count = descriptor->state_count;
    result.clamped = run->runtime[request->node].clamped != 0;
    lc_trace_copy_state(result.values, sample, descriptor->state_count);
    inspections->results[*inspections->count] = result;
    (*inspections->count)++;
    return LC_OK;
}

/* Capture pending inspections strictly before an event boundary. */
static lc_status lc_mixed_capture_inspections_before(
    lc_mixed_run *run,
    lc_state_inspection_config *inspections,
    uint32_t *cursor,
    lc_time_t t,
    lc_network_error *error
) {
    lc_status status;
    if (inspections == NULL) {
        return LC_OK;
    }
    while (*cursor < inspections->request_count &&
           inspections->requests[*cursor].t < t) {
        status = lc_mixed_capture_inspection(run, inspections, *cursor, error);
        if (status != LC_OK) {
            return status;
        }
        (*cursor)++;
    }
    return LC_OK;
}

/* Capture pending inspections through a settled closed boundary. */
static lc_status lc_mixed_capture_inspections_through(
    lc_mixed_run *run,
    lc_state_inspection_config *inspections,
    uint32_t *cursor,
    lc_time_t t,
    lc_network_error *error
) {
    lc_status status;
    if (inspections == NULL) {
        return LC_OK;
    }
    while (*cursor < inspections->request_count &&
           inspections->requests[*cursor].t <= t) {
        status = lc_mixed_capture_inspection(run, inspections, *cursor, error);
        if (status != LC_OK) {
            return status;
        }
        (*cursor)++;
    }
    return LC_OK;
}

/* Event ordering is part of the execution contract for same-time cascades. */
static int lc_event_before(const lc_event *left, const lc_event *right) {
    uint8_t left_order;
    uint8_t right_order;
    if (left->t < right->t) {
        return 1;
    }
    if (left->t > right->t) {
        return 0;
    }
    if (left->phase < right->phase) {
        return 1;
    }
    if (left->phase > right->phase) {
        return 0;
    }
    left_order = left->kind;
    right_order = right->kind;
    if (left->phase == LC_PHASE_BOUNDARY) {
        left_order = left->kind == LC_EVENT_DRIVE_UPDATE ? 0U : 1U;
        right_order = right->kind == LC_EVENT_DRIVE_UPDATE ? 0U : 1U;
    } else if (left->phase == LC_PHASE_DEPOSIT) {
        left_order = left->kind == LC_EVENT_INPUT_SPIKE ? 0U : 1U;
        right_order = right->kind == LC_EVENT_INPUT_SPIKE ? 0U : 1U;
    }
    if (left_order < right_order) {
        return 1;
    }
    if (left_order > right_order) {
        return 0;
    }
    return left->seq < right->seq;
}

static int lc_delivery_sort_before(const void *left, const void *right) {
    const lc_delivery_sort_item *left_item = left;
    const lc_delivery_sort_item *right_item = right;
    if (left_item->delay < right_item->delay) {
        return -1;
    }
    if (left_item->delay > right_item->delay) {
        return 1;
    }
    if (left_item->edge < right_item->edge) {
        return -1;
    }
    if (left_item->edge > right_item->edge) {
        return 1;
    }
    return 0;
}

static uint64_t lc_event_logical_charge(const lc_event *event) {
    if (event->kind == LC_EVENT_DELIVERY && event->subject > 0U) {
        return event->subject;
    }
    return 1U;
}

static lc_status lc_heap_push(lc_heap *heap, lc_event event) {
    uint64_t charge = lc_event_logical_charge(&event);
    uint64_t index;
    if (charge > heap->capacity - heap->logical_size) {
        return LC_QUEUE_OVERFLOW;
    }
    if (heap->next_seq == UINT64_MAX) {
        return LC_NUMERIC_ERROR;
    }
    event.seq = heap->next_seq++;
    index = heap->size++;
    heap->logical_size += charge;
    while (index > 0U) {
        uint64_t parent = (index - 1U) / 2U;
        if (!lc_event_before(&event, &heap->items[parent])) {
            break;
        }
        heap->items[index] = heap->items[parent];
        index = parent;
    }
    heap->items[index] = event;
    if (heap->logical_size > heap->peak) {
        heap->peak = heap->logical_size;
    }
    return LC_OK;
}

/* Push an event and attach capacity details if the queue is full. */
static lc_status lc_heap_push_report(
    lc_heap *heap,
    lc_event event,
    lc_network_error *error
) {
    lc_status status = lc_heap_push(heap, event);
    if (status == LC_QUEUE_OVERFLOW) {
        lc_set_resource_error(
            error, LC_RESOURCE_QUEUE, heap->capacity, heap->logical_size,
            heap->peak, &event
        );
    }
    return status;
}

static lc_event lc_heap_pop(lc_heap *heap) {
    lc_event result = heap->items[0];
    uint64_t index = 0;
    heap->logical_size -= lc_event_logical_charge(&result);
    heap->size--;
    if (heap->size == 0) {
        return result;
    }
    lc_event last = heap->items[heap->size];
    for (;;) {
        uint64_t left = index * 2U + 1U;
        uint64_t right = left + 1U;
        uint64_t smallest = left;
        if (left >= heap->size) {
            break;
        }
        if (right < heap->size &&
            lc_event_before(&heap->items[right], &heap->items[left])) {
            smallest = right;
        }
        if (!lc_event_before(&heap->items[smallest], &last)) {
            break;
        }
        heap->items[index] = heap->items[smallest];
        index = smallest;
    }
    heap->items[index] = last;
    return result;
}

/* Compare event times with a scale-aware floating-point tolerance. */
static int lc_at(
    const lc_heap *heap,
    lc_time_t t,
    lc_network_event_phase phase
) {
    return heap->size > 0 && heap->items[0].t == t && heap->items[0].phase == (uint8_t)phase;
}

static int lc_valid_model(const lc_scalar_lif_model *model) {
    return lc_isfinite(model->a) && lc_isfinite(model->b) && lc_isfinite(model->threshold) &&
           lc_isfinite(model->reset) && lc_isfinite(model->refractory) && model->a < LC_REAL_C(0.0) &&
           model->refractory >= LC_REAL_C(0.0) && model->reset < model->threshold &&
           model->polarity <= LC_MIXED;
}

static int lc_edge_weight_valid(uint32_t polarity, lc_real_t weight) {
    return lc_isfinite(weight) && (polarity == LC_MIXED || weight >= LC_REAL_C(0.0));
}

static lc_real_t lc_signed_edge_weight(uint32_t polarity, lc_real_t weight) {
    return polarity == LC_INHIBITORY ? -weight : weight;
}

/* Advance one scalar LIF node unless it remains refractory. */
static lc_status lc_advance_node(
    const lc_scalar_lif_model *models,
    lc_scalar_state *states,
    lc_node_runtime *runtime,
    uint32_t node,
    lc_time_t t
) {
    if (runtime[node].clamped) {
        if (t < states[node].t_last) {
            return LC_TIME_REVERSED;
        }
        states[node].value = models[node].reset;
        states[node].t_last = t;
        return LC_OK;
    }
    return lc_scalar_advance(&models[node], &states[node], t);
}

/* Replace a scalar node's autonomous prediction by generation number. */
static lc_status lc_schedule_prediction(
    const lc_scalar_lif_model *models,
    const lc_scalar_state *states,
    lc_node_runtime *runtime,
    uint32_t node,
    const lc_run_config *config,
    lc_heap *heap
) {
    lc_time_t t_spike = LC_REAL_C(0.0);
    lc_dispatch_form dispatch = LC_REACTIVE;
    lc_status status;
    lc_event event;

    if (runtime[node].clamped) {
        return LC_OK;
    }
    status = lc_scalar_predict(&models[node], &states[node], &t_spike, &dispatch);
    if (status == LC_NO_CROSSING) {
        return LC_OK;
    }
    if (status != LC_OK) {
        return status;
    }
    if (t_spike > config->t_end) {
        return LC_OK;
    }
    memset(&event, 0, sizeof(event));
    event.t = t_spike;
    event.phase = LC_PHASE_PREDICTION;
    event.kind = LC_EVENT_AUTONOMOUS_SPIKE;
    event.index = node;
    event.generation = runtime[node].generation;
    return lc_heap_push(heap, event);
}

/* Schedule one delayed scalar edge delivery. */
static lc_status lc_push_delivery(
    lc_heap *heap,
    uint32_t edge,
    uint32_t target,
    lc_time_t t,
    const lc_run_config *config,
    lc_network_error *error
) {
    lc_event event;
    if (t > config->t_end) {
        return LC_OK;
    }
    if (!lc_isfinite(t)) {
        return LC_NUMERIC_ERROR;
    }
    memset(&event, 0, sizeof(event));
    event.t = t;
    event.phase = LC_PHASE_DEPOSIT;
    event.kind = LC_EVENT_DELIVERY;
    event.index = edge;
    event.target = target;
    return lc_heap_push_report(heap, event, error);
}

/* Schedule one shared event for edges with equal source and delay. */
static lc_status lc_push_delivery_group(
    lc_heap *heap,
    const lc_compiled_graph *graph,
    uint32_t group_index,
    lc_time_t t,
    const lc_run_config *config,
    lc_network_error *error
) {
    const lc_delivery_group *group;
    lc_event event;
    uint64_t available;
    uint32_t diagnostic_edge;
    if (graph == NULL || group_index >= graph->delivery_group_count) {
        return LC_INVALID_ARGUMENT;
    }
    if (t > config->t_end) {
        return LC_OK;
    }
    if (!lc_isfinite(t)) {
        return LC_NUMERIC_ERROR;
    }
    group = &graph->delivery_groups[group_index];
    if (group->edge_count == 0U) {
        return LC_INVALID_ARGUMENT;
    }
    memset(&event, 0, sizeof(event));
    event.t = t;
    event.phase = LC_PHASE_DEPOSIT;
    event.kind = LC_EVENT_DELIVERY;
    event.index = group_index;
    event.subject = group->edge_count;
    event.auxiliary = group->first_edge;
    event.target = graph->edges[group->first_edge].post;
    available = heap->capacity - heap->logical_size;
    if (group->edge_count > available) {
        diagnostic_edge = graph->delivery_group_edges[
            group->edge_offset + available
        ];
        event.auxiliary = diagnostic_edge;
        event.target = graph->edges[diagnostic_edge].post;
        if (heap->peak < heap->capacity) {
            heap->peak = heap->capacity;
        }
        lc_set_resource_error(
            error, LC_RESOURCE_QUEUE, heap->capacity, heap->capacity,
            heap->peak, &event
        );
        return LC_QUEUE_OVERFLOW;
    }
    return lc_heap_push_report(heap, event, error);
}

/* Commit a scalar same-time firing set and schedule its consequences. */
static lc_status lc_fire_nodes(
    const lc_scalar_lif_model *models,
    lc_scalar_state *states,
    lc_node_runtime *runtime,
    uint32_t node_count,
    const lc_delta_edge *edges,
    const uint64_t *outgoing_offsets,
    const uint32_t *outgoing_edges,
    const lc_run_config *config,
    lc_heap *heap,
    const uint8_t *fired,
    lc_time_t t,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats
) {
    uint32_t node;
    lc_output_sink sink = {
        outputs, config->output_capacity, output_count, NULL
    };
    for (node = 0; node < node_count; ++node) {
        uint64_t position;
        lc_status status;
        if (!fired[node]) {
            continue;
        }
        status = lc_emit_output_spike(t, node, &sink, stats, NULL);
        if (status != LC_OK) {
            return status;
        }

        states[node].value = models[node].reset;
        states[node].t_last = t;
        if (runtime[node].generation == UINT64_MAX) {
            return LC_NUMERIC_ERROR;
        }
        runtime[node].generation++;

        for (position = outgoing_offsets[node]; position < outgoing_offsets[node + 1U];
             ++position) {
            uint32_t edge = outgoing_edges[position];
            lc_time_t delivery_time;
            delivery_time = t + edges[edge].delay;
            if (!lc_isfinite(delivery_time) || delivery_time < t ||
                (edges[edge].delay > LC_REAL_C(0.0) && delivery_time == t)) {
                return LC_NUMERIC_ERROR;
            }
            status = lc_push_delivery(
                heap, edge, edges[edge].post, delivery_time, config, NULL
            );
            if (status != LC_OK) {
                return status;
            }
            if (delivery_time <= config->t_end) {
                stats->deliveries_scheduled++;
            }
        }

        if (models[node].refractory > LC_REAL_C(0.0)) {
            lc_event release;
            lc_time_t release_time = t + models[node].refractory;
            if (!lc_isfinite(release_time) || release_time <= t) {
                return LC_NUMERIC_ERROR;
            }
            runtime[node].clamped = 1;
            if (runtime[node].refractory_generation == UINT64_MAX) {
                return LC_NUMERIC_ERROR;
            }
            runtime[node].refractory_generation++;
            if (release_time <= config->t_end) {
                memset(&release, 0, sizeof(release));
                release.t = release_time;
                release.phase = LC_PHASE_BOUNDARY;
                release.kind = LC_EVENT_REFRACTORY_RELEASE;
                release.index = node;
                release.generation = runtime[node].refractory_generation;
                status = lc_heap_push(heap, release);
                if (status != LC_OK) {
                    return status;
                }
            }
        } else {
            status = lc_schedule_prediction(models, states, runtime, node, config, heap);
            if (status != LC_OK) {
                return status;
            }
        }
    }
    return LC_OK;
}

/* The scalar runner handles homogeneous LIF graphs with delta deposits. */

lc_status lc_delta_network_run(
    const lc_scalar_lif_model *models,
    lc_scalar_state *states,
    uint32_t node_count,
    const lc_delta_edge *edges,
    uint32_t edge_count,
    const lc_input_spike *inputs,
    uint32_t input_count,
    const lc_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_heap heap;
    lc_node_runtime *runtime = NULL;
    lc_scalar_lif_model *active_models = NULL;
    lc_real_t *deposits = NULL;
    uint8_t *affected = NULL;
    uint8_t *boundary_affected = NULL;
    uint8_t *fired = NULL;
    uint64_t *outgoing_offsets = NULL;
    uint64_t *outgoing_cursors = NULL;
    uint32_t *outgoing_edges = NULL;
    lc_status status = LC_OK;
    uint32_t node;
    uint32_t edge;
    uint32_t input;
    uint32_t drive_update;
    size_t csr_node;
    lc_profile_t kernel_start;

    memset(&heap, 0, sizeof(heap));
    if (models == NULL || states == NULL || config == NULL || output_count == NULL ||
        stats == NULL || node_count == 0 || config->queue_capacity == 0 ||
        config->same_time_cascade_limit == 0 || !lc_isfinite(config->t_end) ||
        config->t_end < LC_REAL_C(0.0) || (edge_count > 0 && edges == NULL) ||
        (input_count > 0 && inputs == NULL) ||
        (drive_update_count > 0 && drive_updates == NULL) ||
        (config->output_capacity > 0 && outputs == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *output_count = 0;
    memset(stats, 0, sizeof(*stats));
    kernel_start = lc_monotonic_seconds();

    for (node = 0; node < node_count; ++node) {
        if (!lc_valid_model(&models[node]) || !lc_isfinite(states[node].value) ||
            !lc_isfinite(states[node].t_last) || states[node].t_last < LC_REAL_C(0.0) ||
            states[node].t_last > config->t_end || states[node].value >= models[node].threshold) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (edge = 0; edge < edge_count; ++edge) {
        if (edges[edge].pre >= node_count || edges[edge].post >= node_count ||
            !lc_edge_weight_valid(
                models[edges[edge].pre].polarity, edges[edge].weight
            ) ||
            !lc_isfinite(edges[edge].delay) || edges[edge].delay < LC_REAL_C(0.0)) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (input = 0; input < input_count; ++input) {
        if (inputs[input].node >= node_count || !lc_isfinite(inputs[input].t) ||
            !lc_isfinite(inputs[input].value) || inputs[input].t < states[inputs[input].node].t_last) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        if (drive_updates[drive_update].node >= node_count ||
            !lc_isfinite(drive_updates[drive_update].t) ||
            !lc_isfinite(drive_updates[drive_update].b) ||
            drive_updates[drive_update].t < states[drive_updates[drive_update].node].t_last) {
            return LC_INVALID_ARGUMENT;
        }
    }

    if (lc_allocation_size_overflows(config->queue_capacity, sizeof(lc_event)) ||
        lc_allocation_size_overflows((uint64_t)node_count + 1U, sizeof(uint64_t)) ||
        lc_allocation_size_overflows(edge_count, sizeof(uint32_t))) {
        return LC_ALLOCATION_FAILED;
    }
    heap.items = calloc((size_t)config->queue_capacity, sizeof(lc_event));
    runtime = calloc(node_count, sizeof(lc_node_runtime));
    active_models = calloc(node_count, sizeof(lc_scalar_lif_model));
    deposits = calloc(node_count, sizeof(lc_real_t));
    affected = calloc(node_count, sizeof(uint8_t));
    boundary_affected = calloc(node_count, sizeof(uint8_t));
    fired = calloc(node_count, sizeof(uint8_t));
    outgoing_offsets = calloc((size_t)node_count + 1U, sizeof(uint64_t));
    outgoing_cursors = calloc(node_count, sizeof(uint64_t));
    if (edge_count > 0) {
        outgoing_edges = calloc(edge_count, sizeof(uint32_t));
    }
    if (heap.items == NULL || runtime == NULL || active_models == NULL || deposits == NULL ||
        affected == NULL || boundary_affected == NULL || fired == NULL ||
        outgoing_offsets == NULL || outgoing_cursors == NULL ||
        (edge_count > 0 && outgoing_edges == NULL)) {
        status = LC_ALLOCATION_FAILED;
        goto cleanup;
    }
    heap.capacity = config->queue_capacity;
    memcpy(active_models, models, node_count * sizeof(lc_scalar_lif_model));

    /*
     * Graph edges arrive in canonical order and retain that order per source.
     */
    for (edge = 0; edge < edge_count; ++edge) {
        outgoing_offsets[edges[edge].pre + 1U]++;
    }
    for (csr_node = 1; csr_node <= (size_t)node_count; ++csr_node) {
        outgoing_offsets[csr_node] += outgoing_offsets[csr_node - 1U];
    }
    memcpy(outgoing_cursors, outgoing_offsets, node_count * sizeof(uint64_t));
    for (edge = 0; edge < edge_count; ++edge) {
        uint32_t source = edges[edge].pre;
        outgoing_edges[outgoing_cursors[source]++] = edge;
    }

    for (node = 0; node < node_count; ++node) {
        runtime[node].generation = 1;
        status = lc_schedule_prediction(active_models, states, runtime, node, config, &heap);
        if (status != LC_OK) {
            goto cleanup;
        }
    }
    for (input = 0; input < input_count; ++input) {
        lc_event event;
        if (inputs[input].t > config->t_end) {
            continue;
        }
        memset(&event, 0, sizeof(event));
        event.t = inputs[input].t;
        event.phase = LC_PHASE_DEPOSIT;
        event.kind = LC_EVENT_INPUT_SPIKE;
        event.index = inputs[input].node;
        event.value = inputs[input].value;
        status = lc_heap_push(&heap, event);
        if (status != LC_OK) {
            goto cleanup;
        }
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        lc_event event;
        if (drive_updates[drive_update].t > config->t_end) {
            continue;
        }
        memset(&event, 0, sizeof(event));
        event.t = drive_updates[drive_update].t;
        event.phase = LC_PHASE_BOUNDARY;
        event.kind = LC_EVENT_DRIVE_UPDATE;
        event.index = drive_updates[drive_update].node;
        event.value = drive_updates[drive_update].b;
        status = lc_heap_push(&heap, event);
        if (status != LC_OK) {
            goto cleanup;
        }
    }

    while (heap.size > 0 && heap.items[0].t <= config->t_end) {
        lc_time_t t = heap.items[0].t;
        uint32_t cascade_depth = 0;
        memset(boundary_affected, 0, node_count * sizeof(uint8_t));

        while (lc_at(&heap, t, LC_PHASE_BOUNDARY)) {
            lc_event event = lc_heap_pop(&heap);
            stats->events_popped++;
            if (event.kind == LC_EVENT_DRIVE_UPDATE) {
                stats->drive_updates_processed++;
                status = lc_advance_node(active_models, states, runtime, event.index, t);
                if (status != LC_OK) {
                    goto cleanup;
                }
                active_models[event.index].b = event.value;
                boundary_affected[event.index] = 1;
            } else if (event.kind == LC_EVENT_REFRACTORY_RELEASE &&
                event.generation == runtime[event.index].refractory_generation &&
                runtime[event.index].clamped) {
                stats->refractory_releases_processed++;
                status = lc_advance_node(active_models, states, runtime, event.index, t);
                if (status != LC_OK) {
                    goto cleanup;
                }
                runtime[event.index].clamped = 0;
                boundary_affected[event.index] = 1;
            }
        }

        for (;;) {
            int any_fired = 0;
            memset(deposits, 0, node_count * sizeof(lc_real_t));
            memset(affected, 0, node_count * sizeof(uint8_t));
            memset(fired, 0, node_count * sizeof(uint8_t));
            for (node = 0; node < node_count; ++node) {
                if (boundary_affected[node]) {
                    affected[node] = 1;
                    boundary_affected[node] = 0;
                }
            }

            while (lc_at(&heap, t, LC_PHASE_DEPOSIT)) {
                lc_event event = lc_heap_pop(&heap);
                uint32_t target;
                lc_real_t value;
                stats->events_popped++;
                if (event.kind == LC_EVENT_DELIVERY) {
                    stats->deliveries_processed++;
                    target = edges[event.index].post;
                    value = lc_signed_edge_weight(
                        active_models[edges[event.index].pre].polarity,
                        edges[event.index].weight
                    );
                } else {
                    stats->input_spikes_processed++;
                    target = event.index;
                    value = event.value;
                }
                deposits[target] += value;
                if (!lc_isfinite(deposits[target])) {
                    status = LC_NUMERIC_ERROR;
                    goto cleanup;
                }
                affected[target] = 1;
            }

            for (node = 0; node < node_count; ++node) {
                if (!affected[node]) {
                    continue;
                }
                status = lc_advance_node(active_models, states, runtime, node, t);
                if (status != LC_OK) {
                    goto cleanup;
                }
                if (runtime[node].clamped) {
                    continue;
                }
                states[node].value += deposits[node];
                if (!lc_isfinite(states[node].value)) {
                    status = LC_NUMERIC_ERROR;
                    goto cleanup;
                }
                if (runtime[node].generation == UINT64_MAX) {
                    status = LC_NUMERIC_ERROR;
                    goto cleanup;
                }
                runtime[node].generation++;
                if (states[node].value >= active_models[node].threshold) {
                    fired[node] = 1;
                } else {
                    status = lc_schedule_prediction(
                        active_models, states, runtime, node, config, &heap
                    );
                    if (status != LC_OK) {
                        goto cleanup;
                    }
                }
            }

            while (lc_at(&heap, t, LC_PHASE_PREDICTION)) {
                lc_event event = lc_heap_pop(&heap);
                stats->events_popped++;
                if (event.generation != runtime[event.index].generation ||
                    runtime[event.index].clamped) {
                    stats->stale_predictions++;
                    continue;
                }
                status = lc_advance_node(active_models, states, runtime, event.index, t);
                if (status != LC_OK) {
                    goto cleanup;
                }
                fired[event.index] = 1;
                stats->autonomous_spikes_confirmed++;
            }

            for (node = 0; node < node_count; ++node) {
                if (fired[node]) {
                    any_fired = 1;
                    break;
                }
            }
            if (any_fired) {
                cascade_depth++;
                if (cascade_depth > config->same_time_cascade_limit) {
                    status = LC_CASCADE_LIMIT;
                    goto cleanup;
                }
                if (cascade_depth > stats->max_same_time_cascade_depth) {
                    stats->max_same_time_cascade_depth = cascade_depth;
                }
                status = lc_fire_nodes(
                    active_models, states, runtime, node_count, edges, outgoing_offsets,
                    outgoing_edges, config, &heap, fired, t, outputs, output_count, stats
                );
                if (status != LC_OK) {
                    goto cleanup;
                }
            }

            if (!lc_at(&heap, t, LC_PHASE_DEPOSIT)) {
                break;
            }
        }
    }

    for (node = 0; node < node_count; ++node) {
        status = lc_advance_node(active_models, states, runtime, node, config->t_end);
        if (status != LC_OK) {
            goto cleanup;
        }
    }

cleanup:
    stats->peak_queue_occupancy = heap.peak;
    stats->kernel_seconds = lc_monotonic_seconds() - kernel_start;
    free(heap.items);
    free(runtime);
    free(active_models);
    free(deposits);
    free(affected);
    free(boundary_affected);
    free(fired);
    free(outgoing_offsets);
    free(outgoing_cursors);
    free(outgoing_edges);
    return status;
}

/* Prepare the analytical readout modes used by learning observers. */
static lc_status lc_learning_readout_trajectory_prepare(
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_real_t *parameters,
    const lc_real_t *node_state,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_learning_readout_trajectory *trajectory
) {
    uint32_t index;
    lc_status status;
    if (descriptor == NULL || program_nodes == NULL || node_state == NULL ||
        variables == NULL || workspace == NULL || trajectory == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    memset(trajectory, 0, sizeof(*trajectory));
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE) {
        if (!lc_isfinite(descriptor->scalar_affine_decay) ||
            descriptor->scalar_affine_decay >= LC_REAL_C(0.0) ||
            !lc_isfinite(descriptor->scalar_affine_drive)) {
            return LC_INVALID_ARGUMENT;
        }
        trajectory->mode_count = 1U;
        trajectory->limit = -descriptor->scalar_affine_drive /
            descriptor->scalar_affine_decay;
        trajectory->coefficients[0] =
            node_state[descriptor->readout] - trajectory->limit;
        trajectory->rates[0] = descriptor->scalar_affine_decay;
    } else if (descriptor->crossing_kind == LC_CROSSING_TWO_REAL_EXP) {
        const lc_two_exp_hint *hint = &descriptor->two_exp_hint;
        variables[0] = LC_REAL_C(0.0);
        for (index = 0U; index < descriptor->state_count; ++index) {
            variables[index + 1U] = node_state[index];
        }
        status = lc_expr_evaluate(
            program_nodes, descriptor->program_node_count,
            parameters, descriptor->parameter_count, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
        if (status != LC_OK) {
            return status;
        }
        trajectory->mode_count = 2U;
        trajectory->limit = descriptor->threshold - workspace[hint->limit_root];
        trajectory->coefficients[0] =
            -workspace[hint->coefficient_one_root];
        trajectory->coefficients[1] =
            -workspace[hint->coefficient_two_root];
        trajectory->rates[0] = workspace[hint->rate_one_root];
        trajectory->rates[1] = workspace[hint->rate_two_root];
    } else {
        return LC_INVALID_ARGUMENT;
    }
    if (!lc_isfinite(trajectory->limit)) {
        return LC_NUMERIC_ERROR;
    }
    for (index = 0U; index < trajectory->mode_count; ++index) {
        if (!lc_isfinite(trajectory->coefficients[index]) ||
            !lc_isfinite(trajectory->rates[index]) ||
            trajectory->rates[index] >= LC_REAL_C(0.0)) {
            return LC_NUMERIC_ERROR;
        }
    }
    return LC_OK;
}

/* Convolve one exponential voltage mode with an observer decay. */
static lc_real_t lc_learning_voltage_mode_convolution(
    lc_real_t rate,
    lc_time_t duration,
    lc_real_t tau,
    lc_real_t trace_decay
) {
    lc_real_t rate_difference = rate + LC_REAL_C(1.0) / tau;
    lc_real_t scaled_difference = rate_difference * (lc_real_t)duration;
    if (rate_difference == LC_REAL_C(0.0) ||
        scaled_difference == LC_REAL_C(0.0)) {
        return (lc_real_t)duration * trace_decay / tau;
    }
    if (lc_real_fabs(scaled_difference) < LC_REAL_C(1.0e-4)) {
        return (lc_real_t)duration * trace_decay *
            lc_real_expm1(scaled_difference) / scaled_difference / tau;
    }
    return (lc_real_exp(rate * (lc_real_t)duration) - trace_decay) /
        (tau * rate_difference);
}

/* Advance neuron-local voltage statistics along the exact trajectory. */
static lc_status lc_learning_observer_advance_voltage(
    const lc_compiled_graph *graph,
    lc_learning_observer_state *states,
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    const lc_node_runtime *runtime,
    const lc_real_t *node_state,
    const lc_real_t *parameters,
    uint32_t node,
    lc_time_t from,
    lc_time_t to,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t program_index;
    uint32_t parameter_offset;
    const lc_learning_program *program;
    lc_real_t tau;
    lc_time_t duration;
    lc_real_t decay;
    lc_real_t slow;
    uint32_t mode;
    lc_status status;
    lc_learning_readout_trajectory trajectory;
    if (graph == NULL || states == NULL ||
        graph->learning_observer_program_by_node == NULL) {
        return LC_OK;
    }
    program_index = graph->learning_observer_program_by_node[node];
    if (program_index == UINT32_MAX) {
        return LC_OK;
    }
    if (descriptor == NULL || runtime == NULL || node_state == NULL ||
        program_index >= graph->learning_program_count ||
        !lc_isfinite(from) || !lc_isfinite(to) || to < from) {
        return LC_INVALID_ARGUMENT;
    }
    duration = to - from;
    if (duration == LC_REAL_C(0.0)) {
        return LC_OK;
    }
    if (!lc_isfinite((lc_real_t)duration) || (lc_real_t)duration == LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    program = &graph->learning_programs[program_index];
    parameter_offset = graph->learning_observer_parameter_offset_by_node[node];
    tau = graph->learning_parameters[
        parameter_offset + program->observer.voltage_tau_parameter
    ];
    decay = lc_real_exp(-(lc_real_t)duration / tau);
    slow = states[node].slow_voltage;
    if (runtime->clamped) {
        lc_real_t voltage = node_state[descriptor->readout];
        slow = voltage + (slow - voltage) * decay;
    } else {
        status = lc_learning_readout_trajectory_prepare(
            descriptor, program_nodes, parameters, node_state, variables, workspace,
            workspace_count, &trajectory
        );
        if (status != LC_OK) {
            return status;
        }
        slow = trajectory.limit + (slow - trajectory.limit) * decay;
        for (mode = 0U; mode < trajectory.mode_count; ++mode) {
            slow += trajectory.coefficients[mode] *
                lc_learning_voltage_mode_convolution(
                    trajectory.rates[mode], duration, tau, decay
                );
        }
    }
    if (!lc_isfinite(slow)) {
        return LC_NUMERIC_ERROR;
    }
    states[node].slow_voltage = slow;
    return LC_OK;
}

/* Advance one mixed node using its equation-derived arithmetic plan. */
static lc_status lc_mixed_advance_node_planned(
    const lc_mixed_node *nodes,
    lc_real_t *state,
    lc_time_t *t_last,
    const lc_real_t *parameters,
    lc_node_runtime *runtime,
    uint32_t node,
    lc_time_t t,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    const lc_node_eval_plan *plans,
    const lc_compiled_graph *learning_graph,
    lc_learning_observer_state *learning_observers
) {
    const lc_mixed_node *descriptor = &nodes[node];
    const lc_expr_node *program_nodes = descriptor->program_nodes;
    lc_real_t *node_state = &state[descriptor->state_offset];
    lc_status observer_status = lc_learning_observer_advance_voltage(
        learning_graph, learning_observers, descriptor,
        plans == NULL ? descriptor->program_nodes : plans[node].crossing_nodes,
        &runtime[node],
        node_state,
        descriptor->parameter_count > 0U
            ? &parameters[descriptor->parameter_offset] : NULL,
        node, t_last[node], t, variables, workspace, workspace_count
    );
    if (observer_status != LC_OK) {
        return observer_status;
    }
    if (descriptor->crossing_kind == LC_CROSSING_INTEGRATED_HAZARD &&
        !runtime[node].clamped && t > t_last[node]) {
        const lc_expr_node *hazard_nodes = plans == NULL
            ? descriptor->program_nodes : plans[node].crossing_nodes;
        lc_real_t consumed;
        lc_status hazard_status = lc_hazard_integral(
            descriptor, hazard_nodes,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset] : NULL,
            node_state, t - t_last[node], variables, workspace,
            workspace_count, &consumed
        );
        if (hazard_status != LC_OK) {
            return hazard_status;
        }
        runtime[node].hazard_remaining =
            consumed >= runtime[node].hazard_remaining
                ? LC_REAL_C(0.0) : runtime[node].hazard_remaining - consumed;
    }
    if (lc_mixed_scalar_affine_enabled(descriptor, &runtime[node])) {
        return lc_mixed_scalar_affine_advance(
            descriptor, &runtime[node], node_state, &t_last[node], t
        );
    }
    if (descriptor->dispatch == LC_STEPPED) {
        lc_step_result step_result;
        if (t < t_last[node]) return LC_TIME_REVERSED;
        if (t == t_last[node]) return LC_OK;
        if (lc_runtime_sample_cache(&runtime[node], t, node_state,
                                    descriptor->state_count)) {
            t_last[node] = t;
            return LC_OK;
        }
        return lc_expr_step_advance(
            descriptor->program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count, descriptor->normal_roots,
            descriptor->state_count, descriptor->readout,
            &descriptor->step_config, node_state, &t_last[node], t,
            runtime[node].clamped != 0, variables,
            descriptor->state_count + 1U, workspace, workspace_count,
            &step_result
        );
    }
    if (plans != NULL) {
        program_nodes = runtime[node].clamped
            ? plans[node].clamped_nodes : plans[node].normal_nodes;
    }
    return lc_expr_state_advance(
        program_nodes, descriptor->program_node_count,
        descriptor->parameter_count > 0U
            ? &parameters[descriptor->parameter_offset]
            : NULL,
        descriptor->parameter_count,
        runtime[node].clamped ? descriptor->clamped_roots : descriptor->normal_roots,
        descriptor->state_count, node_state, &t_last[node], t, variables,
        descriptor->state_count + 1U, workspace, workspace_count
    );
}

/* Select normal or clamped propagation for one mixed node. */
static lc_status lc_mixed_advance_node(
    const lc_mixed_node *nodes,
    lc_real_t *state,
    lc_time_t *t_last,
    const lc_real_t *parameters,
    lc_node_runtime *runtime,
    uint32_t node,
    lc_time_t t,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    return lc_mixed_advance_node_planned(
        nodes, state, t_last, parameters, runtime, node, t, variables,
        workspace, workspace_count, NULL, NULL, NULL
    );
}

/* Predict one mixed node's next spike with its resolved crossing method. */
static lc_status lc_mixed_schedule_prediction_planned(
    const lc_mixed_node *nodes,
    const lc_real_t *state,
    const lc_time_t *t_last,
    const lc_real_t *parameters,
    lc_node_runtime *runtime,
    uint32_t node,
    const lc_run_config *config,
    lc_heap *heap,
    lc_network_error *error,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    const lc_node_eval_plan *plans
) {
    const lc_mixed_node *descriptor = &nodes[node];
    const lc_expr_node *program_nodes = plans == NULL
        ? descriptor->program_nodes : plans[node].crossing_nodes;
    lc_time_t t_spike = LC_REAL_C(0.0);
    lc_status status;
    lc_event event;
    if (runtime[node].clamped) {
        return LC_OK;
    }
    if (t_last[node] >= config->t_end) return LC_OK;
    if (descriptor->crossing_kind == LC_CROSSING_INTEGRATED_HAZARD) {
        lc_time_t horizon = config->t_end - t_last[node];
        lc_time_t duration = NAN;
        if (!runtime[node].hazard_initialized) {
            status = lc_hazard_draw(runtime + node, config->stochastic_seed, node);
            if (status != LC_OK) return status;
        }
        if (horizon <= LC_REAL_C(0.0)) {
            return LC_OK;
        }
        status = lc_hazard_predict_duration(
            descriptor, program_nodes,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset] : NULL,
            &state[descriptor->state_offset], runtime[node].hazard_remaining,
            horizon, variables, workspace, workspace_count, &duration
        );
        if (status == LC_NO_CROSSING) {
            return LC_OK;
        }
        if (status != LC_OK) return status;
        if (!lc_isfinite(duration) || duration < LC_REAL_C(0.0) || duration > horizon) {
            return LC_NUMERIC_ERROR;
        }
        t_spike = t_last[node] + duration;
        if (t_spike <= t_last[node]) {
#if LACUNA_REAL_BITS < 64
            return LC_NUMERIC_ERROR;
#else
            t_spike = lc_time_nextafter(t_last[node], INFINITY);
#endif
        }
        status = LC_OK;
    } else if (lc_mixed_scalar_affine_enabled(descriptor, &runtime[node])) {
        lc_scalar_lif_model model;
        lc_scalar_state scalar_state;
        lc_dispatch_form dispatch = LC_REACTIVE;
        model.a = descriptor->scalar_affine_decay;
        model.b = descriptor->scalar_affine_drive;
        model.threshold = descriptor->threshold;
        model.reset = descriptor->scalar_affine_reset;
        model.refractory = descriptor->refractory;
        model.polarity = descriptor->polarity;
        scalar_state.value = state[descriptor->state_offset];
        scalar_state.t_last = t_last[node];
        status = lc_scalar_predict(
            &model, &scalar_state, &t_spike, &dispatch
        );
    } else if (descriptor->crossing_kind == LC_CROSSING_REACTIVE) {
        return LC_OK;
    } else if (descriptor->crossing_kind == LC_CROSSING_SCALAR_LOG) {
        status = lc_expr_scalar_log_predict(
            program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count, &descriptor->scalar_log_hint,
            &state[descriptor->state_offset], descriptor->state_count,
            descriptor->readout, t_last[node], &t_spike, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
    } else if (descriptor->crossing_kind == LC_CROSSING_ALPHA_REAL) {
        lc_root_result root;
        status = lc_expr_alpha_predict(
            program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count,
            &descriptor->root_hint, &state[descriptor->state_offset],
            descriptor->state_count, t_last[node], &root, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
        if (status == LC_OK) {
            t_spike = root.t_spike;
        } else if (status == LC_ROOT_NONCONVERGENCE && error != NULL) {
            error->node = node;
            error->t = t_last[node];
            error->root = root;
        }
    } else if (descriptor->crossing_kind == LC_CROSSING_TWO_REAL_EXP) {
        lc_root_result root;
        status = lc_expr_two_exp_predict(
            program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count,
            &descriptor->two_exp_hint, &state[descriptor->state_offset],
            descriptor->state_count, t_last[node], &root, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
        if (status == LC_OK) {
            t_spike = root.t_spike;
        } else if (status == LC_ROOT_NONCONVERGENCE && error != NULL) {
            error->node = node;
            error->t = t_last[node];
            error->root = root;
        }
    } else if (descriptor->crossing_kind == LC_CROSSING_MULTI_REAL_EXP) {
        lc_root_result root;
        status = lc_expr_multi_exp_predict(
            program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count,
            &descriptor->multi_exp_hint, &state[descriptor->state_offset],
            descriptor->state_count, t_last[node], &root, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
        if (status == LC_OK) {
            t_spike = root.t_spike;
        } else if (status == LC_ROOT_NONCONVERGENCE && error != NULL) {
            error->node = node;
            error->t = t_last[node];
            error->root = root;
        }
    } else if (descriptor->crossing_kind == LC_CROSSING_REPEATED_REAL_MODE ||
               descriptor->crossing_kind == LC_CROSSING_MULTI_EXP_POLY) {
        lc_root_result root;
        status = lc_expr_exp_poly_predict(
            program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count,
            &descriptor->exp_poly_hint, &state[descriptor->state_offset],
            descriptor->state_count, t_last[node], &root, variables,
            descriptor->state_count + 1U, workspace, workspace_count
        );
        if (status == LC_OK) {
            t_spike = root.t_spike;
        } else if (status == LC_ROOT_NONCONVERGENCE && error != NULL) {
            error->node = node;
            error->t = t_last[node];
            error->root = root;
        }
    } else if (descriptor->crossing_kind == LC_CROSSING_NUMERICAL) {
        lc_step_result step_result;
#if LACUNA_REAL_BITS == 64 && LACUNA_TIME_BITS == 64
        /* Bound speculation independently of total simulation duration. A full
         * history also ends the search early. Either case schedules a wakeup,
         * never treats a local no-crossing certificate as a global one. */
        lc_time_t span = descriptor->step_config.maximum_step;
        lc_time_t horizon = lc_time_fmin(config->t_end, t_last[node] + span);
        if (!(horizon > t_last[node])) return LC_NUMERIC_ERROR;
        if (runtime[node].step_cache == NULL) {
            runtime[node].step_cache = calloc(1U, sizeof(lc_step_cache));
            if (runtime[node].step_cache == NULL) return LC_ALLOCATION_FAILED;
        }
        if (runtime[node].cache_generation != runtime[node].generation) {
            runtime[node].step_cache->resume_valid = 0U;
        }
        runtime[node].cache_generation = runtime[node].generation;
        status = lc_expr_step_predict_cached(
            descriptor->program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U ? &parameters[descriptor->parameter_offset] : NULL,
            descriptor->parameter_count, descriptor->normal_roots,
            descriptor->state_count, descriptor->readout, descriptor->threshold,
            &descriptor->step_config, &state[descriptor->state_offset],
            t_last[node], horizon, variables, descriptor->state_count + 1U,
            workspace, workspace_count, &step_result, runtime[node].step_cache);
        if (status == LC_NO_CROSSING) {
            if (step_result.t_reached >= config->t_end) return LC_OK;
            if (!(step_result.t_reached > t_last[node])) return LC_NUMERIC_ERROR;
            memset(&event, 0, sizeof(event));
            event.t = step_result.t_reached;
            event.phase = LC_PHASE_PREDICTION;
            event.kind = LC_EVENT_NUMERICAL_CONTINUATION;
            event.index = node;
            event.generation = runtime[node].generation;
            return lc_heap_push_report(heap, event, error);
        }
#else
        status = lc_expr_step_predict(
            descriptor->program_nodes, descriptor->program_node_count,
            descriptor->parameter_count > 0U
                ? &parameters[descriptor->parameter_offset]
                : NULL,
            descriptor->parameter_count, descriptor->normal_roots,
            descriptor->state_count, descriptor->readout, descriptor->threshold,
            &descriptor->step_config, &state[descriptor->state_offset],
            t_last[node], config->t_end, variables,
            descriptor->state_count + 1U, workspace, workspace_count,
            &step_result
        );
#endif
        if (status == LC_OK) {
            t_spike = step_result.t_crossing;
        }
    } else {
        return LC_UNSUPPORTED_MODEL;
    }
    if (status == LC_NO_CROSSING) {
        return LC_OK;
    }
    if (status != LC_OK) {
        return status;
    }
#if LACUNA_REAL_BITS == 16
    if (!lc_isfinite(t_spike) || t_spike <= t_last[node]) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (t_spike > config->t_end) {
        return LC_OK;
    }
    memset(&event, 0, sizeof(event));
    event.t = t_spike;
    event.phase = LC_PHASE_PREDICTION;
    event.kind = LC_EVENT_AUTONOMOUS_SPIKE;
    event.index = node;
    event.generation = runtime[node].generation;
    return lc_heap_push_report(heap, event, error);
}

/* Schedule a new prediction and invalidate the previous generation. */
static lc_status lc_mixed_schedule_prediction(
    const lc_mixed_node *nodes,
    const lc_real_t *state,
    const lc_time_t *t_last,
    const lc_real_t *parameters,
    lc_node_runtime *runtime,
    uint32_t node,
    const lc_run_config *config,
    lc_heap *heap,
    lc_network_error *error,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    return lc_mixed_schedule_prediction_planned(
        nodes, state, t_last, parameters, runtime, node, config, heap, error,
        variables, workspace, workspace_count, NULL
    );
}

/* Decay one legacy plasticity trace exactly to the requested time. */
static lc_status lc_plastic_decay(
    lc_real_t *value,
    lc_time_t *t_last,
    lc_time_t t,
    lc_real_t tau
) {
    lc_real_t factor;
    if (value == NULL || t_last == NULL || !lc_isfinite(*value) ||
        !lc_isfinite(*t_last) || !lc_isfinite(t) || !lc_isfinite(tau) || tau <= LC_REAL_C(0.0) ||
        t < *t_last) {
        return LC_NUMERIC_ERROR;
    }
    if (!lc_isfinite((lc_real_t)(t - *t_last)) ||
        (t > *t_last && (lc_real_t)(t - *t_last) == LC_REAL_C(0.0))) {
        return LC_NUMERIC_ERROR;
    }
    factor = lc_real_exp(-(lc_real_t)(t - *t_last) / tau);
    *value *= factor;
    *t_last = t;
    return lc_isfinite(*value) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Convert a bounded physical weight into its normalized learning coordinate. */
static lc_real_t lc_plastic_normalized_weight(
    const lc_plasticity_rule *rule,
    lc_real_t weight
) {
    return (weight - rule->weight_min) / (rule->weight_max - rule->weight_min);
}

/* Read an edge-local or shared physical weight. */
static lc_real_t lc_plastic_slot_weight(
    const lc_compiled_graph *graph,
    const lc_plasticity_state *states,
    uint32_t slot
) {
    uint32_t master;
    if (graph == NULL || states == NULL || slot >= graph->plastic_edge_count) {
        return NAN;
    }
    master = graph->weight_master_slots == NULL
        ? slot : graph->weight_master_slots[slot];
    if (master >= graph->plastic_edge_count) {
        return NAN;
    }
    return states[master].weight;
}

/* Clamp and distribute a normalized legacy plasticity update. */
static lc_status lc_plastic_set_normalized_weight(
    const lc_compiled_graph *graph,
    const lc_plasticity_rule *rule,
    lc_plasticity_state *states,
    uint32_t slot,
    lc_real_t normalized
) {
    uint32_t master;
    if (normalized < LC_REAL_C(0.0)) {
        normalized = LC_REAL_C(0.0);
    } else if (normalized > LC_REAL_C(1.0)) {
        normalized = LC_REAL_C(1.0);
    }
    if (graph == NULL || states == NULL || slot >= graph->plastic_edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    master = graph->weight_master_slots == NULL
        ? slot : graph->weight_master_slots[slot];
    if (master >= graph->plastic_edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    states[master].weight = rule->weight_min +
        normalized * (rule->weight_max - rule->weight_min);
    return lc_isfinite(states[master].weight) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Scale shared updates by the number of contributing edge copies. */
static lc_real_t lc_plastic_learning_scale(
    const lc_compiled_graph *graph,
    uint32_t slot
) {
    if (graph == NULL || slot >= graph->plastic_edge_count) {
        return NAN;
    }
    return graph->weight_learning_scales == NULL
        ? LC_REAL_C(1.0) : graph->weight_learning_scales[slot];
}

static lc_real_t *lc_learning_trace_value(
    lc_plasticity_state *state,
    uint32_t storage_slot
) {
    if (state == NULL) {
        return NULL;
    }
    switch (storage_slot) {
        case 0U: return &state->pre_fast;
        case 1U: return &state->post_fast;
        case 2U: return &state->pre_slow;
        case 3U: return &state->post_slow;
        case 4U: return &state->eligibility_plus;
        case 5U: return &state->eligibility_minus;
        default: return NULL;
    }
}

static lc_time_t *lc_learning_trace_time(
    lc_plasticity_state *state,
    uint32_t storage_slot
) {
    if (state == NULL) {
        return NULL;
    }
    switch (storage_slot) {
        case 0U: return &state->t_pre_fast;
        case 1U: return &state->t_post_fast;
        case 2U: return &state->t_pre_slow;
        case 3U: return &state->t_post_slow;
        case 4U: return &state->t_eligibility_plus;
        case 5U: return &state->t_eligibility_minus;
        default: return NULL;
    }
}

/* Commit one equation-derived weight update to its sharing group. */
static lc_status lc_learning_set_normalized_weight(
    const lc_compiled_graph *graph,
    const lc_learning_program *program,
    const lc_learning_binding *binding,
    lc_plasticity_state *states,
    uint32_t slot,
    lc_real_t normalized
) {
    uint32_t master;
    if (graph == NULL || program == NULL || binding == NULL || states == NULL ||
        slot >= graph->plastic_edge_count || !lc_isfinite(normalized)) {
        return LC_INVALID_ARGUMENT;
    }
    if (program->clamp_normalized_weight != 0U) {
        if (normalized < LC_REAL_C(0.0)) {
            normalized = LC_REAL_C(0.0);
        } else if (normalized > LC_REAL_C(1.0)) {
            normalized = LC_REAL_C(1.0);
        }
    }
    master = graph->weight_master_slots == NULL
        ? slot : graph->weight_master_slots[slot];
    if (master >= graph->plastic_edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    states[master].weight = binding->weight_min +
        normalized * (binding->weight_max - binding->weight_min);
    return lc_isfinite(states[master].weight) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Learning equations update only the traces selected for this event kind. */
static lc_status lc_learning_execute_event(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t slot,
    const lc_learning_program *selected_program,
    uint32_t event_kind,
    lc_real_t modulation,
    lc_real_t post_readout,
    lc_real_t observation_gain,
    lc_real_t event_amplitude,
    lc_real_t input_accepted,
    lc_time_t t,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_real_t *weight_deltas,
    uint8_t *weight_touched,
    uint32_t *weight_masters,
    uint32_t *weight_master_count
) {
    uint32_t edge;
    const lc_learning_binding *binding;
    const lc_learning_program *program;
    const lc_learning_event_program *event;
    const lc_real_t *parameters;
    lc_plasticity_state *state;
    lc_real_t normalized = LC_REAL_C(0.0);
    uint32_t trace;
    uint32_t update;
    uint32_t base_variable_count;
    lc_status status;
    int accumulate_weight = weight_deltas != NULL;
    if (graph == NULL || states == NULL || variables == NULL || workspace == NULL ||
        slot >= graph->plastic_edge_count ||
        event_kind >= LC_LEARNING_EVENT_COUNT || !lc_isfinite(post_readout) ||
        !lc_isfinite(observation_gain) || !lc_isfinite(event_amplitude) ||
        (input_accepted != LC_REAL_C(0.0) && input_accepted != LC_REAL_C(1.0)) ||
        (accumulate_weight && (weight_touched == NULL || weight_masters == NULL ||
                               weight_master_count == NULL)) ||
        (!accumulate_weight && (weight_touched != NULL || weight_masters != NULL ||
                                weight_master_count != NULL))) {
        return LC_INVALID_ARGUMENT;
    }
    edge = graph->plastic_edges[slot];
    binding = &graph->learning_bindings[edge];
    if (binding->program >= graph->learning_program_count) {
        return LC_INVALID_ARGUMENT;
    }
    program = &graph->learning_programs[binding->program];
    if (selected_program != NULL) {
        if (selected_program != program) {
            return LC_INVALID_ARGUMENT;
        }
        program = selected_program;
    }
    base_variable_count = program->variable_count - program->trace_count;
    event = &program->events[event_kind];
    if (event->node_count == 0U) {
        return LC_OK;
    }
    parameters = program->parameter_count == 0U
        ? NULL : &graph->learning_parameters[binding->parameter_offset];
    state = &states[slot];
    for (trace = 0U; trace < program->trace_count; ++trace) {
        if ((event->advance_mask & (UINT32_C(1) << trace)) != 0U) {
            lc_real_t *value = lc_learning_trace_value(
                state, program->trace_storage_slots[trace]
            );
            lc_time_t *last = lc_learning_trace_time(
                state, program->trace_storage_slots[trace]
            );
            status = lc_plastic_decay(
                value, last, t, parameters[program->trace_tau_parameters[trace]]
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    if ((event->variable_mask & UINT32_C(1)) != 0U ||
        (accumulate_weight && event->weight_root != UINT32_MAX)) {
        normalized = (
            lc_plastic_slot_weight(graph, states, slot) - binding->weight_min
        ) / (binding->weight_max - binding->weight_min);
        if (!lc_isfinite(normalized)) {
            return LC_NUMERIC_ERROR;
        }
        if ((event->variable_mask & UINT32_C(1)) != 0U) {
            variables[0] = normalized;
        }
    }
    if ((event->variable_mask & (UINT32_C(1) << 1U)) != 0U) {
        variables[1] = modulation;
    }
    if ((event->variable_mask & (UINT32_C(1) << 2U)) != 0U) {
        variables[2] = lc_plastic_learning_scale(graph, slot);
    }
    if ((event->variable_mask & (UINT32_C(1) << 3U)) != 0U) {
        variables[3] = post_readout;
    }
    if (base_variable_count >= 6U &&
        (event->variable_mask & (UINT32_C(1) << 4U)) != 0U) {
        variables[4] = observation_gain;
    }
    if (base_variable_count >= 6U &&
        (event->variable_mask & (UINT32_C(1) << 5U)) != 0U) {
        variables[5] = event_amplitude;
    }
    if ((base_variable_count == 5U || base_variable_count == 7U) &&
        (event->variable_mask & (
            UINT32_C(1) << (base_variable_count - 1U))) != 0U) {
        variables[base_variable_count - 1U] = input_accepted;
    }
    for (trace = 0U; trace < program->trace_count; ++trace) {
        if ((event->variable_mask & (
                UINT32_C(1) << (base_variable_count + trace)
            )) != 0U) {
            lc_real_t *value = lc_learning_trace_value(
                state, program->trace_storage_slots[trace]
            );
            if (value == NULL) {
                return LC_INVALID_ARGUMENT;
            }
            variables[base_variable_count + trace] = *value;
        }
    }
    status = lc_expr_evaluate(
        event->nodes, event->node_count, parameters, program->parameter_count,
        variables, program->variable_count, workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    if (event->weight_root != UINT32_MAX) {
        if (accumulate_weight) {
            uint32_t master = graph->weight_master_slots == NULL
                ? slot : graph->weight_master_slots[slot];
            lc_real_t delta = workspace[event->weight_root] - normalized;
            if (master >= graph->plastic_edge_count || !lc_isfinite(delta)) {
                return LC_NUMERIC_ERROR;
            }
            weight_deltas[master] += delta;
            if (!lc_isfinite(weight_deltas[master])) {
                return LC_NUMERIC_ERROR;
            }
            if (weight_touched[master] == 0U) {
                if (*weight_master_count >= graph->plastic_edge_count) {
                    return LC_INVALID_ARGUMENT;
                }
                weight_touched[master] = 1U;
                weight_masters[*weight_master_count] = master;
                (*weight_master_count)++;
            }
        } else {
            status = lc_learning_set_normalized_weight(
                graph, program, binding, states, slot,
                workspace[event->weight_root]
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    for (update = 0U; update < event->trace_update_count; ++update) {
        uint32_t trace_index = event->trace_indices[update];
        lc_real_t *value = lc_learning_trace_value(
            state, program->trace_storage_slots[trace_index]
        );
        *value = workspace[event->trace_roots[update]];
        if (!lc_isfinite(*value)) {
            return LC_NUMERIC_ERROR;
        }
    }
    return LC_OK;
}

/* Execute presynaptic learning programs for one delivered edge. */
static lc_status lc_learning_pre(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t edge,
    lc_time_t t,
    lc_real_t post_readout,
    lc_real_t input_accepted,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t slot;
    if (graph == NULL || states == NULL || edge >= graph->edge_count) {
        return LC_OK;
    }
    slot = graph->plastic_slot_by_edge[edge];
    return slot == UINT32_MAX ? LC_OK : lc_learning_execute_event(
        graph, states, slot, NULL, LC_LEARNING_PRE_SPIKE, LC_REAL_C(0.0),
        post_readout, LC_REAL_C(0.0), LC_REAL_C(0.0), input_accepted, t,
        variables, workspace, workspace_count, NULL, NULL, NULL, NULL
    );
}

/* Execute postsynaptic learning batches for one firing neuron. */
static lc_status lc_learning_post(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t node,
    lc_time_t t,
    lc_real_t post_readout,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint64_t batch_position;
    if (graph == NULL || states == NULL || graph->plastic_edge_count == 0U) {
        return LC_OK;
    }
    if (node >= graph->node_count || graph->incoming_learning_batch_offsets == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    for (batch_position = graph->incoming_learning_batch_offsets[node];
         batch_position < graph->incoming_learning_batch_offsets[node + 1U];
         ++batch_position) {
        const lc_learning_batch *batch =
            &graph->incoming_learning_batches[batch_position];
        const lc_learning_program *program;
        uint64_t position;
        if (batch->program >= graph->learning_program_count) {
            return LC_INVALID_ARGUMENT;
        }
        program = &graph->learning_programs[batch->program];
        for (position = batch->slot_offset;
             position < batch->slot_offset + batch->slot_count; ++position) {
            lc_status status = lc_learning_execute_event(
                graph, states, graph->incoming_plastic_slots[position], program,
                LC_LEARNING_POST_SPIKE, LC_REAL_C(0.0), post_readout, LC_REAL_C(0.0), LC_REAL_C(1.0), LC_REAL_C(0.0), t,
                variables, workspace, workspace_count, NULL, NULL, NULL, NULL
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    return LC_OK;
}

/* Update neuron-local observers and distribute their event amplitude. */
static lc_status lc_learning_observe(
    const lc_compiled_graph *graph,
    lc_plasticity_state *edge_states,
    lc_learning_observer_state *observer_states,
    uint32_t node,
    lc_time_t t,
    lc_real_t post_readout,
    lc_real_t event_amplitude,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t program_index;
    uint32_t parameter_offset;
    const lc_learning_program *program;
    const lc_learning_observer_program *observer;
    const lc_real_t *parameters;
    lc_learning_observer_state *state;
    lc_time_t elapsed;
    lc_real_t fast_decay;
    lc_real_t slow_decay;
    lc_real_t observer_variables[6];
    lc_real_t gain;
    uint64_t batch_position;
    lc_status status;
    if (graph == NULL || edge_states == NULL || observer_states == NULL ||
        node >= graph->node_count || !lc_isfinite(t) ||
        !lc_isfinite(post_readout) || !lc_isfinite(event_amplitude) ||
        event_amplitude <= LC_REAL_C(0.0) || event_amplitude > LC_REAL_C(1.0) ||
        variables == NULL || workspace == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    program_index = graph->learning_observer_program_by_node[node];
    if (program_index == UINT32_MAX) {
        return LC_OK;
    }
    program = &graph->learning_programs[program_index];
    observer = &program->observer;
    parameter_offset = graph->learning_observer_parameter_offset_by_node[node];
    parameters = &graph->learning_parameters[parameter_offset];
    state = &observer_states[node];
    elapsed = t - state->t_activity;
    if (!lc_isfinite(elapsed) || elapsed < LC_REAL_C(0.0)) {
        return LC_INVALID_ARGUMENT;
    }
    if (!lc_isfinite((lc_real_t)elapsed) ||
        (elapsed > LC_REAL_C(0.0) && (lc_real_t)elapsed == LC_REAL_C(0.0))) {
        return LC_NUMERIC_ERROR;
    }
    fast_decay = lc_real_exp(
        -(lc_real_t)elapsed / parameters[observer->fast_activity_tau_parameter]
    );
    slow_decay = lc_real_exp(
        -(lc_real_t)elapsed / parameters[observer->slow_activity_tau_parameter]
    );
    state->fast_activity *= fast_decay;
    state->slow_activity *= slow_decay;
    state->t_activity = t;
    observer_variables[0] = post_readout;
    observer_variables[1] = graph->nodes[node].threshold;
    observer_variables[2] = event_amplitude;
    observer_variables[3] = state->slow_voltage;
    observer_variables[4] = state->fast_activity;
    observer_variables[5] = state->slow_activity;
    status = lc_expr_evaluate(
        observer->nodes, observer->node_count, parameters,
        program->parameter_count, observer_variables, 6U, workspace,
        workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    state->fast_activity = workspace[observer->fast_activity_root];
    state->slow_activity = workspace[observer->slow_activity_root];
    gain = workspace[observer->gain_root];
    if (!lc_isfinite(state->fast_activity) || !lc_isfinite(state->slow_activity) ||
        !lc_isfinite(gain) || gain < LC_REAL_C(0.0)) {
        return LC_NUMERIC_ERROR;
    }
    state->sensitivity += gain;
    if (!lc_isfinite(state->sensitivity)) {
        return LC_NUMERIC_ERROR;
    }
    for (batch_position = graph->incoming_learning_batch_offsets[node];
         batch_position < graph->incoming_learning_batch_offsets[node + 1U];
         ++batch_position) {
        const lc_learning_batch *batch =
            &graph->incoming_learning_batches[batch_position];
        const lc_learning_program *edge_program =
            &graph->learning_programs[batch->program];
        uint64_t position;
        if (edge_program->observer.node_count == 0U) {
            continue;
        }
        for (position = batch->slot_offset;
             position < batch->slot_offset + batch->slot_count; ++position) {
            status = lc_learning_execute_event(
                graph, edge_states,
                graph->incoming_plastic_slots[position], edge_program,
                LC_LEARNING_OBSERVATION, LC_REAL_C(0.0), post_readout, gain,
                event_amplitude, LC_REAL_C(0.0), t, variables, workspace, workspace_count,
                NULL, NULL, NULL, NULL
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    return LC_OK;
}

/* Evaluate the compatibility soft-excursion observer at a delivery. */
static lc_status lc_learning_soft_observation_amplitude(
    const lc_compiled_graph *graph,
    const lc_mixed_node *descriptor,
    const lc_expr_node *program_nodes,
    uint32_t node,
    const lc_real_t *node_state,
    const lc_real_t *parameters,
    lc_real_t before,
    lc_real_t after,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_real_t *amplitude
) {
    uint32_t program_index;
    uint32_t parameter_offset;
    const lc_learning_program *program;
    lc_real_t width;
    lc_real_t lower;
    lc_real_t derivative;
    uint32_t mode;
    lc_status status;
    lc_learning_readout_trajectory trajectory;
    if (graph == NULL || descriptor == NULL || program_nodes == NULL ||
        graph->learning_observer_program_by_node == NULL ||
        node >= graph->node_count || node_state == NULL ||
        !lc_isfinite(before) || !lc_isfinite(after) || variables == NULL ||
        workspace == NULL || amplitude == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *amplitude = LC_REAL_C(0.0);
    program_index = graph->learning_observer_program_by_node[node];
    if (program_index == UINT32_MAX || after <= before ||
        after >= descriptor->threshold) {
        return LC_OK;
    }
    status = lc_learning_readout_trajectory_prepare(
        descriptor, program_nodes, parameters, node_state, variables, workspace,
        workspace_count, &trajectory
    );
    if (status != LC_OK) {
        return status;
    }
    derivative = LC_REAL_C(0.0);
    for (mode = 0U; mode < trajectory.mode_count; ++mode) {
        derivative += trajectory.coefficients[mode] * trajectory.rates[mode];
    }
    if (derivative > LC_REAL_C(0.0)) {
        return LC_OK;
    }
    program = &graph->learning_programs[program_index];
    parameter_offset = graph->learning_observer_parameter_offset_by_node[node];
    width = graph->learning_parameters[
        parameter_offset + program->observer.band_width_parameter
    ];
    lower = descriptor->threshold - width;
    if (after <= lower) {
        return LC_OK;
    }
    *amplitude = (after - lower) / width;
    return lc_isfinite(*amplitude) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Apply one third factor to every equation-derived slot in its scope. */
static lc_status lc_learning_modulate(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t modulator,
    lc_real_t value,
    lc_time_t t,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_real_t *weight_deltas,
    uint8_t *weight_touched,
    uint32_t *weight_masters,
    uint32_t *weight_master_count
) {
    uint64_t batch_position;
    uint32_t event_kind = value >= LC_REAL_C(0.0)
        ? LC_LEARNING_MODULATION_POSITIVE
        : LC_LEARNING_MODULATION_NEGATIVE;
    if (graph == NULL || states == NULL || modulator >= graph->modulator_count ||
        !lc_isfinite(value)) {
        return LC_INVALID_ARGUMENT;
    }
    if (graph->modulation_learning_batch_offsets == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    for (batch_position = graph->modulation_learning_batch_offsets[modulator];
         batch_position < graph->modulation_learning_batch_offsets[modulator + 1U];
         ++batch_position) {
        const lc_learning_batch *batch =
            &graph->modulation_learning_batches[batch_position];
        const lc_learning_program *program;
        uint64_t position;
        if (batch->program >= graph->learning_program_count) {
            return LC_INVALID_ARGUMENT;
        }
        program = &graph->learning_programs[batch->program];
        for (position = batch->slot_offset;
             position < batch->slot_offset + batch->slot_count; ++position) {
            lc_status status = lc_learning_execute_event(
                graph, states, graph->modulator_slots[position], program,
                event_kind, value, LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(0.0), LC_REAL_C(0.0), t,
                variables, workspace, workspace_count,
                weight_deltas, weight_touched, weight_masters,
                weight_master_count
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    return LC_OK;
}

/* Commit one simultaneous equation-derived modulation reduction per weight. */
static lc_status lc_learning_commit_modulation_weights(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    lc_real_t *weight_deltas,
    uint8_t *weight_touched,
    uint32_t *weight_masters,
    uint32_t *weight_master_count
) {
    uint32_t position;
    if (graph == NULL || states == NULL || weight_deltas == NULL ||
        weight_touched == NULL || weight_masters == NULL ||
        weight_master_count == NULL ||
        *weight_master_count > graph->plastic_edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (position = 0U; position < *weight_master_count; ++position) {
        uint32_t master = weight_masters[position];
        uint32_t edge;
        const lc_learning_binding *binding;
        const lc_learning_program *program;
        lc_real_t normalized;
        lc_status status;
        if (master >= graph->plastic_edge_count || weight_touched[master] == 0U) {
            return LC_INVALID_ARGUMENT;
        }
        edge = graph->plastic_edges[master];
        binding = &graph->learning_bindings[edge];
        if (binding->program >= graph->learning_program_count) {
            return LC_INVALID_ARGUMENT;
        }
        program = &graph->learning_programs[binding->program];
        normalized = (
            lc_plastic_slot_weight(graph, states, master) - binding->weight_min
        ) / (binding->weight_max - binding->weight_min);
        normalized += weight_deltas[master];
        status = lc_learning_set_normalized_weight(
            graph, program, binding, states, master, normalized
        );
        weight_deltas[master] = LC_REAL_C(0.0);
        weight_touched[master] = 0U;
        if (status != LC_OK) {
            *weight_master_count = 0U;
            return status;
        }
    }
    *weight_master_count = 0U;
    return LC_OK;
}

/* Decay the fast pair traces used by legacy plasticity rules. */
static lc_status lc_plastic_advance_fast(
    const lc_plasticity_rule *rule,
    lc_plasticity_state *state,
    lc_time_t t
) {
    lc_status status = lc_plastic_decay(
        &state->pre_fast, &state->t_pre_fast, t, rule->tau_pre
    );
    if (status != LC_OK) {
        return status;
    }
    return lc_plastic_decay(
        &state->post_fast, &state->t_post_fast, t, rule->tau_post
    );
}

/* Update legacy plasticity state for a presynaptic delivery. */
static lc_status lc_plastic_pre(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t edge,
    lc_time_t t,
    lc_real_t post_readout,
    lc_real_t input_accepted,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint32_t slot;
    const lc_plasticity_rule *rule;
    lc_plasticity_state *state;
    lc_status status;
    lc_real_t normalized;
    if (graph != NULL && graph->learning_program_count > 0U) {
        return lc_learning_pre(
            graph, states, edge, t, post_readout, input_accepted,
            variables, workspace, workspace_count
        );
    }
    if (graph == NULL || states == NULL || edge >= graph->edge_count) {
        return LC_OK;
    }
    slot = graph->plastic_slot_by_edge[edge];
    if (slot == UINT32_MAX) {
        return LC_OK;
    }
    rule = &graph->plasticity[edge];
    state = &states[slot];
    status = lc_plastic_advance_fast(rule, state, t);
    if (status != LC_OK) {
        return status;
    }
    normalized = lc_plastic_normalized_weight(
        rule, lc_plastic_slot_weight(graph, states, slot)
    );
    if (rule->kind == LC_PLASTICITY_PAIR) {
        normalized -= rule->learning_rate * lc_plastic_learning_scale(graph, slot) *
            rule->a2_minus *
            normalized * state->post_fast;
        status = lc_plastic_set_normalized_weight(
            graph, rule, states, slot, normalized
        );
    } else if (rule->kind == LC_PLASTICITY_TRIPLET) {
        status = lc_plastic_decay(
            &state->pre_slow, &state->t_pre_slow, t, rule->tau_pre_slow
        );
        if (status == LC_OK) {
            status = lc_plastic_decay(
                &state->post_slow, &state->t_post_slow, t, rule->tau_post_slow
            );
        }
        if (status == LC_OK) {
            normalized -= rule->learning_rate * lc_plastic_learning_scale(graph, slot) *
                state->post_fast *
                (rule->a2_minus + rule->a3_minus * state->pre_slow);
            status = lc_plastic_set_normalized_weight(
                graph, rule, states, slot, normalized
            );
            state->pre_slow += LC_REAL_C(1.0);
        }
    } else if (rule->kind == LC_PLASTICITY_MODULATED) {
        status = lc_plastic_decay(
            &state->eligibility_minus, &state->t_eligibility_minus, t,
            rule->tau_eligibility_minus
        );
        if (status == LC_OK) {
            state->eligibility_minus += state->post_fast;
        }
    }
    if (status != LC_OK) {
        return status;
    }
    state->pre_fast += LC_REAL_C(1.0);
    return lc_isfinite(state->pre_fast) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Apply one postsynaptic legacy update to an edge or shared slot. */
static lc_status lc_plastic_post_slot(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t slot,
    lc_time_t t
) {
    uint32_t edge = graph->plastic_edges[slot];
    const lc_plasticity_rule *rule = &graph->plasticity[edge];
    lc_plasticity_state *state = &states[slot];
    lc_status status = lc_plastic_advance_fast(rule, state, t);
    lc_real_t normalized;
    if (status != LC_OK) {
        return status;
    }
    normalized = lc_plastic_normalized_weight(
        rule, lc_plastic_slot_weight(graph, states, slot)
    );
    if (rule->kind == LC_PLASTICITY_PAIR) {
        normalized += rule->learning_rate * lc_plastic_learning_scale(graph, slot) *
            rule->a2_plus *
            (LC_REAL_C(1.0) - normalized) * state->pre_fast;
        status = lc_plastic_set_normalized_weight(
            graph, rule, states, slot, normalized
        );
    } else if (rule->kind == LC_PLASTICITY_TRIPLET) {
        status = lc_plastic_decay(
            &state->pre_slow, &state->t_pre_slow, t, rule->tau_pre_slow
        );
        if (status == LC_OK) {
            status = lc_plastic_decay(
                &state->post_slow, &state->t_post_slow, t, rule->tau_post_slow
            );
        }
        if (status == LC_OK) {
            normalized += rule->learning_rate * lc_plastic_learning_scale(graph, slot) *
                state->pre_fast *
                (rule->a2_plus + rule->a3_plus * state->post_slow);
            status = lc_plastic_set_normalized_weight(
                graph, rule, states, slot, normalized
            );
            state->post_slow += LC_REAL_C(1.0);
        }
    } else if (rule->kind == LC_PLASTICITY_MODULATED) {
        status = lc_plastic_decay(
            &state->eligibility_plus, &state->t_eligibility_plus, t,
            rule->tau_eligibility_plus
        );
        if (status == LC_OK) {
            state->eligibility_plus += state->pre_fast;
        }
    }
    if (status != LC_OK) {
        return status;
    }
    state->post_fast += LC_REAL_C(1.0);
    return lc_isfinite(state->post_fast) ? LC_OK : LC_NUMERIC_ERROR;
}

/* Visit all incoming legacy plasticity slots after a neuron spikes. */
static lc_status lc_plastic_post(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t node,
    lc_time_t t,
    lc_real_t post_readout,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    uint64_t position;
    if (graph != NULL && graph->learning_program_count > 0U) {
        return lc_learning_post(
            graph, states, node, t, post_readout,
            variables, workspace, workspace_count
        );
    }
    if (graph == NULL || states == NULL || graph->plastic_edge_count == 0U) {
        return LC_OK;
    }
    for (position = graph->incoming_plastic_offsets[node];
         position < graph->incoming_plastic_offsets[node + 1U]; ++position) {
        lc_status status = lc_plastic_post_slot(
            graph, states, graph->incoming_plastic_slots[position], t
        );
        if (status != LC_OK) {
            return status;
        }
    }
    return LC_OK;
}

/* Apply reward to matching legacy eligibility traces. */
static lc_status lc_plastic_modulate(
    const lc_compiled_graph *graph,
    lc_plasticity_state *states,
    uint32_t modulator,
    lc_real_t value,
    lc_time_t t,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_real_t *weight_deltas,
    uint8_t *weight_touched,
    uint32_t *weight_masters,
    uint32_t *weight_master_count
) {
    uint64_t position;
    if (graph != NULL && graph->learning_program_count > 0U) {
        return lc_learning_modulate(
            graph, states, modulator, value, t,
            variables, workspace, workspace_count,
            weight_deltas, weight_touched, weight_masters,
            weight_master_count
        );
    }
    if (graph == NULL || states == NULL || modulator >= graph->modulator_count ||
        !lc_isfinite(value)) {
        return LC_INVALID_ARGUMENT;
    }
    for (position = graph->modulator_offsets[modulator];
         position < graph->modulator_offsets[modulator + 1U]; ++position) {
        uint32_t slot = graph->modulator_slots[position];
        uint32_t edge = graph->plastic_edges[slot];
        const lc_plasticity_rule *rule = &graph->plasticity[edge];
        lc_plasticity_state *state = &states[slot];
        lc_real_t plus = value >= LC_REAL_C(0.0) ? rule->positive_plus : rule->negative_plus;
        lc_real_t minus = value >= LC_REAL_C(0.0) ? rule->positive_minus : rule->negative_minus;
        lc_real_t normalized;
        lc_status status = lc_plastic_decay(
            &state->eligibility_plus, &state->t_eligibility_plus, t,
            rule->tau_eligibility_plus
        );
        if (status == LC_OK) {
            status = lc_plastic_decay(
                &state->eligibility_minus, &state->t_eligibility_minus, t,
                rule->tau_eligibility_minus
            );
        }
        if (status != LC_OK) {
            return status;
        }
        normalized = lc_plastic_normalized_weight(
            rule, lc_plastic_slot_weight(graph, states, slot)
        );
        normalized += rule->learning_rate * lc_plastic_learning_scale(graph, slot) *
            lc_real_fabs(value) *
            (plus * state->eligibility_plus + minus * state->eligibility_minus);
        status = lc_plastic_set_normalized_weight(
            graph, rule, states, slot, normalized
        );
        if (status != LC_OK) {
            return status;
        }
        if (rule->consume_on_modulation != 0U) {
            state->eligibility_plus = LC_REAL_C(0.0);
            state->eligibility_minus = LC_REAL_C(0.0);
        }
    }
    return LC_OK;
}

/* Resolve the current edge weight from static or learned storage. */
static lc_real_t lc_mixed_edge_weight(
    const lc_compiled_graph *graph,
    const lc_plasticity_state *states,
    uint32_t edge
) {
    uint32_t slot;
    if (graph == NULL || edge >= graph->edge_count) {
        return NAN;
    }
    slot = graph->plastic_slot_by_edge[edge];
    return slot == UINT32_MAX
        ? graph->edges[edge].weight
        : lc_plastic_slot_weight(graph, states, slot);
}

/* Accumulate a direct state deposit with presynaptic polarity. */
static inline lc_status lc_mixed_accumulate_static_add(
    const lc_compiled_graph *graph,
    const lc_mixed_node *active_nodes,
    uint32_t edge,
    lc_real_t *state_deposits,
    uint8_t *affected,
    uint8_t *deposit_affected,
    uint32_t *affected_nodes,
    uint32_t *affected_count
) {
    const lc_mixed_edge *delivery = &graph->edges[edge];
    const lc_mixed_node *descriptor = &active_nodes[delivery->post];
    uint32_t state_index = descriptor->state_offset + delivery->target;
    lc_real_t value = lc_signed_edge_weight(
        active_nodes[delivery->pre].polarity,
        delivery->weight * delivery->deposit_scale
    );
    state_deposits[state_index] += value;
    if (!lc_isfinite(state_deposits[state_index])) {
        return LC_NUMERIC_ERROR;
    }
    lc_sparse_node_add(
        affected, affected_nodes, affected_count, delivery->post
    );
    deposit_affected[delivery->post] = 1U;
    return LC_OK;
}

/* A delivery applies its deposit before testing for a reactive crossing. */
static lc_status lc_mixed_process_delivery(
    const lc_compiled_graph *graph,
    lc_plasticity_state *plasticity,
    lc_learning_observer_state *learning_observers,
    const lc_mixed_node *active_nodes,
    lc_real_t *state,
    lc_time_t *t_last,
    const lc_real_t *active_parameters,
    lc_node_runtime *runtime,
    uint32_t edge,
    lc_time_t t,
    uint64_t generation,
    lc_trace_config *trace,
    lc_network_error *error,
    lc_real_t *state_deposits,
    lc_real_t *program_deposits,
    uint8_t *affected,
    uint8_t *deposit_affected,
    uint32_t *affected_nodes,
    uint32_t *affected_count,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    const lc_mixed_edge *delivery;
    const lc_mixed_node *descriptor;
    uint32_t target_node;
    uint32_t deposit_target;
    lc_real_t magnitude;
    lc_real_t value;
    lc_real_t post_readout;
    lc_status status;
    if (graph == NULL || active_nodes == NULL || state == NULL ||
        t_last == NULL || runtime == NULL || edge >= graph->edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    delivery = &graph->edges[edge];
    target_node = delivery->post;
    descriptor = &active_nodes[target_node];
    status = lc_mixed_advance_node_planned(
        active_nodes, state, t_last, active_parameters, runtime,
        target_node, t, variables, workspace, workspace_count,
        graph->node_eval_plans, graph, learning_observers
    );
    if (status != LC_OK) {
        return status;
    }
    post_readout = state[descriptor->state_offset + descriptor->readout];
    magnitude = lc_mixed_edge_weight(graph, plasticity, edge);
    value = lc_signed_edge_weight(
        active_nodes[delivery->pre].polarity,
        magnitude * delivery->deposit_scale
    );
    /* Match the deposit policy below: only the clamped readout is discarded.
     * Other state (including retained synaptic current) can survive the clamp.
     * A zero current weight is still an accepted input, since it can learn. */
    deposit_target = delivery->deposit_kind == LC_DEPOSIT_PROGRAM
        ? descriptor->deposit_target : delivery->target;
    status = lc_plastic_pre(
        graph, plasticity, edge, t, post_readout,
        runtime[target_node].clamped && deposit_target == descriptor->readout
            ? LC_REAL_C(0.0) : LC_REAL_C(1.0),
        variables, workspace, workspace_count
    );
    if (status != LC_OK) {
        return status;
    }
    status = lc_trace_emit(
        trace, LC_TRACE_DELIVERY, LC_TRACE_PHASE_DEPOSIT, t, target_node,
        edge, generation, value, NULL, NULL, 0U, error
    );
    if (status != LC_OK) {
        return status;
    }
    if (delivery->deposit_kind == LC_DEPOSIT_PROGRAM) {
        program_deposits[target_node] += value;
        if (!lc_isfinite(program_deposits[target_node])) {
            return LC_NUMERIC_ERROR;
        }
    } else {
        uint32_t state_index = descriptor->state_offset + delivery->target;
        state_deposits[state_index] += value;
        if (!lc_isfinite(state_deposits[state_index])) {
            return LC_NUMERIC_ERROR;
        }
    }
    lc_sparse_node_add(
        affected, affected_nodes, affected_count, target_node
    );
    deposit_affected[target_node] = 1U;
    return LC_OK;
}

/* Commit a mixed same-time firing set, reset state, and fan out events. */
static lc_status lc_mixed_fire_nodes(
    const lc_mixed_node *nodes,
    lc_real_t *state,
    lc_time_t *t_last,
    const lc_real_t *parameters,
    lc_node_runtime *runtime,
    uint32_t node_count,
    const lc_mixed_edge *edges,
    const uint64_t *outgoing_offsets,
    const uint32_t *outgoing_edges,
    const lc_run_config *config,
    lc_heap *heap,
    const uint8_t *fired,
    const uint32_t *fired_nodes,
    uint32_t fired_count,
    lc_time_t t,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_decoder_run *decoders,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_trace_config *trace,
    lc_real_t *variables,
    lc_real_t *workspace,
    uint32_t workspace_count,
    const lc_compiled_graph *compiled_graph,
    lc_plasticity_state *plasticity_state,
    lc_learning_observer_state *learning_observers
) {
    uint32_t node;
    uint32_t fired_cursor;
    int direct_compiled_deliveries =
        compiled_graph != NULL &&
        compiled_graph->delivery_group_count == compiled_graph->edge_count;
    lc_output_sink sink = {
        outputs, config->output_capacity, output_count, decoders
    };
    for (fired_cursor = 0U;
         fired_cursor < (fired_nodes == NULL ? node_count : fired_count);
         ++fired_cursor) {
        const lc_mixed_node *descriptor;
        const lc_expr_node *reset_nodes;
        const lc_real_t *node_state;
        lc_real_t before_reset[LC_ANALYTICAL_MAX_STATES];
        int capture_reset;
        uint64_t position;
        lc_status status;
        node = fired_nodes == NULL ? fired_cursor : fired_nodes[fired_cursor];
        if (!fired[node]) {
            continue;
        }
        descriptor = &nodes[node];
        reset_nodes =
            compiled_graph != NULL && compiled_graph->node_eval_plans != NULL
                ? compiled_graph->node_eval_plans[node].reset_nodes
                : descriptor->program_nodes;
        node_state = &state[descriptor->state_offset];
        if (compiled_graph != NULL && learning_observers != NULL &&
            compiled_graph->learning_observer_program_by_node != NULL &&
            compiled_graph->learning_observer_program_by_node[node] !=
                UINT32_MAX) {
            status = lc_learning_observe(
                compiled_graph, plasticity_state, learning_observers, node, t,
                node_state[descriptor->readout], LC_REAL_C(1.0), variables, workspace,
                workspace_count
            );
            if (status != LC_OK) {
                return status;
            }
        }
        status = lc_plastic_post(
            compiled_graph, plasticity_state, node, t,
            node_state[descriptor->readout],
            variables, workspace, workspace_count
        );
        if (status != LC_OK) {
            return status;
        }
        status = lc_emit_output_spike(t, node, &sink, stats, error);
        if (status != LC_OK) {
            return status;
        }
        status = lc_trace_emit(
            trace, LC_TRACE_SPIKE, LC_TRACE_PHASE_FIRE, t, node, UINT32_MAX,
            runtime[node].generation, LC_REAL_C(0.0), node_state, node_state,
            descriptor->state_count, error
        );
        if (status != LC_OK) {
            return status;
        }
        capture_reset = lc_trace_wants_state(trace, LC_TRACE_RESET, node);
        if (capture_reset) {
            lc_trace_copy_state(before_reset, node_state, descriptor->state_count);
        }
        if (lc_mixed_scalar_affine_enabled(descriptor, &runtime[node])) {
            state[descriptor->state_offset] = descriptor->scalar_affine_reset;
            status = LC_OK;
        } else {
            status = lc_expr_state_map(
                reset_nodes, descriptor->program_node_count,
                descriptor->parameter_count > 0U
                    ? &parameters[descriptor->parameter_offset]
                    : NULL,
                descriptor->parameter_count, descriptor->reset_roots,
                descriptor->state_count, &state[descriptor->state_offset],
                variables, descriptor->state_count + 1U, workspace,
                workspace_count
            );
        }
        if (status != LC_OK) {
            return status;
        }
        status = lc_trace_emit(
            trace, LC_TRACE_RESET, LC_TRACE_PHASE_FIRE, t, node, UINT32_MAX,
            runtime[node].generation, LC_REAL_C(0.0),
            capture_reset ? before_reset : NULL,
            capture_reset ? node_state : NULL, descriptor->state_count, error
        );
        if (status != LC_OK) {
            return status;
        }
        t_last[node] = t;
        if (runtime[node].generation == UINT64_MAX) {
            return LC_NUMERIC_ERROR;
        }
        runtime[node].generation++;
        if (descriptor->crossing_kind == LC_CROSSING_INTEGRATED_HAZARD) {
            status = lc_hazard_draw(
                &runtime[node], config->stochastic_seed, node
            );
            if (status != LC_OK) {
                return status;
            }
        }

        if (compiled_graph != NULL) {
            if (direct_compiled_deliveries) {
                for (position = outgoing_offsets[node];
                     position < outgoing_offsets[node + 1U];
                     ++position) {
                    uint32_t edge = outgoing_edges[position];
                    lc_time_t delivery_time = t + edges[edge].delay;
                    if (!lc_isfinite(delivery_time) || delivery_time < t ||
                        (edges[edge].delay > LC_REAL_C(0.0) && delivery_time == t)) {
                        return LC_NUMERIC_ERROR;
                    }
                    status = lc_push_delivery(
                        heap, edge, edges[edge].post,
                        delivery_time, config, error
                    );
                    if (status != LC_OK) {
                        return status;
                    }
                    if (delivery_time <= config->t_end) {
                        stats->deliveries_scheduled++;
                    }
                }
            } else {
                for (position = compiled_graph->outgoing_offsets[node];
                     position < compiled_graph->outgoing_offsets[node + 1U];
                     ++position) {
                    uint32_t group_index = (uint32_t)position;
                    const lc_delivery_group *group =
                        &compiled_graph->delivery_groups[group_index];
                    lc_time_t delivery_time = t + group->delay;
                    if (!lc_isfinite(delivery_time) || delivery_time < t ||
                        (group->delay > LC_REAL_C(0.0) && delivery_time == t)) {
                        return LC_NUMERIC_ERROR;
                    }
                    if (group->edge_count == 1U) {
                        uint32_t edge = group->first_edge;
                        status = lc_push_delivery(
                            heap, edge, compiled_graph->edges[edge].post,
                            delivery_time, config, error
                        );
                    } else {
                        status = lc_push_delivery_group(
                            heap, compiled_graph, group_index, delivery_time,
                            config, error
                        );
                    }
                    if (status != LC_OK) {
                        return status;
                    }
                    if (delivery_time <= config->t_end) {
                        stats->deliveries_scheduled += group->edge_count;
                    }
                }
            }
        } else {
            for (position = outgoing_offsets[node];
                 position < outgoing_offsets[node + 1U]; ++position) {
                uint32_t edge = outgoing_edges[position];
                lc_time_t delivery_time = t + edges[edge].delay;
                if (!lc_isfinite(delivery_time) || delivery_time < t ||
                    (edges[edge].delay > LC_REAL_C(0.0) && delivery_time == t)) {
                    return LC_NUMERIC_ERROR;
                }
                status = lc_push_delivery(
                    heap, edge, edges[edge].post, delivery_time, config, error
                );
                if (status != LC_OK) {
                    return status;
                }
                if (delivery_time <= config->t_end) {
                    stats->deliveries_scheduled++;
                }
            }
        }

        if (descriptor->refractory > LC_REAL_C(0.0)) {
            lc_event release;
            lc_time_t release_time = t + descriptor->refractory;
            if (!lc_isfinite(release_time) || release_time <= t) {
                return LC_NUMERIC_ERROR;
            }
            runtime[node].clamped = 1;
            if (runtime[node].refractory_generation == UINT64_MAX) {
                return LC_NUMERIC_ERROR;
            }
            runtime[node].refractory_generation++;
            status = lc_trace_emit(
                trace, LC_TRACE_REFRACTORY_ENTER, LC_TRACE_PHASE_FIRE, t,
                node, UINT32_MAX, runtime[node].refractory_generation,
                release_time, node_state, node_state, descriptor->state_count,
                error
            );
            if (status != LC_OK) {
                return status;
            }
            if (release_time <= config->t_end) {
                memset(&release, 0, sizeof(release));
                release.t = release_time;
                release.phase = LC_PHASE_BOUNDARY;
                release.kind = LC_EVENT_REFRACTORY_RELEASE;
                release.index = node;
                release.generation = runtime[node].refractory_generation;
                status = lc_heap_push_report(heap, release, error);
                if (status != LC_OK) {
                    return status;
                }
            }
        } else {
            status = lc_mixed_schedule_prediction(
                nodes, state, t_last, parameters, runtime, node, config, heap,
                error, variables, workspace, workspace_count
            );
            if (status != LC_OK) {
                return status;
            }
        }
    }
    return LC_OK;
}

static int lc_mixed_state_program_valid(const lc_mixed_node *descriptor) {
    uint32_t index;
    if (descriptor == NULL || descriptor->polarity > LC_MIXED ||
        descriptor->arithmetic_kind > LC_NODE_ARITHMETIC_SCALAR_AFFINE ||
        descriptor->reset_before_deposit > 1U ||
        (descriptor->reset_before_deposit != 0U &&
         descriptor->crossing_kind != LC_CROSSING_REACTIVE) ||
        descriptor->state_count == 0U ||
        descriptor->state_count > LC_ANALYTICAL_MAX_STATES ||
        descriptor->readout >= descriptor->state_count ||
        descriptor->program_nodes == NULL || descriptor->program_node_count == 0U) {
        return 0;
    }
    if (descriptor->arithmetic_kind == LC_NODE_ARITHMETIC_SCALAR_AFFINE &&
        (descriptor->state_count != 1U || descriptor->readout != 0U ||
         (descriptor->crossing_kind != LC_CROSSING_SCALAR_LOG &&
          descriptor->crossing_kind != LC_CROSSING_INTEGRATED_HAZARD) ||
         !lc_isfinite(descriptor->scalar_affine_decay) ||
         descriptor->scalar_affine_decay >= LC_REAL_C(0.0) ||
         !lc_isfinite(descriptor->scalar_affine_drive) ||
         !lc_isfinite(descriptor->scalar_affine_reset) ||
         descriptor->scalar_affine_reset >= descriptor->threshold)) {
        return 0;
    }
    for (index = 0; index < descriptor->state_count; ++index) {
        if (descriptor->normal_roots[index] >= descriptor->program_node_count ||
            descriptor->clamped_roots[index] >= descriptor->program_node_count ||
            descriptor->reset_roots[index] >= descriptor->program_node_count) {
            return 0;
        }
    }
    if (descriptor->deposit_node_count > 0U &&
        (descriptor->deposit_nodes == NULL ||
         descriptor->deposit_root >= descriptor->deposit_node_count ||
         descriptor->deposit_target >= descriptor->state_count)) {
        return 0;
    }
    return 1;
}

static int lc_mixed_crossing_valid(const lc_mixed_node *descriptor) {
    uint32_t count = descriptor->program_node_count;
    if (descriptor->crossing_kind == LC_CROSSING_REACTIVE) {
        return descriptor->dispatch == LC_REACTIVE;
    }
    if (descriptor->crossing_kind == LC_CROSSING_INTEGRATED_HAZARD) {
        const lc_hazard_config *hazard = &descriptor->hazard;
        uint32_t index;
        if (descriptor->dispatch != LC_ROOT_FIND ||
            hazard->trajectory_mode_count > LC_ANALYTICAL_MAX_STATES ||
            (hazard->trajectory_mode_count > 0U &&
             hazard->trajectory_limit_root >= count)) {
            return 0;
        }
        for (index = 0U; index < hazard->trajectory_mode_count; ++index) {
            if (hazard->trajectory_coefficient_roots[index] >= count ||
                hazard->trajectory_rate_roots[index] >= count) {
                return 0;
            }
        }
        return
            hazard->kind == LC_HAZARD_EXPONENTIAL_VOLTAGE &&
            lc_isfinite(hazard->log_scale) &&
            lc_isfinite(hazard->voltage_gain) && hazard->voltage_gain > LC_REAL_C(0.0) &&
            lc_isfinite(hazard->relative_tolerance) &&
            hazard->relative_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(hazard->absolute_tolerance) &&
            hazard->absolute_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(hazard->time_tolerance) && hazard->time_tolerance > LC_REAL_C(0.0) &&
            hazard->maximum_quadrature_depth > 0U &&
            hazard->maximum_quadrature_depth <= 32U &&
            hazard->maximum_root_iterations > 0U &&
            hazard->maximum_root_iterations <= 4096U;
    }
    if (descriptor->crossing_kind == LC_CROSSING_SCALAR_LOG) {
        return (descriptor->dispatch == LC_REACTIVE ||
                descriptor->dispatch == LC_CLOSED_FORM) &&
            descriptor->scalar_log_hint.decay_root < count &&
            descriptor->scalar_log_hint.affine_root < count &&
            descriptor->scalar_log_hint.threshold_root < count;
    }
    if (descriptor->crossing_kind == LC_CROSSING_ALPHA_REAL) {
        const lc_root_hint *hint = &descriptor->root_hint;
        return descriptor->dispatch == LC_ROOT_FIND && hint->g_root < count &&
            hint->g_prime_root < count && hint->extremum_root < count &&
            hint->extremum_prime_root < count && hint->asymptote_root < count &&
            hint->membrane_coefficient_root < count &&
            hint->synapse_constant_root < count &&
            hint->synapse_linear_root < count && hint->membrane_rate_root < count &&
            hint->synapse_rate_root < count && hint->threshold_root < count &&
            lc_isfinite(hint->relative_tolerance) && hint->relative_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(hint->fastest_time_constant) && hint->fastest_time_constant > LC_REAL_C(0.0);
    }
    if (descriptor->crossing_kind == LC_CROSSING_TWO_REAL_EXP) {
        const lc_two_exp_hint *hint = &descriptor->two_exp_hint;
        return descriptor->dispatch == LC_ROOT_FIND && hint->g_root < count &&
            hint->g_prime_root < count && hint->limit_root < count &&
            hint->coefficient_one_root < count && hint->coefficient_two_root < count &&
            hint->rate_one_root < count && hint->rate_two_root < count &&
            lc_isfinite(hint->relative_tolerance) && hint->relative_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(hint->fastest_time_constant) && hint->fastest_time_constant > LC_REAL_C(0.0);
    }
    if (descriptor->crossing_kind == LC_CROSSING_MULTI_REAL_EXP) {
        const lc_multi_exp_hint *hint = &descriptor->multi_exp_hint;
        uint32_t index;
        if (descriptor->dispatch != LC_ROOT_FIND || hint->limit_root >= count ||
            hint->mode_count == 0U ||
            hint->mode_count > LC_ANALYTICAL_MAX_STATES ||
            hint->iteration_cap == 0U || hint->iteration_cap > 4096U ||
            !lc_isfinite(hint->relative_tolerance) ||
            hint->relative_tolerance <= LC_REAL_C(0.0) ||
            !lc_isfinite(hint->fastest_time_constant) ||
            hint->fastest_time_constant <= LC_REAL_C(0.0)) {
            return 0;
        }
        for (index = 0U; index < hint->mode_count; ++index) {
            if (hint->coefficient_roots[index] >= count ||
                hint->rate_roots[index] >= count) {
                return 0;
            }
        }
        return 1;
    }
    if (descriptor->crossing_kind == LC_CROSSING_REPEATED_REAL_MODE ||
        descriptor->crossing_kind == LC_CROSSING_MULTI_EXP_POLY) {
        const lc_exp_poly_hint *hint = &descriptor->exp_poly_hint;
        uint32_t block;
        uint32_t index;
        if (descriptor->dispatch != LC_ROOT_FIND || hint->limit_root >= count ||
            hint->block_count == 0U ||
            hint->block_count > LC_ANALYTICAL_MAX_STATES ||
            hint->coefficient_count == 0U ||
            hint->coefficient_count > LC_ANALYTICAL_MAX_STATES ||
            hint->coefficient_offsets[0] != 0U ||
            hint->coefficient_offsets[hint->block_count] !=
                hint->coefficient_count ||
            hint->iteration_cap == 0U || hint->iteration_cap > 4096U ||
            !lc_isfinite(hint->relative_tolerance) ||
            hint->relative_tolerance <= LC_REAL_C(0.0) ||
            !lc_isfinite(hint->fastest_time_constant) ||
            hint->fastest_time_constant <= LC_REAL_C(0.0)) {
            return 0;
        }
        for (block = 0U; block < hint->block_count; ++block) {
            if (hint->rate_roots[block] >= count ||
                hint->coefficient_offsets[block] >=
                    hint->coefficient_offsets[block + 1U] ||
                hint->coefficient_offsets[block + 1U] >
                    hint->coefficient_count) {
                return 0;
            }
        }
        for (index = 0U; index < hint->coefficient_count; ++index) {
            if (hint->coefficient_roots[index] >= count) {
                return 0;
            }
        }
        return 1;
    }
    if (descriptor->crossing_kind == LC_CROSSING_NUMERICAL) {
        const lc_step_config *config = &descriptor->step_config;
        return descriptor->dispatch == LC_STEPPED &&
            lc_isfinite(config->relative_tolerance) &&
            config->relative_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(config->absolute_tolerance) &&
            config->absolute_tolerance > LC_REAL_C(0.0) &&
            lc_isfinite(config->initial_step) && config->initial_step > LC_REAL_C(0.0) &&
            lc_isfinite(config->minimum_step) && config->minimum_step > LC_REAL_C(0.0) &&
            lc_isfinite(config->maximum_step) &&
            config->maximum_step >= config->initial_step &&
            config->initial_step >= config->minimum_step &&
            lc_isfinite(config->event_tolerance) && config->event_tolerance > LC_REAL_C(0.0) &&
            config->maximum_steps > 0U &&
            config->maximum_rhs_evaluations >= 7U;
    }
    return 0;
}

/* Validate a deposit against the target node's state layout. */
static int lc_mixed_deposit_valid(
    const lc_mixed_node *descriptor,
    uint32_t kind,
    uint32_t target
) {
    if (kind == LC_DEPOSIT_STATE_ADD) {
        return target < descriptor->state_count;
    }
    if (kind == LC_DEPOSIT_PROGRAM) {
        return descriptor->deposit_nodes != NULL && descriptor->deposit_node_count > 0U &&
            descriptor->deposit_root < descriptor->deposit_node_count &&
            descriptor->deposit_target < descriptor->state_count &&
            target == descriptor->deposit_target;
    }
    return 0;
}

/* Compile and execute a temporary mixed graph for compatibility callers. */
lc_status lc_mixed_network_run(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    lc_real_t *state,
    uint32_t state_count,
    lc_time_t *t_last,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    uint32_t edge_count,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_heap heap;
    lc_mixed_node *active_nodes = NULL;
    lc_real_t *active_parameters = NULL;
    lc_node_runtime *runtime = NULL;
    lc_real_t *state_deposits = NULL;
    lc_real_t *program_deposits = NULL;
    uint8_t *affected = NULL;
    uint8_t *deposit_affected = NULL;
    uint8_t *timestamp_reset = NULL;
    uint8_t *boundary_affected = NULL;
    uint8_t *fired = NULL;
    uint64_t *outgoing_offsets = NULL;
    uint64_t *outgoing_cursors = NULL;
    uint32_t *outgoing_edges = NULL;
    lc_real_t *workspace = NULL;
    lc_real_t variables[LC_LEARNING_MAX_TRACES + LC_LEARNING_BASE_VARIABLE_COUNT];
    uint32_t workspace_count = 1U;
    uint32_t expected_state_offset = 0U;
    uint32_t expected_parameter_offset = 0U;
    uint32_t node;
    uint32_t edge;
    uint32_t input;
    uint32_t drive_update;
    size_t csr_node;
    lc_status status = LC_OK;
    lc_profile_t kernel_start;

    memset(&heap, 0, sizeof(heap));
    if (nodes == NULL || node_count == 0U || state == NULL || state_count == 0U ||
        t_last == NULL || config == NULL || output_count == NULL || stats == NULL ||
        error == NULL || config->queue_capacity == 0U ||
        config->same_time_cascade_limit == 0U || !lc_isfinite(config->t_end) ||
        config->t_end < LC_REAL_C(0.0) || (parameter_count > 0U && parameters == NULL) ||
        (edge_count > 0U && edges == NULL) || (input_count > 0U && inputs == NULL) ||
        (drive_update_count > 0U && drive_updates == NULL) ||
        (config->output_capacity > 0U && outputs == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *output_count = 0U;
    memset(stats, 0, sizeof(*stats));
    kernel_start = lc_monotonic_seconds();
    memset(error, 0, sizeof(*error));

    for (node = 0; node < node_count; ++node) {
        const lc_mixed_node *descriptor = &nodes[node];
        uint32_t local;
        if (descriptor->state_offset != expected_state_offset ||
            descriptor->parameter_offset != expected_parameter_offset ||
            !lc_mixed_state_program_valid(descriptor) ||
            !lc_mixed_crossing_valid(descriptor) || !lc_isfinite(descriptor->threshold) ||
            !lc_isfinite(descriptor->refractory) || descriptor->refractory < LC_REAL_C(0.0) ||
            !lc_isfinite(t_last[node]) || t_last[node] < LC_REAL_C(0.0) ||
            t_last[node] > config->t_end) {
            return LC_INVALID_ARGUMENT;
        }
        if (descriptor->program_node_count > workspace_count) {
            workspace_count = descriptor->program_node_count;
        }
        if (descriptor->deposit_node_count > workspace_count) {
            workspace_count = descriptor->deposit_node_count;
        }
        if (descriptor->state_count > UINT32_MAX - expected_state_offset ||
            descriptor->parameter_count > UINT32_MAX - expected_parameter_offset) {
            return LC_INVALID_ARGUMENT;
        }
        expected_state_offset += descriptor->state_count;
        expected_parameter_offset += descriptor->parameter_count;
        for (local = 0; local < descriptor->state_count; ++local) {
            if (!lc_isfinite(state[descriptor->state_offset + local])) {
                return LC_INVALID_ARGUMENT;
            }
        }
        if (descriptor->crossing_kind != LC_CROSSING_REACTIVE &&
            descriptor->crossing_kind != LC_CROSSING_INTEGRATED_HAZARD &&
            state[descriptor->state_offset + descriptor->readout] >=
                descriptor->threshold) {
            return LC_INVALID_ARGUMENT;
        }
    }
    if (expected_state_offset != state_count || expected_parameter_offset != parameter_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0; node < parameter_count; ++node) {
        if (!lc_isfinite(parameters[node])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (edge = 0; edge < edge_count; ++edge) {
        const lc_mixed_node *post;
        if (edges[edge].pre >= node_count || edges[edge].post >= node_count ||
            !lc_edge_weight_valid(
                nodes[edges[edge].pre].polarity, edges[edge].weight
            ) ||
            !lc_isfinite(edges[edge].delay) || edges[edge].delay < LC_REAL_C(0.0) ||
            !lc_isfinite(edges[edge].deposit_scale) ||
            edges[edge].deposit_scale <= LC_REAL_C(0.0)) {
            return LC_INVALID_ARGUMENT;
        }
        post = &nodes[edges[edge].post];
        if (!lc_mixed_deposit_valid(post, edges[edge].deposit_kind, edges[edge].target)) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (input = 0; input < input_count; ++input) {
        const lc_mixed_node *target;
        if (inputs[input].node >= node_count || !lc_isfinite(inputs[input].t) ||
            !lc_isfinite(inputs[input].value) || inputs[input].t < t_last[inputs[input].node]) {
            return LC_INVALID_ARGUMENT;
        }
        target = &nodes[inputs[input].node];
        if (!lc_mixed_deposit_valid(
                target, inputs[input].deposit_kind, inputs[input].target
            )) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        const lc_mixed_node *target;
        if (drive_updates[drive_update].node >= node_count ||
            !lc_isfinite(drive_updates[drive_update].t) ||
            !lc_isfinite(drive_updates[drive_update].value) ||
            drive_updates[drive_update].t < t_last[drive_updates[drive_update].node]) {
            return LC_INVALID_ARGUMENT;
        }
        target = &nodes[drive_updates[drive_update].node];
        if (drive_updates[drive_update].binding >= target->parameter_count) {
            return LC_INVALID_ARGUMENT;
        }
    }

    if (lc_allocation_size_overflows(config->queue_capacity, sizeof(lc_event)) ||
        lc_allocation_size_overflows((uint64_t)node_count + 1U, sizeof(uint64_t)) ||
        lc_allocation_size_overflows(edge_count, sizeof(uint32_t)) ||
        lc_allocation_size_overflows(state_count, sizeof(lc_real_t)) ||
        lc_allocation_size_overflows(parameter_count, sizeof(lc_real_t)) ||
        lc_allocation_size_overflows(workspace_count, sizeof(lc_real_t))) {
        return LC_ALLOCATION_FAILED;
    }
    heap.items = calloc((size_t)config->queue_capacity, sizeof(lc_event));
    active_nodes = calloc(node_count, sizeof(lc_mixed_node));
    runtime = calloc(node_count, sizeof(lc_node_runtime));
    state_deposits = calloc(state_count, sizeof(lc_real_t));
    program_deposits = calloc(node_count, sizeof(lc_real_t));
    affected = calloc(node_count, sizeof(uint8_t));
    deposit_affected = calloc(node_count, sizeof(uint8_t));
    timestamp_reset = calloc(node_count, sizeof(uint8_t));
    boundary_affected = calloc(node_count, sizeof(uint8_t));
    fired = calloc(node_count, sizeof(uint8_t));
    outgoing_offsets = calloc((size_t)node_count + 1U, sizeof(uint64_t));
    outgoing_cursors = calloc(node_count, sizeof(uint64_t));
    workspace = calloc(workspace_count, sizeof(lc_real_t));
    if (parameter_count > 0U) {
        active_parameters = calloc(parameter_count, sizeof(lc_real_t));
    }
    if (edge_count > 0U) {
        outgoing_edges = calloc(edge_count, sizeof(uint32_t));
    }
    if (heap.items == NULL || active_nodes == NULL || runtime == NULL ||
        state_deposits == NULL || program_deposits == NULL || affected == NULL ||
        deposit_affected == NULL || timestamp_reset == NULL ||
        boundary_affected == NULL || fired == NULL || outgoing_offsets == NULL ||
        outgoing_cursors == NULL || workspace == NULL ||
        (parameter_count > 0U && active_parameters == NULL) ||
        (edge_count > 0U && outgoing_edges == NULL)) {
        status = LC_ALLOCATION_FAILED;
        goto mixed_cleanup;
    }
    heap.capacity = config->queue_capacity;
    memcpy(active_nodes, nodes, node_count * sizeof(lc_mixed_node));
    if (parameter_count > 0U) {
        memcpy(active_parameters, parameters, parameter_count * sizeof(lc_real_t));
    }

    for (edge = 0; edge < edge_count; ++edge) {
        outgoing_offsets[edges[edge].pre + 1U]++;
    }
    for (csr_node = 1; csr_node <= (size_t)node_count; ++csr_node) {
        outgoing_offsets[csr_node] += outgoing_offsets[csr_node - 1U];
    }
    memcpy(outgoing_cursors, outgoing_offsets, node_count * sizeof(uint64_t));
    for (edge = 0; edge < edge_count; ++edge) {
        uint32_t source = edges[edge].pre;
        outgoing_edges[outgoing_cursors[source]++] = edge;
    }

    for (node = 0; node < node_count; ++node) {
        runtime[node].generation = 1U;
        status = lc_mixed_schedule_prediction(
            active_nodes, state, t_last, active_parameters, runtime, node, config,
            &heap, error, variables, workspace, workspace_count
        );
        if (status != LC_OK) {
            goto mixed_cleanup;
        }
    }
    for (input = 0; input < input_count; ++input) {
        lc_event event;
        if (inputs[input].t > config->t_end) {
            continue;
        }
        memset(&event, 0, sizeof(event));
        event.t = inputs[input].t;
        event.phase = LC_PHASE_DEPOSIT;
        event.kind = LC_EVENT_INPUT_SPIKE;
        event.index = input;
        status = lc_heap_push_report(&heap, event, error);
        if (status != LC_OK) {
            goto mixed_cleanup;
        }
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        lc_event event;
        if (drive_updates[drive_update].t > config->t_end) {
            continue;
        }
        memset(&event, 0, sizeof(event));
        event.t = drive_updates[drive_update].t;
        event.phase = LC_PHASE_BOUNDARY;
        event.kind = LC_EVENT_DRIVE_UPDATE;
        event.index = drive_update;
        status = lc_heap_push_report(&heap, event, error);
        if (status != LC_OK) {
            goto mixed_cleanup;
        }
    }

    while (heap.size > 0U && heap.items[0].t <= config->t_end) {
        lc_time_t t = heap.items[0].t;
        uint32_t cascade_depth = 0U;
        memset(boundary_affected, 0, node_count * sizeof(uint8_t));
        memset(timestamp_reset, 0, node_count * sizeof(uint8_t));
        while (lc_at(&heap, t, LC_PHASE_BOUNDARY)) {
            lc_event event = lc_heap_pop(&heap);
            stats->events_popped++;
            if (event.kind == LC_EVENT_DRIVE_UPDATE) {
                const lc_mixed_drive_update *update = &drive_updates[event.index];
                const lc_mixed_node *descriptor = &active_nodes[update->node];
                stats->drive_updates_processed++;
                status = lc_mixed_advance_node(
                    active_nodes, state, t_last, active_parameters, runtime,
                    update->node, t, variables, workspace, workspace_count
                );
                if (status != LC_OK) {
                    goto mixed_cleanup;
                }
                active_parameters[descriptor->parameter_offset + update->binding] =
                    update->value;
                runtime[update->node].affine_disabled = 1;
                boundary_affected[update->node] = 1U;
            } else if (event.kind == LC_EVENT_REFRACTORY_RELEASE &&
                event.generation == runtime[event.index].refractory_generation &&
                runtime[event.index].clamped) {
                stats->refractory_releases_processed++;
                status = lc_mixed_advance_node(
                    active_nodes, state, t_last, active_parameters, runtime,
                    event.index, t, variables, workspace, workspace_count
                );
                if (status != LC_OK) {
                    goto mixed_cleanup;
                }
                runtime[event.index].clamped = 0;
                boundary_affected[event.index] = 1U;
            }
        }

        for (;;) {
            int any_fired = 0;
            memset(state_deposits, 0, state_count * sizeof(lc_real_t));
            memset(program_deposits, 0, node_count * sizeof(lc_real_t));
            memset(affected, 0, node_count * sizeof(uint8_t));
            memset(deposit_affected, 0, node_count * sizeof(uint8_t));
            memset(fired, 0, node_count * sizeof(uint8_t));
            for (node = 0; node < node_count; ++node) {
                if (boundary_affected[node]) {
                    affected[node] = 1U;
                    boundary_affected[node] = 0U;
                }
            }
            while (lc_at(&heap, t, LC_PHASE_DEPOSIT)) {
                lc_event event = lc_heap_pop(&heap);
                uint32_t target_node;
                uint32_t deposit_kind;
                uint32_t target;
                lc_real_t value;
                const lc_mixed_node *descriptor;
                stats->events_popped++;
                if (event.kind == LC_EVENT_DELIVERY) {
                    const lc_mixed_edge *delivery = &edges[event.index];
                    stats->deliveries_processed++;
                    target_node = delivery->post;
                    deposit_kind = delivery->deposit_kind;
                    target = delivery->target;
                    value = lc_signed_edge_weight(
                        active_nodes[delivery->pre].polarity,
                        delivery->weight * delivery->deposit_scale
                    );
                } else {
                    const lc_mixed_input_spike *injected = &inputs[event.index];
                    stats->input_spikes_processed++;
                    target_node = injected->node;
                    deposit_kind = injected->deposit_kind;
                    target = injected->target;
                    value = injected->value;
                }
                descriptor = &active_nodes[target_node];
                if (deposit_kind == LC_DEPOSIT_PROGRAM) {
                    program_deposits[target_node] += value;
                    if (!lc_isfinite(program_deposits[target_node])) {
                        status = LC_NUMERIC_ERROR;
                        goto mixed_cleanup;
                    }
                } else {
                    uint32_t state_index = descriptor->state_offset + target;
                    state_deposits[state_index] += value;
                    if (!lc_isfinite(state_deposits[state_index])) {
                        status = LC_NUMERIC_ERROR;
                        goto mixed_cleanup;
                    }
                }
                affected[target_node] = 1U;
                deposit_affected[target_node] = 1U;
            }

            for (node = 0; node < node_count; ++node) {
                const lc_mixed_node *descriptor;
                uint32_t local;
                if (!affected[node]) {
                    continue;
                }
                descriptor = &active_nodes[node];
                status = lc_mixed_advance_node(
                    active_nodes, state, t_last, active_parameters, runtime, node,
                    t, variables, workspace, workspace_count
                );
                if (status != LC_OK) {
                    goto mixed_cleanup;
                }
                if (deposit_affected[node] && descriptor->reset_before_deposit &&
                    !timestamp_reset[node]) {
                    status = lc_expr_state_map(
                        descriptor->program_nodes, descriptor->program_node_count,
                        descriptor->parameter_count > 0U
                            ? &active_parameters[descriptor->parameter_offset]
                            : NULL,
                        descriptor->parameter_count, descriptor->reset_roots,
                        descriptor->state_count, &state[descriptor->state_offset],
                        variables, descriptor->state_count + 1U, workspace,
                        workspace_count
                    );
                    if (status != LC_OK) {
                        goto mixed_cleanup;
                    }
                    t_last[node] = t;
                    timestamp_reset[node] = 1U;
                }
                for (local = 0; local < descriptor->state_count; ++local) {
                    uint32_t state_index = descriptor->state_offset + local;
                    if (runtime[node].clamped && local == descriptor->readout) {
                        continue;
                    }
                    state[state_index] += state_deposits[state_index];
                    if (!lc_isfinite(state[state_index])) {
                        status = LC_NUMERIC_ERROR;
                        goto mixed_cleanup;
                    }
                }
                if (program_deposits[node] != LC_REAL_C(0.0) &&
                    !(runtime[node].clamped &&
                      descriptor->deposit_target == descriptor->readout)) {
                    status = lc_expr_state_deposit(
                        descriptor->deposit_nodes, descriptor->deposit_node_count,
                        descriptor->parameter_count > 0U
                            ? &active_parameters[descriptor->parameter_offset]
                            : NULL,
                        descriptor->parameter_count, descriptor->deposit_root,
                        program_deposits[node], &state[descriptor->state_offset],
                        descriptor->state_count, descriptor->deposit_target,
                        variables, 1U, workspace, workspace_count
                    );
                    if (status != LC_OK) {
                        goto mixed_cleanup;
                    }
                }
                if (runtime[node].clamped) {
                    continue;
                }
                if (runtime[node].generation == UINT64_MAX) {
                    status = LC_NUMERIC_ERROR;
                    goto mixed_cleanup;
                }
                runtime[node].generation++;
                if (descriptor->crossing_kind !=
                        LC_CROSSING_INTEGRATED_HAZARD &&
                    state[descriptor->state_offset + descriptor->readout] >=
                        descriptor->threshold) {
                    fired[node] = 1U;
                } else {
                    status = lc_mixed_schedule_prediction(
                        active_nodes, state, t_last, active_parameters, runtime,
                        node, config, &heap, error, variables, workspace,
                        workspace_count
                    );
                    if (status != LC_OK) {
                        goto mixed_cleanup;
                    }
                }
            }

            while (lc_at(&heap, t, LC_PHASE_PREDICTION)) {
                lc_event event = lc_heap_pop(&heap);
                stats->events_popped++;
                if (event.generation != runtime[event.index].generation ||
                    runtime[event.index].clamped) {
                    stats->stale_predictions++;
                    continue;
                }
                status = lc_mixed_advance_node(
                    active_nodes, state, t_last, active_parameters, runtime,
                    event.index, t, variables, workspace, workspace_count
                );
                if (status != LC_OK) {
                    goto mixed_cleanup;
                }
                if (event.kind == LC_EVENT_NUMERICAL_CONTINUATION) {
                    status = lc_mixed_schedule_prediction(
                        active_nodes, state, t_last, active_parameters, runtime,
                        event.index, config, &heap, error, variables, workspace,
                        workspace_count);
                    if (status != LC_OK) goto mixed_cleanup;
                    continue;
                }
                fired[event.index] = 1U;
                stats->autonomous_spikes_confirmed++;
            }

            for (node = 0; node < node_count; ++node) {
                if (fired[node]) {
                    any_fired = 1;
                    break;
                }
            }
            if (any_fired) {
                cascade_depth++;
                if (cascade_depth > config->same_time_cascade_limit) {
                    status = LC_CASCADE_LIMIT;
                    goto mixed_cleanup;
                }
                if (cascade_depth > stats->max_same_time_cascade_depth) {
                    stats->max_same_time_cascade_depth = cascade_depth;
                }
                status = lc_mixed_fire_nodes(
                    active_nodes, state, t_last, active_parameters, runtime,
                    node_count, edges, outgoing_offsets, outgoing_edges, config,
                    &heap, fired, NULL, 0U, t, outputs, output_count, NULL,
                    stats, error,
                    NULL, variables, workspace, workspace_count, NULL, NULL,
                    NULL
                );
                if (status != LC_OK) {
                    goto mixed_cleanup;
                }
            }
            if (!lc_at(&heap, t, LC_PHASE_DEPOSIT)) {
                break;
            }
        }
    }

    for (node = 0; node < node_count; ++node) {
        status = lc_mixed_advance_node(
            active_nodes, state, t_last, active_parameters, runtime, node,
            config->t_end, variables, workspace, workspace_count
        );
        if (status != LC_OK) {
            goto mixed_cleanup;
        }
    }

mixed_cleanup:
    stats->peak_queue_occupancy = heap.peak;
    stats->kernel_seconds = lc_monotonic_seconds() - kernel_start;
    free(heap.items);
    free(active_nodes);
    free(active_parameters);
    lc_runtime_clear_caches(runtime, node_count);
    free(runtime);
    free(state_deposits);
    free(program_deposits);
    free(affected);
    free(deposit_affected);
    free(timestamp_reset);
    free(boundary_affected);
    free(fired);
    free(outgoing_offsets);
    free(outgoing_cursors);
    free(outgoing_edges);
    free(workspace);
    return status;
}

static void lc_compiled_graph_release(lc_compiled_graph *compiled) {
    if (compiled == NULL) {
        return;
    }
    if (compiled->references > 1U) {
        compiled->references--;
        return;
    }
    free(compiled->nodes);
    free(compiled->expression_nodes);
    free(compiled->specialized_expression_nodes);
    free(compiled->node_eval_plans);
    free(compiled->parameters);
    free(compiled->edges);
    free(compiled->delivery_groups);
    free(compiled->delivery_group_edges);
    free(compiled->outgoing_offsets);
    free(compiled->plasticity);
    free(compiled->learning_programs);
    free(compiled->learning_expression_nodes);
    free(compiled->learning_parameters);
    free(compiled->learning_bindings);
    free(compiled->learning_observer_program_by_node);
    free(compiled->learning_observer_parameter_offset_by_node);
    free(compiled->plastic_slot_by_edge);
    free(compiled->plastic_edges);
    free(compiled->weight_master_slots);
    free(compiled->weight_learning_scales);
    free(compiled->incoming_plastic_offsets);
    free(compiled->incoming_plastic_slots);
    free(compiled->incoming_learning_batch_offsets);
    free(compiled->incoming_learning_batches);
    free(compiled->modulator_offsets);
    free(compiled->modulator_slots);
    free(compiled->modulation_learning_batch_offsets);
    free(compiled->modulation_learning_batches);
    free(compiled);
}

/* Validate a legacy rule and every parameter used by its event maps. */
static int lc_plasticity_rule_valid(
    const lc_plasticity_rule *rule,
    lc_real_t weight,
    uint32_t edge_count
) {
    if (rule == NULL || rule->kind > LC_PLASTICITY_MODULATED ||
        rule->weight_group > edge_count) {
        return 0;
    }
    if (rule->kind == LC_PLASTICITY_NONE) {
        return 1;
    }
    if (rule->consume_on_modulation > 1U ||
        !lc_isfinite(rule->learning_rate) || rule->learning_rate < LC_REAL_C(0.0) ||
        !lc_isfinite(rule->weight_min) || rule->weight_min < LC_REAL_C(0.0) ||
        !lc_isfinite(rule->weight_max) || rule->weight_max <= rule->weight_min ||
        weight < rule->weight_min || weight > rule->weight_max) {
        return 0;
    }
    if (!lc_isfinite(rule->tau_pre) || rule->tau_pre <= LC_REAL_C(0.0) ||
        !lc_isfinite(rule->tau_post) || rule->tau_post <= LC_REAL_C(0.0)) {
        return 0;
    }
    if (rule->kind == LC_PLASTICITY_PAIR) {
        return lc_isfinite(rule->a2_plus) && rule->a2_plus >= LC_REAL_C(0.0) &&
               lc_isfinite(rule->a2_minus) && rule->a2_minus >= LC_REAL_C(0.0);
    }
    if (rule->kind == LC_PLASTICITY_TRIPLET) {
        return lc_isfinite(rule->tau_pre_slow) && rule->tau_pre_slow > LC_REAL_C(0.0) &&
               lc_isfinite(rule->tau_post_slow) && rule->tau_post_slow > LC_REAL_C(0.0) &&
               lc_isfinite(rule->a2_plus) && rule->a2_plus >= LC_REAL_C(0.0) &&
               lc_isfinite(rule->a2_minus) && rule->a2_minus >= LC_REAL_C(0.0) &&
               lc_isfinite(rule->a3_plus) && rule->a3_plus >= LC_REAL_C(0.0) &&
               lc_isfinite(rule->a3_minus) && rule->a3_minus >= LC_REAL_C(0.0);
    }
    return rule->modulator < edge_count &&
           lc_isfinite(rule->tau_eligibility_plus) &&
           rule->tau_eligibility_plus > LC_REAL_C(0.0) &&
           lc_isfinite(rule->tau_eligibility_minus) &&
           rule->tau_eligibility_minus > LC_REAL_C(0.0) &&
           lc_isfinite(rule->positive_plus) && lc_isfinite(rule->positive_minus) &&
           lc_isfinite(rule->negative_plus) && lc_isfinite(rule->negative_minus);
}

/* Confirm that all copies of a shared weight use compatible rules. */
static int lc_shared_rules_compatible(
    const lc_plasticity_rule *left,
    const lc_plasticity_rule *right
) {
    return left != NULL && right != NULL &&
           left->kind == right->kind &&
           left->consume_on_modulation == right->consume_on_modulation &&
           left->tau_pre == right->tau_pre &&
           left->tau_post == right->tau_post &&
           left->tau_pre_slow == right->tau_pre_slow &&
           left->tau_post_slow == right->tau_post_slow &&
           left->a2_plus == right->a2_plus &&
           left->a2_minus == right->a2_minus &&
           left->a3_plus == right->a3_plus &&
           left->a3_minus == right->a3_minus &&
           left->tau_eligibility_plus == right->tau_eligibility_plus &&
           left->tau_eligibility_minus == right->tau_eligibility_minus &&
           left->positive_plus == right->positive_plus &&
           left->positive_minus == right->positive_minus &&
           left->negative_plus == right->negative_plus &&
           left->negative_minus == right->negative_minus &&
           left->learning_rate == right->learning_rate &&
           left->weight_min == right->weight_min &&
           left->weight_max == right->weight_max;
}

typedef struct lc_edge_learning_layout {
    const lc_plasticity_rule *legacy_rules;
    const lc_learning_binding *bindings;
    const lc_learning_program *programs;
    const lc_real_t *parameters;
    uint32_t program_count;
    uint32_t parameter_count;
} lc_edge_learning_layout;

/* Report whether an edge owns any compiled learning behavior. */
static int lc_edge_learning_enabled(
    const lc_edge_learning_layout *layout,
    uint32_t edge
) {
    if (layout->bindings != NULL) {
        return layout->bindings[edge].program != UINT32_MAX;
    }
    return layout->legacy_rules != NULL &&
        layout->legacy_rules[edge].kind != LC_PLASTICITY_NONE;
}

/* Return the shared group from the active learning representation. */
static uint32_t lc_edge_learning_weight_group(
    const lc_edge_learning_layout *layout,
    uint32_t edge
) {
    return layout->bindings != NULL
        ? layout->bindings[edge].weight_group
        : layout->legacy_rules[edge].weight_group;
}

/* Return the modulator scope from the active learning representation. */
static uint32_t lc_edge_learning_modulator(
    const lc_edge_learning_layout *layout,
    uint32_t edge
) {
    if (layout->bindings != NULL) {
        return layout->bindings[edge].modulator;
    }
    return layout->legacy_rules[edge].kind == LC_PLASTICITY_MODULATED
        ? layout->legacy_rules[edge].modulator : UINT32_MAX;
}

/* Compare equation-derived bindings that share one weight. */
static int lc_edge_learning_bindings_compatible(
    const lc_edge_learning_layout *layout,
    uint32_t left_edge,
    uint32_t right_edge
) {
    if (layout->bindings == NULL) {
        return lc_shared_rules_compatible(
            &layout->legacy_rules[left_edge],
            &layout->legacy_rules[right_edge]
        );
    }
    {
        const lc_learning_binding *left = &layout->bindings[left_edge];
        const lc_learning_binding *right = &layout->bindings[right_edge];
        uint32_t count;
        if (left->program != right->program ||
            left->program >= layout->program_count ||
            left->weight_min != right->weight_min ||
            left->weight_max != right->weight_max) {
            return 0;
        }
        count = layout->programs[left->program].parameter_count;
        if (left->parameter_offset > layout->parameter_count ||
            right->parameter_offset > layout->parameter_count ||
            count > layout->parameter_count - left->parameter_offset ||
            count > layout->parameter_count - right->parameter_offset) {
            return 0;
        }
        return count == 0U || memcmp(
            &layout->parameters[left->parameter_offset],
            &layout->parameters[right->parameter_offset],
            count * sizeof(lc_real_t)
        ) == 0;
    }
}

/* Compilation owns immutable graph data and allocates edge-local state. */
static lc_status lc_mixed_graph_compile_layout(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    const lc_plasticity_rule *plasticity,
    const lc_learning_binding *learning_bindings,
    const lc_learning_program *learning_programs,
    uint32_t learning_program_count,
    const lc_real_t *learning_parameters,
    uint32_t learning_parameter_count,
    uint32_t edge_count,
    int specialize_expressions,
    lc_compiled_graph **compiled
) {
    lc_compiled_graph *result = NULL;
    uint64_t expression_count = 0U;
    uint64_t expression_cursor = 0U;
    uint64_t specialized_expression_count = 0U;
    uint64_t specialized_expression_cursor = 0U;
    uint32_t expected_state_offset = 0U;
    uint32_t expected_parameter_offset = 0U;
    uint32_t workspace_count = 1U;
    uint32_t node;
    uint32_t edge;
    uint32_t program_ref_count = 0U;
    uint32_t unique_program_count = 0U;
    uint32_t plastic_count = 0U;
    uint32_t modulator_count = 0U;
    int has_shared_weights = 0;
    uint64_t *cursors = NULL;
    uint64_t *modulator_cursors = NULL;
    uint64_t *edge_offsets = NULL;
    lc_delivery_sort_item *delivery_items = NULL;
    lc_expression_program_ref *program_refs = NULL;
    lc_edge_learning_layout learning_layout;
    lc_status specialization_status = LC_OK;
    size_t csr_node;

    learning_layout.legacy_rules = plasticity;
    learning_layout.bindings = learning_bindings;
    learning_layout.programs = learning_programs;
    learning_layout.parameters = learning_parameters;
    learning_layout.program_count = learning_program_count;
    learning_layout.parameter_count = learning_parameter_count;

    if (compiled == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *compiled = NULL;
    if (nodes == NULL || node_count == 0U || state_count == 0U ||
        (plasticity != NULL && learning_bindings != NULL) ||
        (parameter_count > 0U && parameters == NULL) ||
        (edge_count > 0U && edges == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0; node < node_count; ++node) {
        const lc_mixed_node *descriptor = &nodes[node];
        if (descriptor->state_offset != expected_state_offset ||
            descriptor->parameter_offset != expected_parameter_offset ||
            !lc_mixed_state_program_valid(descriptor) ||
            !lc_mixed_crossing_valid(descriptor) || !lc_isfinite(descriptor->threshold) ||
            !lc_isfinite(descriptor->refractory) || descriptor->refractory < LC_REAL_C(0.0)) {
            return LC_INVALID_ARGUMENT;
        }
        if (descriptor->program_node_count > workspace_count) {
            workspace_count = descriptor->program_node_count;
        }
        if (descriptor->deposit_node_count > workspace_count) {
            workspace_count = descriptor->deposit_node_count;
        }
        if (descriptor->state_count > UINT32_MAX - expected_state_offset ||
            descriptor->parameter_count > UINT32_MAX - expected_parameter_offset) {
            return LC_INVALID_ARGUMENT;
        }
        expected_state_offset += descriptor->state_count;
        expected_parameter_offset += descriptor->parameter_count;
    }
    if (expected_state_offset != state_count ||
        expected_parameter_offset != parameter_count ||
        lc_allocation_size_overflows(node_count, sizeof(lc_mixed_node)) ||
        lc_allocation_size_overflows(
            (uint64_t)node_count * 2U, sizeof(lc_expression_program_ref)
        ) ||
        lc_allocation_size_overflows(parameter_count, sizeof(lc_real_t)) ||
        lc_allocation_size_overflows(edge_count, sizeof(lc_mixed_edge)) ||
        lc_allocation_size_overflows(edge_count, sizeof(lc_delivery_group)) ||
        lc_allocation_size_overflows(edge_count, sizeof(lc_delivery_sort_item)) ||
        lc_allocation_size_overflows((uint64_t)node_count + 1U, sizeof(uint64_t)) ||
        lc_allocation_size_overflows(edge_count, sizeof(uint32_t))) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0; node < parameter_count; ++node) {
        if (!lc_isfinite(parameters[node])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (edge = 0; edge < edge_count; ++edge) {
        const lc_mixed_node *post;
        if (edges[edge].pre >= node_count || edges[edge].post >= node_count ||
            !lc_edge_weight_valid(
                nodes[edges[edge].pre].polarity, edges[edge].weight
            ) ||
            !lc_isfinite(edges[edge].delay) || edges[edge].delay < LC_REAL_C(0.0) ||
            !lc_isfinite(edges[edge].deposit_scale) ||
            edges[edge].deposit_scale <= LC_REAL_C(0.0)) {
            return LC_INVALID_ARGUMENT;
        }
        post = &nodes[edges[edge].post];
        if (!lc_mixed_deposit_valid(post, edges[edge].deposit_kind, edges[edge].target)) {
            return LC_INVALID_ARGUMENT;
        }
        if (plasticity != NULL &&
            !lc_plasticity_rule_valid(&plasticity[edge], edges[edge].weight, edge_count)) {
            return LC_INVALID_ARGUMENT;
        }
        if (lc_edge_learning_enabled(&learning_layout, edge)) {
            uint32_t weight_group = lc_edge_learning_weight_group(
                &learning_layout, edge
            );
            uint32_t modulator = lc_edge_learning_modulator(
                &learning_layout, edge
            );
            if (nodes[edges[edge].pre].polarity == LC_MIXED) {
                return LC_INVALID_ARGUMENT;
            }
            plastic_count++;
            if (weight_group > edge_count) {
                return LC_INVALID_ARGUMENT;
            }
            if (weight_group > 0U) {
                has_shared_weights = 1;
            }
            if (modulator != UINT32_MAX && modulator + 1U > modulator_count) {
                modulator_count = modulator + 1U;
            }
        }
    }

    program_refs = calloc(
        (size_t)node_count * 2U, sizeof(lc_expression_program_ref)
    );
    if (program_refs == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    for (node = 0U; node < node_count; ++node) {
        program_refs[program_ref_count].source = nodes[node].program_nodes;
        program_refs[program_ref_count].count = nodes[node].program_node_count;
        program_ref_count++;
        if (nodes[node].deposit_node_count > 0U) {
            program_refs[program_ref_count].source = nodes[node].deposit_nodes;
            program_refs[program_ref_count].count = nodes[node].deposit_node_count;
            program_ref_count++;
        }
    }
    qsort(
        program_refs, program_ref_count, sizeof(lc_expression_program_ref),
        lc_expression_program_ref_before
    );
    for (node = 0U; node < program_ref_count; ++node) {
        if (unique_program_count == 0U ||
            lc_expression_program_ref_before(
                &program_refs[node], &program_refs[unique_program_count - 1U]
            ) != 0) {
            program_refs[unique_program_count] = program_refs[node];
            expression_count += program_refs[node].count;
            unique_program_count++;
        }
    }
    for (node = 0U; node < unique_program_count; ++node) {
        program_refs[node].main_representative = UINT32_MAX;
        program_refs[node].deposit_representative = UINT32_MAX;
        program_refs[node].normal_specialized = NULL;
        program_refs[node].clamped_specialized = NULL;
        program_refs[node].reset_specialized = NULL;
        program_refs[node].crossing_specialized = NULL;
        program_refs[node].deposit_specialized = NULL;
    }
    if (lc_allocation_size_overflows(expression_count, sizeof(lc_expr_node))) {
        free(program_refs);
        return LC_INVALID_ARGUMENT;
    }

    result = calloc(1U, sizeof(lc_compiled_graph));
    if (result == NULL) {
        free(program_refs);
        return LC_ALLOCATION_FAILED;
    }
    result->references = 1U;
    result->node_count = node_count;
    result->state_count = state_count;
    result->parameter_count = parameter_count;
    result->edge_count = edge_count;
    result->workspace_count = workspace_count;
    result->plastic_edge_count = plastic_count;
    result->modulator_count = modulator_count;
    result->nodes = calloc(node_count, sizeof(lc_mixed_node));
    result->outgoing_offsets = calloc((size_t)node_count + 1U, sizeof(uint64_t));
    if (expression_count > 0U) {
        result->expression_nodes = calloc((size_t)expression_count, sizeof(lc_expr_node));
    }
    if (parameter_count > 0U) {
        result->parameters = calloc(parameter_count, sizeof(lc_real_t));
    }
    if (edge_count > 0U) {
        result->edges = calloc(edge_count, sizeof(lc_mixed_edge));
        result->delivery_groups = calloc(edge_count, sizeof(lc_delivery_group));
        result->delivery_group_edges = calloc(edge_count, sizeof(uint32_t));
        if (plasticity != NULL) {
            result->plasticity = calloc(edge_count, sizeof(lc_plasticity_rule));
        }
        result->plastic_slot_by_edge = calloc(edge_count, sizeof(uint32_t));
    }
    if (plastic_count > 0U) {
        result->plastic_edges = calloc(plastic_count, sizeof(uint32_t));
        if (has_shared_weights) {
            result->weight_master_slots = calloc(plastic_count, sizeof(uint32_t));
            result->weight_learning_scales = calloc(plastic_count, sizeof(lc_real_t));
        }
        result->incoming_plastic_slots = calloc(plastic_count, sizeof(uint32_t));
        result->incoming_plastic_offsets = calloc(
            (size_t)node_count + 1U, sizeof(uint64_t)
        );
    }
    if (modulator_count > 0U) {
        result->modulator_offsets = calloc(
            (size_t)modulator_count + 1U, sizeof(uint64_t)
        );
        result->modulator_slots = calloc(plastic_count, sizeof(uint32_t));
    }
    cursors = calloc(node_count, sizeof(uint64_t));
    edge_offsets = calloc((size_t)node_count + 1U, sizeof(uint64_t));
    if (edge_count > 0U) {
        delivery_items = calloc(edge_count, sizeof(lc_delivery_sort_item));
    }
    if (modulator_count > 0U) {
        modulator_cursors = calloc(modulator_count, sizeof(uint64_t));
    }
    if (result->nodes == NULL || result->outgoing_offsets == NULL || cursors == NULL ||
        edge_offsets == NULL ||
        (modulator_count > 0U && modulator_cursors == NULL) ||
        (expression_count > 0U && result->expression_nodes == NULL) ||
        (parameter_count > 0U && result->parameters == NULL) ||
        (edge_count > 0U &&
         (result->edges == NULL || result->delivery_groups == NULL ||
          result->delivery_group_edges == NULL || delivery_items == NULL ||
          (plasticity != NULL && result->plasticity == NULL) ||
          result->plastic_slot_by_edge == NULL)) ||
        (plastic_count > 0U &&
         (result->plastic_edges == NULL ||
          (has_shared_weights && (result->weight_master_slots == NULL ||
                                  result->weight_learning_scales == NULL)) ||
          result->incoming_plastic_slots == NULL ||
          result->incoming_plastic_offsets == NULL)) ||
        (modulator_count > 0U &&
         (result->modulator_offsets == NULL || result->modulator_slots == NULL))) {
        free(cursors);
        free(modulator_cursors);
        free(edge_offsets);
        free(delivery_items);
        free(program_refs);
        lc_compiled_graph_release(result);
        return LC_ALLOCATION_FAILED;
    }
    memcpy(result->nodes, nodes, node_count * sizeof(lc_mixed_node));
    if (parameter_count > 0U) {
        memcpy(result->parameters, parameters, parameter_count * sizeof(lc_real_t));
    }
    if (edge_count > 0U) {
        memcpy(result->edges, edges, edge_count * sizeof(lc_mixed_edge));
        if (plasticity != NULL) {
            memcpy(
                result->plasticity, plasticity,
                edge_count * sizeof(lc_plasticity_rule)
            );
        }
        for (edge = 0U; edge < edge_count; ++edge) {
            result->plastic_slot_by_edge[edge] = UINT32_MAX;
        }
    }
    for (node = 0U; node < unique_program_count; ++node) {
        program_refs[node].owned = &result->expression_nodes[expression_cursor];
        memcpy(
            program_refs[node].owned, program_refs[node].source,
            program_refs[node].count * sizeof(lc_expr_node)
        );
        expression_cursor += program_refs[node].count;
    }
    for (node = 0U; node < node_count; ++node) {
        lc_mixed_node *target = &result->nodes[node];
        const lc_mixed_node *source = &nodes[node];
        lc_expression_program_ref *program = lc_expression_program_find(
            program_refs, unique_program_count, source->program_nodes,
            source->program_node_count
        );
        lc_expression_program_ref *deposit = NULL;
        if (source->deposit_node_count > 0U) {
            deposit = lc_expression_program_find(
                program_refs, unique_program_count, source->deposit_nodes,
                source->deposit_node_count
            );
        }
        if (program == NULL ||
            (source->deposit_node_count > 0U && deposit == NULL)) {
            free(cursors);
            free(modulator_cursors);
            free(edge_offsets);
            free(delivery_items);
            free(program_refs);
            lc_compiled_graph_release(result);
            return LC_INVALID_ARGUMENT;
        }
        target->program_nodes = program->owned;
        target->deposit_nodes = deposit == NULL ? NULL : deposit->owned;
        if (program->main_representative == UINT32_MAX) {
            program->main_representative = node;
        }
        if (deposit != NULL && deposit->deposit_representative == UINT32_MAX) {
            deposit->deposit_representative = node;
        }
    }
    if (specialize_expressions) {
        for (node = 0U; node < unique_program_count; ++node) {
            const lc_expression_program_ref *program = &program_refs[node];
            if (program->main_representative != UINT32_MAX) {
                const lc_mixed_node *representative =
                    &nodes[program->main_representative];
                if (representative->dispatch != LC_STEPPED &&
                    representative->crossing_kind != LC_CROSSING_NUMERICAL) {
                    if (program->count >
                        (UINT64_MAX - specialized_expression_count) / 4U) {
                        specialization_status = LC_INVALID_ARGUMENT;
                        break;
                    }
                    specialized_expression_count += 4U * program->count;
                }
            }
            if (program->deposit_representative != UINT32_MAX) {
                if (program->count >
                    UINT64_MAX - specialized_expression_count) {
                    specialization_status = LC_INVALID_ARGUMENT;
                    break;
                }
                specialized_expression_count += program->count;
            }
        }
        if (specialization_status == LC_OK &&
            lc_allocation_size_overflows(
                specialized_expression_count, sizeof(lc_expr_node)
            )) {
            specialization_status = LC_INVALID_ARGUMENT;
        }
        if (specialization_status == LC_OK) {
            result->node_eval_plans = calloc(
                node_count, sizeof(lc_node_eval_plan)
            );
            if (specialized_expression_count > 0U) {
                result->specialized_expression_nodes = calloc(
                    (size_t)specialized_expression_count, sizeof(lc_expr_node)
                );
            }
            if (result->node_eval_plans == NULL ||
                (specialized_expression_count > 0U &&
                 result->specialized_expression_nodes == NULL)) {
                specialization_status = LC_ALLOCATION_FAILED;
            }
        }
        for (node = 0U;
             specialization_status == LC_OK && node < unique_program_count;
             ++node) {
            lc_expression_program_ref *program = &program_refs[node];
            if (program->main_representative != UINT32_MAX) {
                const lc_mixed_node *representative =
                    &nodes[program->main_representative];
                if (representative->dispatch != LC_STEPPED &&
                    representative->crossing_kind != LC_CROSSING_NUMERICAL) {
                    uint32_t crossing_roots[LC_EVAL_PLAN_MAX_ROOTS];
                    uint32_t crossing_root_count = 0U;
                    if (!lc_mixed_crossing_roots(
                            representative, crossing_roots,
                            &crossing_root_count)) {
                        specialization_status = LC_UNSUPPORTED_MODEL;
                        break;
                    }
                    program->normal_specialized =
                        &result->specialized_expression_nodes[
                            specialized_expression_cursor
                        ];
                    specialized_expression_cursor += program->count;
                    program->clamped_specialized =
                        &result->specialized_expression_nodes[
                            specialized_expression_cursor
                        ];
                    specialized_expression_cursor += program->count;
                    program->reset_specialized =
                        &result->specialized_expression_nodes[
                            specialized_expression_cursor
                        ];
                    specialized_expression_cursor += program->count;
                    program->crossing_specialized =
                        &result->specialized_expression_nodes[
                            specialized_expression_cursor
                        ];
                    specialized_expression_cursor += program->count;
                    specialization_status = lc_expr_specialize_roots(
                        program->owned, program->count,
                        representative->normal_roots,
                        representative->state_count,
                        program->normal_specialized
                    );
                    if (specialization_status == LC_OK) {
                        specialization_status = lc_expr_specialize_roots(
                            program->owned, program->count,
                            representative->clamped_roots,
                            representative->state_count,
                            program->clamped_specialized
                        );
                    }
                    if (specialization_status == LC_OK) {
                        specialization_status = lc_expr_specialize_roots(
                            program->owned, program->count,
                            representative->reset_roots,
                            representative->state_count,
                            program->reset_specialized
                        );
                    }
                    if (specialization_status == LC_OK) {
                        specialization_status = lc_expr_specialize_roots(
                            program->owned, program->count, crossing_roots,
                            crossing_root_count,
                            program->crossing_specialized
                        );
                    }
                }
            }
            if (specialization_status == LC_OK &&
                program->deposit_representative != UINT32_MAX) {
                const lc_mixed_node *representative =
                    &nodes[program->deposit_representative];
                program->deposit_specialized =
                    &result->specialized_expression_nodes[
                        specialized_expression_cursor
                    ];
                specialized_expression_cursor += program->count;
                specialization_status = lc_expr_specialize_roots(
                    program->owned, program->count,
                    &representative->deposit_root, 1U,
                    program->deposit_specialized
                );
            }
        }
        for (node = 0U;
             specialization_status == LC_OK && node < node_count;
             ++node) {
            const lc_mixed_node *source = &nodes[node];
            lc_node_eval_plan *plan = &result->node_eval_plans[node];
            lc_expression_program_ref *program = lc_expression_program_find(
                program_refs, unique_program_count, source->program_nodes,
                source->program_node_count
            );
            lc_expression_program_ref *deposit = NULL;
            plan->normal_nodes = result->nodes[node].program_nodes;
            plan->clamped_nodes = result->nodes[node].program_nodes;
            plan->reset_nodes = result->nodes[node].program_nodes;
            plan->crossing_nodes = result->nodes[node].program_nodes;
            plan->deposit_nodes = result->nodes[node].deposit_nodes;
            if (program != NULL && program->normal_specialized != NULL &&
                source->dispatch != LC_STEPPED &&
                source->crossing_kind != LC_CROSSING_NUMERICAL) {
                const lc_mixed_node *representative =
                    &nodes[program->main_representative];
                uint32_t source_crossing_roots[LC_EVAL_PLAN_MAX_ROOTS];
                uint32_t representative_crossing_roots[LC_EVAL_PLAN_MAX_ROOTS];
                uint32_t source_crossing_count = 0U;
                uint32_t representative_crossing_count = 0U;
                if (source->state_count == representative->state_count &&
                    memcmp(
                        source->normal_roots, representative->normal_roots,
                        source->state_count * sizeof(uint32_t)
                    ) == 0) {
                    plan->normal_nodes = program->normal_specialized;
                }
                if (source->state_count == representative->state_count &&
                    memcmp(
                        source->clamped_roots, representative->clamped_roots,
                        source->state_count * sizeof(uint32_t)
                    ) == 0) {
                    plan->clamped_nodes = program->clamped_specialized;
                }
                if (source->state_count == representative->state_count &&
                    memcmp(
                        source->reset_roots, representative->reset_roots,
                        source->state_count * sizeof(uint32_t)
                    ) == 0) {
                    plan->reset_nodes = program->reset_specialized;
                }
                if (lc_mixed_crossing_roots(
                        source, source_crossing_roots,
                        &source_crossing_count) &&
                    lc_mixed_crossing_roots(
                        representative, representative_crossing_roots,
                        &representative_crossing_count) &&
                    lc_root_lists_equal(
                        source_crossing_roots, source_crossing_count,
                        representative_crossing_roots,
                        representative_crossing_count)) {
                    plan->crossing_nodes = program->crossing_specialized;
                }
            }
            if (source->deposit_node_count > 0U) {
                deposit = lc_expression_program_find(
                    program_refs, unique_program_count, source->deposit_nodes,
                    source->deposit_node_count
                );
                if (deposit != NULL && deposit->deposit_specialized != NULL &&
                    source->deposit_root ==
                        nodes[deposit->deposit_representative].deposit_root) {
                    plan->deposit_nodes = deposit->deposit_specialized;
                }
            }
        }
        if (specialization_status == LC_OK &&
            specialized_expression_cursor != specialized_expression_count) {
            specialization_status = LC_INVALID_ARGUMENT;
        }
        if (specialization_status != LC_OK) {
            free(cursors);
            free(modulator_cursors);
            free(edge_offsets);
            free(delivery_items);
            free(program_refs);
            lc_compiled_graph_release(result);
            return specialization_status;
        }
    }
    free(program_refs);
    program_refs = NULL;
    for (edge = 0; edge < edge_count; ++edge) {
        edge_offsets[edges[edge].pre + 1U]++;
    }
    for (csr_node = 1U; csr_node <= (size_t)node_count; ++csr_node) {
        edge_offsets[csr_node] += edge_offsets[csr_node - 1U];
    }
    memcpy(cursors, edge_offsets, node_count * sizeof(uint64_t));
    for (edge = 0; edge < edge_count; ++edge) {
        uint32_t source = edges[edge].pre;
        uint64_t position = cursors[source]++;
        delivery_items[position].delay = edges[edge].delay;
        delivery_items[position].edge = edge;
    }
    {
        uint32_t group_count = 0U;
        uint64_t group_edge_cursor = 0U;
        for (node = 0U; node < node_count; ++node) {
            uint64_t start = edge_offsets[node];
            uint64_t stop = edge_offsets[node + 1U];
            uint64_t position;
            if (stop - start > 1U) {
                qsort(
                    &delivery_items[start], (size_t)(stop - start),
                    sizeof(lc_delivery_sort_item), lc_delivery_sort_before
                );
            }
            result->outgoing_offsets[node] = group_count;
            for (position = start; position < stop; ++position) {
                const lc_delivery_sort_item *item = &delivery_items[position];
                lc_delivery_group *group;
                if (position == start ||
                    item->delay != result->delivery_groups[group_count - 1U].delay) {
                    group = &result->delivery_groups[group_count];
                    group->edge_offset = group_edge_cursor;
                    group->first_edge = item->edge;
                    group->delay = item->delay;
                    group_count++;
                }
                group = &result->delivery_groups[group_count - 1U];
                result->delivery_group_edges[group_edge_cursor++] = item->edge;
                group->edge_count++;
            }
            result->outgoing_offsets[node + 1U] = group_count;
        }
        result->delivery_group_count = group_count;
        if (group_count == edge_count) {
            memcpy(cursors, edge_offsets, node_count * sizeof(uint64_t));
            for (edge = 0U; edge < edge_count; ++edge) {
                uint32_t source = edges[edge].pre;
                result->delivery_group_edges[cursors[source]++] = edge;
            }
        }
    }
    free(edge_offsets);
    edge_offsets = NULL;
    free(delivery_items);
    delivery_items = NULL;
    if (plastic_count > 0U) {
        uint32_t slot = 0U;
        uint32_t *group_masters = NULL;
        uint32_t *group_counts = NULL;
        if (has_shared_weights) {
            group_masters = malloc(edge_count * sizeof(uint32_t));
            group_counts = calloc(edge_count, sizeof(uint32_t));
        }
        if (has_shared_weights &&
            (group_masters == NULL || group_counts == NULL)) {
            free(group_masters);
            free(group_counts);
            free(cursors);
            free(modulator_cursors);
            lc_compiled_graph_release(result);
            return LC_ALLOCATION_FAILED;
        }
        if (has_shared_weights) {
            for (edge = 0U; edge < edge_count; ++edge) {
                group_masters[edge] = UINT32_MAX;
            }
        }
        for (edge = 0U; edge < edge_count; ++edge) {
            uint32_t modulator;
            if (!lc_edge_learning_enabled(&learning_layout, edge)) {
                continue;
            }
            result->plastic_slot_by_edge[edge] = slot;
            result->plastic_edges[slot] = edge;
            result->incoming_plastic_offsets[edges[edge].post + 1U]++;
            modulator = lc_edge_learning_modulator(&learning_layout, edge);
            if (modulator != UINT32_MAX) {
                result->modulator_offsets[modulator + 1U]++;
            }
            slot++;
        }
        if (has_shared_weights) {
            for (slot = 0U; slot < plastic_count; ++slot) {
                uint32_t plastic_edge = result->plastic_edges[slot];
                uint32_t encoded_group = lc_edge_learning_weight_group(
                    &learning_layout, plastic_edge
                );
                if (encoded_group == 0U) {
                    result->weight_master_slots[slot] = slot;
                    result->weight_learning_scales[slot] = LC_REAL_C(1.0);
                    continue;
                }
                {
                    uint32_t group = encoded_group - 1U;
                    uint32_t master = group_masters[group];
                    if (master == UINT32_MAX) {
                        group_masters[group] = slot;
                        master = slot;
                    } else {
                        uint32_t master_edge = result->plastic_edges[master];
                        if (result->edges[master_edge].weight !=
                                result->edges[plastic_edge].weight ||
                            result->nodes[result->edges[master_edge].pre].polarity !=
                                result->nodes[result->edges[plastic_edge].pre].polarity ||
                            result->edges[master_edge].deposit_kind !=
                                result->edges[plastic_edge].deposit_kind ||
                            result->edges[master_edge].target !=
                                result->edges[plastic_edge].target ||
                            result->edges[master_edge].deposit_scale !=
                                result->edges[plastic_edge].deposit_scale ||
                            !lc_edge_learning_bindings_compatible(
                                &learning_layout, master_edge, plastic_edge
                            )) {
                            free(group_masters);
                            free(group_counts);
                            free(cursors);
                            free(modulator_cursors);
                            lc_compiled_graph_release(result);
                            return LC_INVALID_ARGUMENT;
                        }
                    }
                    result->weight_master_slots[slot] = master;
                    group_counts[group]++;
                }
            }
            for (slot = 0U; slot < plastic_count; ++slot) {
                uint32_t plastic_edge = result->plastic_edges[slot];
                uint32_t encoded_group = lc_edge_learning_weight_group(
                    &learning_layout, plastic_edge
                );
                if (encoded_group > 0U) {
                    uint32_t count = group_counts[encoded_group - 1U];
                    if (count == 0U) {
                        free(group_masters);
                        free(group_counts);
                        free(cursors);
                        free(modulator_cursors);
                        lc_compiled_graph_release(result);
                        return LC_INVALID_ARGUMENT;
                    }
                    result->weight_learning_scales[slot] = LC_REAL_C(1.0) / (lc_real_t)count;
                }
            }
        }
        free(group_masters);
        free(group_counts);
        for (csr_node = 1U; csr_node <= (size_t)node_count; ++csr_node) {
            result->incoming_plastic_offsets[csr_node] +=
                result->incoming_plastic_offsets[csr_node - 1U];
        }
        memcpy(
            cursors, result->incoming_plastic_offsets,
            node_count * sizeof(uint64_t)
        );
        for (slot = 0U; slot < plastic_count; ++slot) {
            uint32_t plastic_edge = result->plastic_edges[slot];
            uint32_t post = edges[plastic_edge].post;
            result->incoming_plastic_slots[cursors[post]++] = slot;
        }
        if (modulator_count > 0U) {
            for (csr_node = 1U; csr_node <= (size_t)modulator_count; ++csr_node) {
                result->modulator_offsets[csr_node] +=
                    result->modulator_offsets[csr_node - 1U];
            }
            memcpy(
                modulator_cursors, result->modulator_offsets,
                modulator_count * sizeof(uint64_t)
            );
            for (slot = 0U; slot < plastic_count; ++slot) {
                uint32_t plastic_edge = result->plastic_edges[slot];
                uint32_t modulator = lc_edge_learning_modulator(
                    &learning_layout, plastic_edge
                );
                if (modulator != UINT32_MAX) {
                    result->modulator_slots[modulator_cursors[modulator]++] = slot;
                }
            }
        }
    }
    free(cursors);
    free(modulator_cursors);
    *compiled = result;
    return LC_OK;
}

/* Compile a graph using compatibility plasticity rule descriptors. */
lc_status lc_mixed_graph_compile_plastic(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    const lc_plasticity_rule *plasticity,
    uint32_t edge_count,
    lc_compiled_graph **compiled
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (compiled != NULL) *compiled = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_mixed_graph_compile_layout(
        nodes, node_count, state_count, parameters, parameter_count,
        edges, plasticity, NULL, NULL, 0U, NULL, 0U,
        edge_count, 0, compiled
    );
}

/* Compile a graph with static edge weights. */
lc_status lc_mixed_graph_compile(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    uint32_t edge_count,
    lc_compiled_graph **compiled
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (compiled != NULL) *compiled = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_mixed_graph_compile_plastic(
        nodes, node_count, state_count, parameters, parameter_count,
        edges, NULL, edge_count, compiled
    );
}

/* Validate a learning expression DAG and its permitted variable masks. */
static int lc_learning_expression_valid(
    const lc_expr_node *nodes,
    uint32_t node_count,
    uint32_t parameter_count,
    uint32_t variable_count
) {
    uint32_t index;
    if (nodes == NULL || node_count == 0U) {
        return 0;
    }
    for (index = 0U; index < node_count; ++index) {
        const lc_expr_node *node = &nodes[index];
        switch ((lc_expr_op)node->op) {
            case LC_EXPR_CONST:
                if (!lc_isfinite(node->value)) return 0;
                break;
            case LC_EXPR_PARAM:
                if (node->binding >= parameter_count) return 0;
                break;
            case LC_EXPR_VAR:
                if (node->binding >= variable_count) return 0;
                break;
            case LC_EXPR_NEG:
            case LC_EXPR_EXP:
            case LC_EXPR_LOG:
            case LC_EXPR_PHI1:
            case LC_EXPR_PHI1_DERIV:
            case LC_EXPR_SIN:
            case LC_EXPR_COS:
            case LC_EXPR_TANH:
                if (node->lhs >= index) return 0;
                break;
            case LC_EXPR_ADD:
            case LC_EXPR_SUB:
            case LC_EXPR_MUL:
            case LC_EXPR_DIV:
            case LC_EXPR_POW:
            case LC_EXPR_MAX:
                if (node->lhs >= index || node->rhs >= index) return 0;
                break;
            default:
                return 0;
        }
    }
    return 1;
}

static int lc_learning_program_valid(const lc_learning_program *program) {
    uint32_t trace;
    uint32_t event_kind;
    uint32_t storage_mask = 0U;
    if (program == NULL || program->trace_count > LC_LEARNING_MAX_TRACES ||
        program->variable_count < 4U + program->trace_count ||
        program->variable_count >
            LC_LEARNING_BASE_VARIABLE_COUNT + program->trace_count ||
        program->clamp_normalized_weight > 1U ||
        program->compatibility_kind > LC_PLASTICITY_MODULATED) {
        return 0;
    }
    if ((program->observer.node_count == 0U &&
         program->variable_count > 5U + program->trace_count) ||
        (program->observer.node_count > 0U &&
         program->variable_count < 6U + program->trace_count)) {
        return 0;
    }
    for (trace = 0U; trace < program->trace_count; ++trace) {
        uint32_t storage = program->trace_storage_slots[trace];
        if (program->trace_tau_parameters[trace] >= program->parameter_count ||
            storage >= LC_LEARNING_MAX_TRACES ||
            (storage_mask & (UINT32_C(1) << storage)) != 0U) {
            return 0;
        }
        storage_mask |= UINT32_C(1) << storage;
    }
    for (event_kind = 0U; event_kind < LC_LEARNING_EVENT_COUNT; ++event_kind) {
        const lc_learning_event_program *event = &program->events[event_kind];
        uint32_t update;
        uint32_t update_mask = 0U;
        uint32_t valid_trace_mask = program->trace_count == 0U
            ? 0U : (UINT32_C(1) << program->trace_count) - 1U;
        if (event->node_count == 0U) {
            if (event->nodes != NULL || event->advance_mask != 0U ||
                event->variable_mask != 0U ||
                event->weight_root != UINT32_MAX ||
                event->trace_update_count != 0U) {
                return 0;
            }
            continue;
        }
        if (!lc_learning_expression_valid(
                event->nodes, event->node_count,
                program->parameter_count, program->variable_count) ||
            (event->advance_mask & ~valid_trace_mask) != 0U ||
            (event->variable_mask & ~(
                (UINT32_C(1) << program->variable_count) - 1U
            )) != 0U ||
            event->trace_update_count > program->trace_count ||
            (event->weight_root != UINT32_MAX &&
             event->weight_root >= event->node_count)) {
            return 0;
        }
        {
            uint32_t node_index;
            uint32_t variable_mask = 0U;
            for (node_index = 0U; node_index < event->node_count; ++node_index) {
                if (event->nodes[node_index].op == LC_EXPR_VAR) {
                    variable_mask |= UINT32_C(1) <<
                        event->nodes[node_index].binding;
                }
            }
            if (event->variable_mask != variable_mask) {
                return 0;
            }
        }
        for (update = 0U; update < event->trace_update_count; ++update) {
            uint32_t trace_index = event->trace_indices[update];
            if (trace_index >= program->trace_count ||
                event->trace_roots[update] >= event->node_count ||
                (event->advance_mask & (UINT32_C(1) << trace_index)) == 0U ||
                (update_mask & (UINT32_C(1) << trace_index)) != 0U) {
                return 0;
            }
            update_mask |= UINT32_C(1) << trace_index;
        }
    }
    if (program->observer.node_count == 0U) {
        if (program->observer.nodes != NULL ||
            program->observer.variable_mask != 0U) {
            return 0;
        }
    } else {
        const lc_learning_observer_program *observer = &program->observer;
        uint32_t index;
        uint32_t variable_mask = 0U;
        uint32_t parameter_mask = 0U;
        if (!lc_learning_expression_valid(
                observer->nodes, observer->node_count,
                program->parameter_count, 6U) ||
            program->parameter_count > 31U ||
            observer->voltage_tau_parameter >= program->parameter_count ||
            observer->fast_activity_tau_parameter >= program->parameter_count ||
            observer->slow_activity_tau_parameter >= program->parameter_count ||
            observer->band_width_parameter >= program->parameter_count ||
            observer->fast_activity_root >= observer->node_count ||
            observer->slow_activity_root >= observer->node_count ||
            observer->gain_root >= observer->node_count) {
            return 0;
        }
        for (index = 0U; index < observer->node_count; ++index) {
            if (observer->nodes[index].op == LC_EXPR_VAR) {
                variable_mask |= UINT32_C(1) << observer->nodes[index].binding;
            } else if (observer->nodes[index].op == LC_EXPR_PARAM) {
                parameter_mask |= UINT32_C(1) <<
                    observer->nodes[index].binding;
            }
        }
        parameter_mask |=
            UINT32_C(1) << observer->voltage_tau_parameter;
        parameter_mask |=
            UINT32_C(1) << observer->fast_activity_tau_parameter;
        parameter_mask |=
            UINT32_C(1) << observer->slow_activity_tau_parameter;
        parameter_mask |=
            UINT32_C(1) << observer->band_width_parameter;
        if (observer->variable_mask != variable_mask ||
            observer->parameter_mask != parameter_mask) {
            return 0;
        }
    }
    return 1;
}

/* Validate the batches that group plastic slots by learning program. */
static int lc_loaded_learning_batches_valid(
    const lc_compiled_graph *graph,
    uint32_t scope_count,
    const uint64_t *slot_offsets,
    const uint32_t *slots,
    const uint64_t *batch_offsets,
    const lc_learning_batch *batches,
    uint32_t batch_count
) {
    uint32_t scope;
    if ((scope_count > 0U && slot_offsets == NULL) || batch_offsets == NULL ||
        (batch_count > 0U && batches == NULL)) {
        return 0;
    }
    if ((scope_count > 0U && slot_offsets[0] != 0U) ||
        batch_offsets[0] != 0U ||
        batch_offsets[scope_count] != batch_count) {
        return 0;
    }
    for (scope = 0U; scope < scope_count; ++scope) {
        uint64_t slot_cursor = slot_offsets[scope];
        uint64_t slot_stop = slot_offsets[scope + 1U];
        uint64_t batch_start = batch_offsets[scope];
        uint64_t batch_stop = batch_offsets[scope + 1U];
        uint64_t batch_index;
        uint32_t previous_program = 0U;
        int has_previous_program = 0;
        if (slot_stop < slot_cursor ||
            slot_stop > graph->plastic_edge_count ||
            batch_stop < batch_start || batch_stop > batch_count) {
            return 0;
        }
        for (batch_index = batch_start; batch_index < batch_stop;
             ++batch_index) {
            const lc_learning_batch *batch = &batches[batch_index];
            uint64_t end;
            uint64_t position;
            if (batch->slot_offset != slot_cursor ||
                batch->slot_count == 0U ||
                batch->program >= graph->learning_program_count ||
                (has_previous_program &&
                 batch->program <= previous_program) ||
                batch->slot_count > UINT64_MAX - slot_cursor) {
                return 0;
            }
            previous_program = batch->program;
            has_previous_program = 1;
            end = slot_cursor + batch->slot_count;
            if (end > slot_stop) {
                return 0;
            }
            for (position = slot_cursor; position < end; ++position) {
                uint32_t slot = slots[position];
                uint32_t edge;
                if (slot >= graph->plastic_edge_count) {
                    return 0;
                }
                if (position > slot_cursor && slot <= slots[position - 1U]) {
                    return 0;
                }
                edge = graph->plastic_edges[slot];
                if (edge >= graph->edge_count ||
                    graph->learning_bindings[edge].program != batch->program) {
                    return 0;
                }
            }
            slot_cursor = end;
        }
        if (slot_cursor != slot_stop) {
            return 0;
        }
    }
    return 1;
}

/* Compare observers that share one postsynaptic neuron. */
static int lc_loaded_learning_observers_compatible(
    const lc_compiled_graph *graph,
    uint32_t left_program,
    uint32_t left_offset,
    uint32_t right_program,
    uint32_t right_offset
) {
    const lc_learning_program *left = &graph->learning_programs[left_program];
    const lc_learning_program *right = &graph->learning_programs[right_program];
    const lc_learning_observer_program *left_observer = &left->observer;
    const lc_learning_observer_program *right_observer = &right->observer;
    uint32_t parameter;
    if (left_observer->node_count == 0U ||
        left_observer->node_count != right_observer->node_count ||
        left_observer->variable_mask != right_observer->variable_mask ||
        left_observer->parameter_mask != right_observer->parameter_mask ||
        left_observer->voltage_tau_parameter !=
            right_observer->voltage_tau_parameter ||
        left_observer->fast_activity_tau_parameter !=
            right_observer->fast_activity_tau_parameter ||
        left_observer->slow_activity_tau_parameter !=
            right_observer->slow_activity_tau_parameter ||
        left_observer->band_width_parameter !=
            right_observer->band_width_parameter ||
        left_observer->fast_activity_root !=
            right_observer->fast_activity_root ||
        left_observer->slow_activity_root !=
            right_observer->slow_activity_root ||
        left_observer->gain_root != right_observer->gain_root ||
        memcmp(
            left_observer->nodes, right_observer->nodes,
            left_observer->node_count * sizeof(lc_expr_node)
        ) != 0) {
        return 0;
    }
    for (parameter = 0U; parameter < 32U; ++parameter) {
        if ((left_observer->parameter_mask &
                (UINT32_C(1) << parameter)) != 0U &&
            graph->learning_parameters[left_offset + parameter] !=
                graph->learning_parameters[right_offset + parameter]) {
            return 0;
        }
    }
    return 1;
}

/* Reject a persisted graph before it can reach the execution runtime. */
int lc_compiled_graph_validate_loaded(const lc_compiled_graph *graph) {
    uint32_t expected_state_offset = 0U;
    uint32_t expected_parameter_offset = 0U;
    uint32_t expected_learning_parameter_offset = 0U;
    uint32_t expected_plastic_count = 0U;
    uint32_t expected_modulator_count = 0U;
    uint32_t required_workspace = 1U;
    uint8_t *seen = NULL;
    uint32_t *expected_observer_programs = NULL;
    uint32_t *expected_observer_offsets = NULL;
    uint32_t *group_masters = NULL;
    uint32_t *group_counts = NULL;
    uint32_t node;
    uint32_t edge;
    uint32_t slot;
    uint32_t program;
    int has_shared_weights = 0;
    int valid = 0;
    lc_edge_learning_layout learning_layout;
    if (graph == NULL || graph->references == 0U || graph->node_count == 0U ||
        graph->state_count == 0U || graph->nodes == NULL ||
        graph->outgoing_offsets == NULL || graph->workspace_count == 0U ||
        (graph->parameter_count > 0U && graph->parameters == NULL) ||
        (graph->edge_count > 0U &&
         (graph->edges == NULL || graph->delivery_groups == NULL ||
          graph->delivery_group_edges == NULL ||
          graph->plastic_slot_by_edge == NULL)) ||
        graph->delivery_group_count > graph->edge_count ||
        graph->plastic_edge_count > graph->edge_count ||
        (graph->plasticity != NULL && graph->learning_bindings != NULL) ||
        ((graph->learning_observer_program_by_node == NULL) !=
         (graph->learning_observer_parameter_offset_by_node == NULL)) ||
        (graph->plasticity != NULL &&
         graph->learning_observer_program_by_node != NULL) ||
        ((graph->learning_program_count > 0U ||
          graph->learning_parameter_count > 0U) &&
         graph->learning_observer_program_by_node == NULL) ||
        (graph->edge_count > 0U &&
         graph->learning_observer_program_by_node != NULL &&
         graph->learning_bindings == NULL) ||
        (graph->learning_program_count > 0U &&
         (graph->learning_programs == NULL ||
          graph->learning_bindings == NULL)) ||
        (graph->learning_parameter_count > 0U &&
         graph->learning_parameters == NULL) ||
        (graph->plastic_edge_count > 0U &&
         (graph->plastic_edges == NULL ||
          graph->incoming_plastic_offsets == NULL ||
          graph->incoming_plastic_slots == NULL))) {
        return 0;
    }

    learning_layout.legacy_rules = graph->plasticity;
    learning_layout.bindings = graph->learning_bindings;
    learning_layout.programs = graph->learning_programs;
    learning_layout.parameters = graph->learning_parameters;
    learning_layout.program_count = graph->learning_program_count;
    learning_layout.parameter_count = graph->learning_parameter_count;

    if (graph->learning_observer_program_by_node != NULL) {
        expected_observer_programs = malloc(
            graph->node_count * sizeof(uint32_t)
        );
        expected_observer_offsets = malloc(
            graph->node_count * sizeof(uint32_t)
        );
        if (expected_observer_programs == NULL ||
            expected_observer_offsets == NULL) {
            goto cleanup;
        }
        for (node = 0U; node < graph->node_count; ++node) {
            expected_observer_programs[node] = UINT32_MAX;
            expected_observer_offsets[node] = UINT32_MAX;
        }
    }

    for (node = 0U; node < graph->node_count; ++node) {
        const lc_mixed_node *descriptor = &graph->nodes[node];
        if (descriptor->state_offset != expected_state_offset ||
            descriptor->parameter_offset != expected_parameter_offset ||
            !lc_mixed_state_program_valid(descriptor) ||
            !lc_mixed_crossing_valid(descriptor) ||
            !lc_isfinite(descriptor->threshold) ||
            !lc_isfinite(descriptor->refractory) || descriptor->refractory < LC_REAL_C(0.0) ||
            !lc_learning_expression_valid(
                descriptor->program_nodes, descriptor->program_node_count,
                descriptor->parameter_count, descriptor->state_count + 1U
            ) || descriptor->state_count >
                UINT32_MAX - expected_state_offset ||
            descriptor->parameter_count >
                UINT32_MAX - expected_parameter_offset) {
            goto cleanup;
        }
        if (descriptor->deposit_node_count > 0U &&
            !lc_learning_expression_valid(
                descriptor->deposit_nodes, descriptor->deposit_node_count,
                descriptor->parameter_count, 1U
            )) {
            goto cleanup;
        }
        if (graph->node_eval_plans != NULL) {
            const lc_node_eval_plan *plan = &graph->node_eval_plans[node];
            if (!lc_learning_expression_valid(
                    plan->normal_nodes, descriptor->program_node_count,
                    descriptor->parameter_count, descriptor->state_count + 1U
                ) || !lc_learning_expression_valid(
                    plan->clamped_nodes, descriptor->program_node_count,
                    descriptor->parameter_count, descriptor->state_count + 1U
                ) || !lc_learning_expression_valid(
                    plan->reset_nodes, descriptor->program_node_count,
                    descriptor->parameter_count, descriptor->state_count + 1U
                ) || !lc_learning_expression_valid(
                    plan->crossing_nodes, descriptor->program_node_count,
                    descriptor->parameter_count, descriptor->state_count + 1U
                ) || (descriptor->deposit_node_count > 0U &&
                    !lc_learning_expression_valid(
                        plan->deposit_nodes, descriptor->deposit_node_count,
                        descriptor->parameter_count, 1U
                    )) || (descriptor->deposit_node_count == 0U &&
                            plan->deposit_nodes != NULL)) {
                goto cleanup;
            }
        }
        expected_state_offset += descriptor->state_count;
        expected_parameter_offset += descriptor->parameter_count;
        if (descriptor->program_node_count > required_workspace) {
            required_workspace = descriptor->program_node_count;
        }
        if (descriptor->deposit_node_count > required_workspace) {
            required_workspace = descriptor->deposit_node_count;
        }
    }
    if (expected_state_offset != graph->state_count ||
        expected_parameter_offset != graph->parameter_count ||
        required_workspace > graph->workspace_count) {
        goto cleanup;
    }
    for (node = 0U; node < graph->parameter_count; ++node) {
        if (!lc_isfinite(graph->parameters[node])) {
            goto cleanup;
        }
    }
    for (edge = 0U; edge < graph->edge_count; ++edge) {
        const lc_mixed_edge *descriptor = &graph->edges[edge];
        if (descriptor->pre >= graph->node_count ||
            descriptor->post >= graph->node_count ||
            !lc_edge_weight_valid(
                graph->nodes[descriptor->pre].polarity, descriptor->weight
            ) ||
            !lc_isfinite(descriptor->delay) || descriptor->delay < LC_REAL_C(0.0) ||
            !lc_isfinite(descriptor->deposit_scale) ||
            descriptor->deposit_scale <= LC_REAL_C(0.0) ||
            !lc_mixed_deposit_valid(
                &graph->nodes[descriptor->post], descriptor->deposit_kind,
                descriptor->target
            )) {
            goto cleanup;
        }
    }

    if (graph->outgoing_offsets[0] != 0U ||
        graph->outgoing_offsets[graph->node_count] !=
            (graph->delivery_group_count == graph->edge_count
                ? graph->edge_count : graph->delivery_group_count)) {
        goto cleanup;
    }
    if (graph->edge_count > 0U) {
        seen = calloc(graph->edge_count, sizeof(uint8_t));
        if (seen == NULL) {
            goto cleanup;
        }
    }
    if (graph->delivery_group_count == graph->edge_count) {
        for (node = 0U; node < graph->node_count; ++node) {
            uint64_t start = graph->outgoing_offsets[node];
            uint64_t stop = graph->outgoing_offsets[node + 1U];
            uint64_t position;
            if (stop < start || stop > graph->edge_count) {
                goto cleanup;
            }
            for (position = start; position < stop; ++position) {
                edge = graph->delivery_group_edges[position];
                if (edge >= graph->edge_count || seen[edge] != 0U ||
                    graph->edges[edge].pre != node) {
                    goto cleanup;
                }
                seen[edge] = 1U;
            }
        }
        if (graph->edge_count > 0U) {
            memset(seen, 0, graph->edge_count * sizeof(uint8_t));
        }
        for (node = 0U; node < graph->node_count; ++node) {
            uint64_t group_index;
            for (group_index = graph->outgoing_offsets[node];
                 group_index < graph->outgoing_offsets[node + 1U];
                 ++group_index) {
                const lc_delivery_group *group =
                    &graph->delivery_groups[group_index];
                edge = group->first_edge;
                if (group->edge_count != 1U ||
                    group->edge_offset != group_index ||
                    edge >= graph->edge_count || seen[edge] != 0U ||
                    graph->edges[edge].pre != node ||
                    group->delay != graph->edges[edge].delay) {
                    goto cleanup;
                }
                seen[edge] = 1U;
            }
        }
    } else {
        uint64_t expected_edge_cursor = 0U;
        for (node = 0U; node < graph->node_count; ++node) {
            uint64_t start = graph->outgoing_offsets[node];
            uint64_t stop = graph->outgoing_offsets[node + 1U];
            uint64_t group_index;
            if (stop < start || stop > graph->delivery_group_count) {
                goto cleanup;
            }
            for (group_index = start; group_index < stop; ++group_index) {
                const lc_delivery_group *group =
                    &graph->delivery_groups[group_index];
                uint64_t end;
                uint64_t position;
                if (group->edge_count == 0U ||
                    group->edge_offset != expected_edge_cursor ||
                    group->edge_count > UINT64_MAX - expected_edge_cursor) {
                    goto cleanup;
                }
                end = expected_edge_cursor + group->edge_count;
                if (end > graph->edge_count ||
                    group->first_edge >= graph->edge_count ||
                    graph->edges[group->first_edge].pre != node ||
                    group->delay != graph->edges[group->first_edge].delay) {
                    goto cleanup;
                }
                for (position = expected_edge_cursor; position < end;
                     ++position) {
                    edge = graph->delivery_group_edges[position];
                    if (edge >= graph->edge_count || seen[edge] != 0U ||
                        graph->edges[edge].pre != node ||
                        graph->edges[edge].delay != group->delay) {
                        goto cleanup;
                    }
                    seen[edge] = 1U;
                }
                expected_edge_cursor = end;
            }
        }
        if (expected_edge_cursor != graph->edge_count) {
            goto cleanup;
        }
    }

    for (program = 0U; program < graph->learning_program_count; ++program) {
        const lc_learning_program *descriptor =
            &graph->learning_programs[program];
        uint32_t event_kind;
        if (!lc_learning_program_valid(descriptor)) {
            goto cleanup;
        }
        for (event_kind = 0U; event_kind < LC_LEARNING_EVENT_COUNT;
             ++event_kind) {
            if (descriptor->events[event_kind].node_count >
                required_workspace) {
                required_workspace =
                    descriptor->events[event_kind].node_count;
            }
        }
        if (descriptor->observer.node_count > required_workspace) {
            required_workspace = descriptor->observer.node_count;
        }
    }
    if (required_workspace > graph->workspace_count) {
        goto cleanup;
    }
    for (program = 0U; program < graph->learning_parameter_count; ++program) {
        if (!lc_isfinite(graph->learning_parameters[program])) {
            goto cleanup;
        }
    }

    for (edge = 0U; edge < graph->edge_count; ++edge) {
        int enabled = 0;
        uint32_t modulator = UINT32_MAX;
        if (graph->plasticity != NULL) {
            if (!lc_plasticity_rule_valid(
                    &graph->plasticity[edge], graph->edges[edge].weight,
                    graph->edge_count
                )) {
                goto cleanup;
            }
            enabled = graph->plasticity[edge].kind != LC_PLASTICITY_NONE;
            if (graph->plasticity[edge].kind == LC_PLASTICITY_MODULATED) {
                modulator = graph->plasticity[edge].modulator;
            }
        } else if (graph->learning_bindings != NULL) {
            const lc_learning_binding *binding =
                &graph->learning_bindings[edge];
            if (binding->program == UINT32_MAX) {
                if (binding->parameter_offset !=
                        expected_learning_parameter_offset ||
                    binding->modulator != UINT32_MAX ||
                    binding->weight_group != 0U) {
                    goto cleanup;
                }
            } else {
                const lc_learning_program *learning;
                uint32_t trace;
                int has_modulation;
                if (binding->program >= graph->learning_program_count ||
                    binding->parameter_offset !=
                        expected_learning_parameter_offset ||
                    !lc_isfinite(binding->weight_min) ||
                    binding->weight_min < LC_REAL_C(0.0) ||
                    !lc_isfinite(binding->weight_max) ||
                    binding->weight_max <= binding->weight_min ||
                    graph->edges[edge].weight < binding->weight_min ||
                    graph->edges[edge].weight > binding->weight_max ||
                    binding->weight_group > graph->edge_count) {
                    goto cleanup;
                }
                learning = &graph->learning_programs[binding->program];
                if (learning->parameter_count >
                    graph->learning_parameter_count -
                        expected_learning_parameter_offset) {
                    goto cleanup;
                }
                has_modulation =
                    learning->events[LC_LEARNING_MODULATION_POSITIVE]
                        .node_count > 0U ||
                    learning->events[LC_LEARNING_MODULATION_NEGATIVE]
                        .node_count > 0U;
                if ((has_modulation &&
                     (binding->modulator == UINT32_MAX ||
                      binding->modulator >= graph->edge_count)) ||
                    (!has_modulation && binding->modulator != UINT32_MAX)) {
                    goto cleanup;
                }
                for (trace = 0U; trace < learning->trace_count; ++trace) {
                    lc_real_t tau = graph->learning_parameters[
                        expected_learning_parameter_offset +
                        learning->trace_tau_parameters[trace]
                    ];
                    if (!(tau > LC_REAL_C(0.0))) {
                        goto cleanup;
                    }
                }
                if (learning->observer.node_count > 0U) {
                    const lc_learning_observer_program *observer =
                        &learning->observer;
                    const lc_mixed_node *post =
                        &graph->nodes[graph->edges[edge].post];
                    lc_real_t tau_voltage = graph->learning_parameters[
                        expected_learning_parameter_offset +
                        observer->voltage_tau_parameter
                    ];
                    lc_real_t tau_fast = graph->learning_parameters[
                        expected_learning_parameter_offset +
                        observer->fast_activity_tau_parameter
                    ];
                    lc_real_t tau_slow = graph->learning_parameters[
                        expected_learning_parameter_offset +
                        observer->slow_activity_tau_parameter
                    ];
                    lc_real_t band_width = graph->learning_parameters[
                        expected_learning_parameter_offset +
                        observer->band_width_parameter
                    ];
                    if (!(tau_voltage > LC_REAL_C(0.0)) || !(tau_fast > LC_REAL_C(0.0)) ||
                        !(tau_slow > tau_fast) || !(band_width > LC_REAL_C(0.0)) ||
                        !((post->arithmetic_kind ==
                               LC_NODE_ARITHMETIC_SCALAR_AFFINE &&
                           post->scalar_affine_decay < LC_REAL_C(0.0)) ||
                          post->crossing_kind == LC_CROSSING_TWO_REAL_EXP) ||
                        graph->edges[edge].deposit_kind !=
                            LC_DEPOSIT_STATE_ADD ||
                        graph->edges[edge].target != post->readout) {
                        goto cleanup;
                    }
                    if (expected_observer_programs[
                            graph->edges[edge].post
                        ] == UINT32_MAX) {
                        expected_observer_programs[graph->edges[edge].post] =
                            binding->program;
                        expected_observer_offsets[graph->edges[edge].post] =
                            binding->parameter_offset;
                    } else if (!lc_loaded_learning_observers_compatible(
                            graph,
                            expected_observer_programs[
                                graph->edges[edge].post
                            ],
                            expected_observer_offsets[
                                graph->edges[edge].post
                            ],
                            binding->program,
                            binding->parameter_offset
                        )) {
                        goto cleanup;
                    }
                }
                enabled = 1;
                modulator = binding->modulator;
                expected_learning_parameter_offset +=
                    learning->parameter_count;
            }
        }
        if (enabled) {
            uint32_t weight_group = graph->plasticity != NULL
                ? graph->plasticity[edge].weight_group
                : graph->learning_bindings[edge].weight_group;
            if (graph->nodes[graph->edges[edge].pre].polarity == LC_MIXED) {
                goto cleanup;
            }
            if (expected_plastic_count == UINT32_MAX) {
                goto cleanup;
            }
            expected_plastic_count++;
            if (weight_group > 0U) {
                has_shared_weights = 1;
            }
            if (modulator != UINT32_MAX &&
                modulator + 1U > expected_modulator_count) {
                expected_modulator_count = modulator + 1U;
            }
        }
    }
    if (expected_learning_parameter_offset !=
            graph->learning_parameter_count ||
        expected_plastic_count != graph->plastic_edge_count ||
        expected_modulator_count != graph->modulator_count) {
        goto cleanup;
    }
    if (expected_observer_programs != NULL) {
        for (node = 0U; node < graph->node_count; ++node) {
            if (graph->learning_observer_program_by_node[node] !=
                    expected_observer_programs[node] ||
                graph->learning_observer_parameter_offset_by_node[node] !=
                    expected_observer_offsets[node]) {
                goto cleanup;
            }
        }
        for (edge = 0U; edge < graph->edge_count; ++edge) {
            uint32_t post = graph->edges[edge].post;
            if (expected_observer_programs[post] != UINT32_MAX &&
                (graph->edges[edge].deposit_kind != LC_DEPOSIT_STATE_ADD ||
                 graph->edges[edge].target != graph->nodes[post].readout)) {
                goto cleanup;
            }
        }
    }

    if (graph->edge_count > 0U) {
        memset(seen, 0, graph->edge_count * sizeof(uint8_t));
    }
    for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
        edge = graph->plastic_edges[slot];
        if (edge >= graph->edge_count || seen[edge] != 0U ||
            graph->plastic_slot_by_edge[edge] != slot) {
            goto cleanup;
        }
        seen[edge] = 1U;
    }
    for (edge = 0U; edge < graph->edge_count; ++edge) {
        int enabled = graph->plasticity != NULL
            ? graph->plasticity[edge].kind != LC_PLASTICITY_NONE
            : graph->learning_bindings != NULL &&
                graph->learning_bindings[edge].program != UINT32_MAX;
        if ((enabled && graph->plastic_slot_by_edge[edge] >=
                            graph->plastic_edge_count) ||
            (!enabled && graph->plastic_slot_by_edge[edge] != UINT32_MAX)) {
            goto cleanup;
        }
    }
    if ((graph->weight_master_slots == NULL) !=
        (graph->weight_learning_scales == NULL)) {
        goto cleanup;
    }
    if (has_shared_weights != (graph->weight_master_slots != NULL)) {
        goto cleanup;
    }
    if (has_shared_weights) {
        group_masters = malloc(graph->edge_count * sizeof(uint32_t));
        group_counts = calloc(graph->edge_count, sizeof(uint32_t));
        if (group_masters == NULL || group_counts == NULL) {
            goto cleanup;
        }
        for (edge = 0U; edge < graph->edge_count; ++edge) {
            group_masters[edge] = UINT32_MAX;
        }
        for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
            uint32_t plastic_edge = graph->plastic_edges[slot];
            uint32_t encoded_group = graph->plasticity != NULL
                ? graph->plasticity[plastic_edge].weight_group
                : graph->learning_bindings[plastic_edge].weight_group;
            uint32_t expected_master;
            if (encoded_group == 0U) {
                expected_master = slot;
            } else {
                uint32_t group = encoded_group - 1U;
                expected_master = group_masters[group];
                if (expected_master == UINT32_MAX) {
                    group_masters[group] = slot;
                    expected_master = slot;
                } else {
                    uint32_t master_edge =
                        graph->plastic_edges[expected_master];
                    if (graph->edges[master_edge].weight !=
                            graph->edges[plastic_edge].weight ||
                        graph->nodes[graph->edges[master_edge].pre].polarity !=
                            graph->nodes[graph->edges[plastic_edge].pre].polarity ||
                        graph->edges[master_edge].deposit_kind !=
                            graph->edges[plastic_edge].deposit_kind ||
                        graph->edges[master_edge].target !=
                            graph->edges[plastic_edge].target ||
                        graph->edges[master_edge].deposit_scale !=
                            graph->edges[plastic_edge].deposit_scale ||
                        !lc_edge_learning_bindings_compatible(
                            &learning_layout, master_edge, plastic_edge
                        )) {
                        goto cleanup;
                    }
                }
                group_counts[group]++;
            }
            if (graph->weight_master_slots[slot] != expected_master) {
                goto cleanup;
            }
        }
        for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
            uint32_t plastic_edge = graph->plastic_edges[slot];
            uint32_t encoded_group = graph->plasticity != NULL
                ? graph->plasticity[plastic_edge].weight_group
                : graph->learning_bindings[plastic_edge].weight_group;
            lc_real_t expected_scale = encoded_group == 0U
                ? LC_REAL_C(1.0) : LC_REAL_C(1.0) / (lc_real_t)group_counts[encoded_group - 1U];
            if (graph->weight_learning_scales[slot] != expected_scale) {
                goto cleanup;
            }
        }
    }

    if (graph->plastic_edge_count > 0U) {
        if (graph->incoming_plastic_offsets[0] != 0U ||
            graph->incoming_plastic_offsets[graph->node_count] !=
                graph->plastic_edge_count) {
            goto cleanup;
        }
        memset(seen, 0, graph->edge_count * sizeof(uint8_t));
        for (node = 0U; node < graph->node_count; ++node) {
            uint64_t start = graph->incoming_plastic_offsets[node];
            uint64_t stop = graph->incoming_plastic_offsets[node + 1U];
            uint64_t position;
            if (stop < start || stop > graph->plastic_edge_count) {
                goto cleanup;
            }
            for (position = start; position < stop; ++position) {
                slot = graph->incoming_plastic_slots[position];
                if (slot >= graph->plastic_edge_count || seen[slot] != 0U ||
                    graph->edges[graph->plastic_edges[slot]].post != node) {
                    goto cleanup;
                }
                seen[slot] = 1U;
            }
        }
    } else if (graph->incoming_plastic_offsets != NULL ||
               graph->incoming_plastic_slots != NULL) {
        goto cleanup;
    }

    if (graph->modulator_count > 0U) {
        uint64_t modulated_slot_count;
        if (graph->modulator_offsets == NULL ||
            graph->modulator_slots == NULL ||
            graph->modulator_offsets[0] != 0U) {
            goto cleanup;
        }
        modulated_slot_count =
            graph->modulator_offsets[graph->modulator_count];
        if (modulated_slot_count > graph->plastic_edge_count) {
            goto cleanup;
        }
        if (graph->plastic_edge_count > 0U) {
            memset(seen, 0, graph->edge_count * sizeof(uint8_t));
        }
        for (node = 0U; node < graph->modulator_count; ++node) {
            uint64_t start = graph->modulator_offsets[node];
            uint64_t stop = graph->modulator_offsets[node + 1U];
            uint64_t position;
            if (stop < start || stop > modulated_slot_count) {
                goto cleanup;
            }
            for (position = start; position < stop; ++position) {
                uint32_t expected;
                slot = graph->modulator_slots[position];
                if (slot >= graph->plastic_edge_count || seen[slot] != 0U) {
                    goto cleanup;
                }
                edge = graph->plastic_edges[slot];
                expected = graph->plasticity != NULL
                    ? graph->plasticity[edge].modulator
                    : graph->learning_bindings[edge].modulator;
                if (expected != node) {
                    goto cleanup;
                }
                seen[slot] = 1U;
            }
        }
        for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
            uint32_t plastic_edge = graph->plastic_edges[slot];
            uint32_t expected = graph->plasticity != NULL
                ? (graph->plasticity[plastic_edge].kind ==
                        LC_PLASTICITY_MODULATED
                    ? graph->plasticity[plastic_edge].modulator
                    : UINT32_MAX)
                : graph->learning_bindings[plastic_edge].modulator;
            if ((expected == UINT32_MAX && seen[slot] != 0U) ||
                (expected != UINT32_MAX && seen[slot] == 0U)) {
                goto cleanup;
            }
        }
    } else if (graph->modulator_offsets != NULL ||
               graph->modulator_slots != NULL) {
        goto cleanup;
    }

    if (graph->learning_bindings != NULL &&
        graph->plastic_edge_count > 0U) {
        if (!lc_loaded_learning_batches_valid(
                graph, graph->node_count, graph->incoming_plastic_offsets,
                graph->incoming_plastic_slots,
                graph->incoming_learning_batch_offsets,
                graph->incoming_learning_batches,
                graph->incoming_learning_batch_count
            ) || !lc_loaded_learning_batches_valid(
                graph, graph->modulator_count, graph->modulator_offsets,
                graph->modulator_slots,
                graph->modulation_learning_batch_offsets,
                graph->modulation_learning_batches,
                graph->modulation_learning_batch_count
            )) {
            goto cleanup;
        }
    } else if (graph->incoming_learning_batch_offsets != NULL ||
               graph->incoming_learning_batches != NULL ||
               graph->incoming_learning_batch_count != 0U ||
               graph->modulation_learning_batch_offsets != NULL ||
               graph->modulation_learning_batches != NULL ||
               graph->modulation_learning_batch_count != 0U) {
        goto cleanup;
    }

    valid = 1;

cleanup:
    free(seen);
    free(expected_observer_programs);
    free(expected_observer_offsets);
    free(group_masters);
    free(group_counts);
    return valid;
}

/* Build deterministic incoming and modulator batches by program. */
static lc_status lc_learning_build_scope_batches(
    const lc_compiled_graph *graph,
    uint32_t scope_count,
    const uint64_t *scope_offsets,
    uint32_t *slots,
    uint64_t **batch_offsets_out,
    lc_learning_batch **batches_out,
    uint32_t *batch_count_out
) {
    uint64_t total_slots;
    uint64_t *batch_offsets = NULL;
    lc_learning_batch *batches = NULL;
    lc_learning_sort_item *items = NULL;
    uint32_t batch_count = 0U;
    uint32_t scope;
    if (graph == NULL || batch_offsets_out == NULL || batches_out == NULL ||
        batch_count_out == NULL || (scope_count > 0U && scope_offsets == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *batch_offsets_out = NULL;
    *batches_out = NULL;
    *batch_count_out = 0U;
    total_slots = scope_count == 0U ? 0U : scope_offsets[scope_count];
    if (total_slots > UINT32_MAX ||
        lc_allocation_size_overflows((uint64_t)scope_count + 1U, sizeof(uint64_t)) ||
        lc_allocation_size_overflows(total_slots, sizeof(lc_learning_batch)) ||
        lc_allocation_size_overflows(total_slots, sizeof(lc_learning_sort_item)) ||
        (total_slots > 0U && slots == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    batch_offsets = calloc((size_t)scope_count + 1U, sizeof(uint64_t));
    if (total_slots > 0U) {
        batches = calloc((size_t)total_slots, sizeof(lc_learning_batch));
        items = calloc((size_t)total_slots, sizeof(lc_learning_sort_item));
    }
    if (batch_offsets == NULL ||
        (total_slots > 0U && (batches == NULL || items == NULL))) {
        free(batch_offsets);
        free(batches);
        free(items);
        return LC_ALLOCATION_FAILED;
    }
    for (scope = 0U; scope < scope_count; ++scope) {
        uint64_t start = scope_offsets[scope];
        uint64_t stop = scope_offsets[scope + 1U];
        uint64_t position;
        batch_offsets[scope] = batch_count;
        if (stop < start || stop > total_slots) {
            free(batch_offsets);
            free(batches);
            free(items);
            return LC_INVALID_ARGUMENT;
        }
        for (position = start; position < stop; ++position) {
            uint32_t slot = slots[position];
            uint32_t edge;
            if (slot >= graph->plastic_edge_count) {
                free(batch_offsets);
                free(batches);
                free(items);
                return LC_INVALID_ARGUMENT;
            }
            edge = graph->plastic_edges[slot];
            items[position].program = graph->learning_bindings[edge].program;
            items[position].slot = slot;
        }
        if (stop - start > 1U) {
            qsort(
                &items[start], (size_t)(stop - start),
                sizeof(lc_learning_sort_item), lc_learning_sort_before
            );
        }
        for (position = start; position < stop; ++position) {
            lc_learning_batch *batch;
            slots[position] = items[position].slot;
            if (position == start ||
                items[position].program != items[position - 1U].program) {
                batch = &batches[batch_count++];
                batch->slot_offset = position;
                batch->program = items[position].program;
            }
            batch = &batches[batch_count - 1U];
            batch->slot_count++;
        }
    }
    batch_offsets[scope_count] = batch_count;
    free(items);
    *batch_offsets_out = batch_offsets;
    *batches_out = batches;
    *batch_count_out = batch_count;
    return LC_OK;
}

/* Compile equation-derived learning programs and edge bindings. */
lc_status lc_mixed_graph_compile_learning(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    const lc_learning_program *learning_programs,
    uint32_t learning_program_count,
    const lc_real_t *learning_parameters,
    uint32_t learning_parameter_count,
    const lc_learning_binding *learning_bindings,
    uint32_t edge_count,
    lc_compiled_graph **compiled
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (compiled != NULL) *compiled = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_graph *result = NULL;
    uint64_t expression_count = 0U;
    uint64_t expression_cursor = 0U;
    uint32_t expected_parameter_offset = 0U;
    uint32_t program_index;
    uint32_t edge;
    lc_status status;
    if (compiled == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *compiled = NULL;
    if ((learning_program_count > 0U && learning_programs == NULL) ||
        (learning_parameter_count > 0U && learning_parameters == NULL) ||
        (edge_count > 0U && learning_bindings == NULL) ||
        lc_allocation_size_overflows(
            learning_program_count, sizeof(lc_learning_program)
        ) ||
        lc_allocation_size_overflows(learning_parameter_count, sizeof(lc_real_t)) ||
        lc_allocation_size_overflows(edge_count, sizeof(lc_learning_binding))) {
        return LC_INVALID_ARGUMENT;
    }
    for (program_index = 0U; program_index < learning_program_count;
         ++program_index) {
        uint32_t event_kind;
        if (!lc_learning_program_valid(&learning_programs[program_index])) {
            return LC_INVALID_ARGUMENT;
        }
        for (event_kind = 0U; event_kind < LC_LEARNING_EVENT_COUNT; ++event_kind) {
            uint32_t count = learning_programs[program_index].events[event_kind].node_count;
            if (expression_count > UINT64_MAX - count) {
                return LC_INVALID_ARGUMENT;
            }
            expression_count += count;
        }
        if (expression_count > UINT64_MAX -
                learning_programs[program_index].observer.node_count) {
            return LC_INVALID_ARGUMENT;
        }
        expression_count += learning_programs[program_index].observer.node_count;
    }
    if (lc_allocation_size_overflows(expression_count, sizeof(lc_expr_node))) {
        return LC_INVALID_ARGUMENT;
    }
    for (program_index = 0U; program_index < learning_parameter_count;
         ++program_index) {
        if (!lc_isfinite(learning_parameters[program_index])) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (edge = 0U; edge < edge_count; ++edge) {
        const lc_learning_binding *binding = &learning_bindings[edge];
        if (binding->program == UINT32_MAX) {
            if (binding->parameter_offset != expected_parameter_offset ||
                binding->modulator != UINT32_MAX || binding->weight_group != 0U) {
                return LC_INVALID_ARGUMENT;
            }
            continue;
        }
        if (binding->program >= learning_program_count ||
            binding->parameter_offset != expected_parameter_offset ||
            !lc_isfinite(binding->weight_min) || binding->weight_min < LC_REAL_C(0.0) ||
            !lc_isfinite(binding->weight_max) ||
            binding->weight_max <= binding->weight_min ||
            edges == NULL || edges[edge].weight < binding->weight_min ||
            edges[edge].weight > binding->weight_max ||
            binding->weight_group > edge_count) {
            return LC_INVALID_ARGUMENT;
        }
        {
            const lc_learning_program *program =
                &learning_programs[binding->program];
            uint32_t trace;
            int has_modulation =
                program->events[LC_LEARNING_MODULATION_POSITIVE].node_count > 0U ||
                program->events[LC_LEARNING_MODULATION_NEGATIVE].node_count > 0U;
            if (program->parameter_count > learning_parameter_count -
                    expected_parameter_offset ||
                (has_modulation &&
                 (binding->modulator == UINT32_MAX ||
                  binding->modulator >= edge_count)) ||
                (!has_modulation && binding->modulator != UINT32_MAX)) {
                return LC_INVALID_ARGUMENT;
            }
            for (trace = 0U; trace < program->trace_count; ++trace) {
                lc_real_t tau = learning_parameters[
                    expected_parameter_offset +
                    program->trace_tau_parameters[trace]
                ];
                if (!lc_isfinite(tau) || tau <= LC_REAL_C(0.0)) {
                    return LC_INVALID_ARGUMENT;
                }
            }
            if (program->observer.node_count > 0U) {
                const lc_learning_observer_program *observer =
                    &program->observer;
                lc_real_t tau_voltage = learning_parameters[
                    expected_parameter_offset +
                    observer->voltage_tau_parameter
                ];
                lc_real_t tau_fast = learning_parameters[
                    expected_parameter_offset +
                    observer->fast_activity_tau_parameter
                ];
                lc_real_t tau_slow = learning_parameters[
                    expected_parameter_offset +
                    observer->slow_activity_tau_parameter
                ];
                lc_real_t band_width = learning_parameters[
                    expected_parameter_offset +
                    observer->band_width_parameter
                ];
                const lc_mixed_node *post = &nodes[edges[edge].post];
                if (!(tau_voltage > LC_REAL_C(0.0)) || !(tau_fast > LC_REAL_C(0.0)) ||
                    !(tau_slow > tau_fast) || !(band_width > LC_REAL_C(0.0)) ||
                    !(
                        (post->arithmetic_kind ==
                            LC_NODE_ARITHMETIC_SCALAR_AFFINE &&
                         post->scalar_affine_decay < LC_REAL_C(0.0)) ||
                        post->crossing_kind == LC_CROSSING_TWO_REAL_EXP
                    ) ||
                    edges[edge].deposit_kind != LC_DEPOSIT_STATE_ADD ||
                    edges[edge].target != post->readout) {
                    return LC_INVALID_ARGUMENT;
                }
            }
            expected_parameter_offset += program->parameter_count;
        }
    }
    if (expected_parameter_offset != learning_parameter_count) {
        return LC_INVALID_ARGUMENT;
    }
    status = lc_mixed_graph_compile_layout(
        nodes, node_count, state_count, parameters, parameter_count,
        edges, NULL, learning_bindings, learning_programs,
        learning_program_count, learning_parameters, learning_parameter_count,
        edge_count, 1, &result
    );
    if (status != LC_OK) {
        return status;
    }
    if (result->plasticity != NULL) {
        lc_compiled_graph_release(result);
        return LC_INVALID_ARGUMENT;
    }
    if (learning_program_count > 0U) {
        result->learning_programs = calloc(
            learning_program_count, sizeof(lc_learning_program)
        );
    }
    if (expression_count > 0U) {
        result->learning_expression_nodes = calloc(
            (size_t)expression_count, sizeof(lc_expr_node)
        );
    }
    if (learning_parameter_count > 0U) {
        result->learning_parameters = calloc(
            learning_parameter_count, sizeof(lc_real_t)
        );
    }
    if (edge_count > 0U) {
        result->learning_bindings = calloc(
            edge_count, sizeof(lc_learning_binding)
        );
    }
    if (node_count > 0U) {
        result->learning_observer_program_by_node = malloc(
            node_count * sizeof(uint32_t)
        );
        result->learning_observer_parameter_offset_by_node = malloc(
            node_count * sizeof(uint32_t)
        );
    }
    if ((learning_program_count > 0U && result->learning_programs == NULL) ||
        (expression_count > 0U && result->learning_expression_nodes == NULL) ||
        (learning_parameter_count > 0U && result->learning_parameters == NULL) ||
        (edge_count > 0U && result->learning_bindings == NULL) ||
        (node_count > 0U &&
         (result->learning_observer_program_by_node == NULL ||
          result->learning_observer_parameter_offset_by_node == NULL))) {
        lc_compiled_graph_release(result);
        return LC_ALLOCATION_FAILED;
    }
    result->learning_program_count = learning_program_count;
    result->learning_parameter_count = learning_parameter_count;
    if (learning_parameter_count > 0U) {
        memcpy(
            result->learning_parameters, learning_parameters,
            learning_parameter_count * sizeof(lc_real_t)
        );
    }
    if (edge_count > 0U) {
        memcpy(
            result->learning_bindings, learning_bindings,
            edge_count * sizeof(lc_learning_binding)
        );
    }
    for (program_index = 0U; program_index < node_count; ++program_index) {
        result->learning_observer_program_by_node[program_index] = UINT32_MAX;
        result->learning_observer_parameter_offset_by_node[program_index] =
            UINT32_MAX;
    }
    for (program_index = 0U; program_index < learning_program_count;
         ++program_index) {
        uint32_t event_kind;
        result->learning_programs[program_index] = learning_programs[program_index];
        for (event_kind = 0U; event_kind < LC_LEARNING_EVENT_COUNT; ++event_kind) {
            const lc_learning_event_program *source =
                &learning_programs[program_index].events[event_kind];
            lc_learning_event_program *target =
                &result->learning_programs[program_index].events[event_kind];
            if (source->node_count > 0U) {
                target->nodes = &result->learning_expression_nodes[expression_cursor];
                memcpy(
                    (lc_expr_node *)target->nodes, source->nodes,
                    source->node_count * sizeof(lc_expr_node)
                );
                expression_cursor += source->node_count;
                if (source->node_count > result->workspace_count) {
                    result->workspace_count = source->node_count;
                }
            } else {
                target->nodes = NULL;
            }
        }
        {
            const lc_learning_observer_program *source =
                &learning_programs[program_index].observer;
            lc_learning_observer_program *target =
                &result->learning_programs[program_index].observer;
            if (source->node_count > 0U) {
                target->nodes =
                    &result->learning_expression_nodes[expression_cursor];
                memcpy(
                    (lc_expr_node *)target->nodes, source->nodes,
                    source->node_count * sizeof(lc_expr_node)
                );
                expression_cursor += source->node_count;
                if (source->node_count > result->workspace_count) {
                    result->workspace_count = source->node_count;
                }
            } else {
                target->nodes = NULL;
            }
        }
    }
    for (edge = 0U; edge < edge_count; ++edge) {
        const lc_learning_binding *binding = &result->learning_bindings[edge];
        const lc_learning_program *program;
        uint32_t post;
        uint32_t previous_program;
        if (binding->program == UINT32_MAX) {
            continue;
        }
        program = &result->learning_programs[binding->program];
        if (program->observer.node_count == 0U) {
            continue;
        }
        post = result->edges[edge].post;
        if (!(
                (result->nodes[post].arithmetic_kind ==
                    LC_NODE_ARITHMETIC_SCALAR_AFFINE &&
                 result->nodes[post].scalar_affine_decay < LC_REAL_C(0.0)) ||
                result->nodes[post].crossing_kind ==
                    LC_CROSSING_TWO_REAL_EXP
            )) {
            lc_compiled_graph_release(result);
            return LC_INVALID_ARGUMENT;
        }
        previous_program = result->learning_observer_program_by_node[post];
        if (previous_program == UINT32_MAX) {
            result->learning_observer_program_by_node[post] = binding->program;
            result->learning_observer_parameter_offset_by_node[post] =
                binding->parameter_offset;
        } else {
            uint32_t previous_offset =
                result->learning_observer_parameter_offset_by_node[post];
            const lc_learning_program *previous =
                &result->learning_programs[previous_program];
            uint32_t index;
            if (previous->observer.node_count == 0U ||
                previous->observer.node_count != program->observer.node_count ||
                previous->observer.variable_mask !=
                    program->observer.variable_mask ||
                previous->observer.parameter_mask !=
                    program->observer.parameter_mask ||
                previous->observer.voltage_tau_parameter !=
                    program->observer.voltage_tau_parameter ||
                previous->observer.fast_activity_tau_parameter !=
                    program->observer.fast_activity_tau_parameter ||
                previous->observer.slow_activity_tau_parameter !=
                    program->observer.slow_activity_tau_parameter ||
                previous->observer.band_width_parameter !=
                    program->observer.band_width_parameter ||
                previous->observer.fast_activity_root !=
                    program->observer.fast_activity_root ||
                previous->observer.slow_activity_root !=
                    program->observer.slow_activity_root ||
                previous->observer.gain_root !=
                    program->observer.gain_root ||
                memcmp(
                    previous->observer.nodes, program->observer.nodes,
                    program->observer.node_count * sizeof(lc_expr_node)
                ) != 0) {
                lc_compiled_graph_release(result);
                return LC_INVALID_ARGUMENT;
            }
            for (index = 0U; index < 32U; ++index) {
                if ((program->observer.parameter_mask &
                        (UINT32_C(1) << index)) != 0U &&
                    result->learning_parameters[previous_offset + index] !=
                        result->learning_parameters[
                            binding->parameter_offset + index
                        ]) {
                    lc_compiled_graph_release(result);
                    return LC_INVALID_ARGUMENT;
                }
            }
        }
    }
    for (edge = 0U; edge < edge_count; ++edge) {
        uint32_t post = result->edges[edge].post;
        const lc_mixed_node *descriptor = &result->nodes[post];
        if (result->learning_observer_program_by_node[post] != UINT32_MAX &&
            (result->edges[edge].deposit_kind != LC_DEPOSIT_STATE_ADD ||
             result->edges[edge].target != descriptor->readout)) {
            lc_compiled_graph_release(result);
            return LC_INVALID_ARGUMENT;
        }
    }
    if (result->plastic_edge_count > 0U) {
        status = lc_learning_build_scope_batches(
            result, result->node_count, result->incoming_plastic_offsets,
            result->incoming_plastic_slots,
            &result->incoming_learning_batch_offsets,
            &result->incoming_learning_batches,
            &result->incoming_learning_batch_count
        );
        if (status == LC_OK) {
            status = lc_learning_build_scope_batches(
                result, result->modulator_count, result->modulator_offsets,
                result->modulator_slots,
                &result->modulation_learning_batch_offsets,
                &result->modulation_learning_batches,
                &result->modulation_learning_batch_count
            );
        }
        if (status != LC_OK) {
            lc_compiled_graph_release(result);
            return status;
        }
    }
    *compiled = result;
    return LC_OK;
}

void lc_mixed_graph_destroy(lc_compiled_graph *compiled) {
    lc_compiled_graph_release(compiled);
}

/* Restore initial state and clear all run-local event and learning state. */
lc_status lc_mixed_run_reset(
    lc_mixed_run *run,
    const lc_real_t *initial_state,
    uint32_t state_count,
    const lc_time_t *t_last,
    uint32_t node_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_graph *graph;
    uint32_t node;
    uint32_t local;
    if (run == NULL || run->graph == NULL || initial_state == NULL || t_last == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    graph = run->graph;
    if (state_count != graph->state_count || node_count != graph->node_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0; node < graph->node_count; ++node) {
        const lc_mixed_node *descriptor = &graph->nodes[node];
        if (!lc_isfinite(t_last[node]) || t_last[node] < LC_REAL_C(0.0)) {
            return LC_INVALID_ARGUMENT;
        }
        for (local = 0; local < descriptor->state_count; ++local) {
            if (!lc_isfinite(initial_state[descriptor->state_offset + local])) {
                return LC_INVALID_ARGUMENT;
            }
        }
        if (descriptor->crossing_kind != LC_CROSSING_REACTIVE &&
            descriptor->crossing_kind != LC_CROSSING_INTEGRATED_HAZARD &&
            initial_state[descriptor->state_offset + descriptor->readout] >=
                descriptor->threshold) {
            return LC_INVALID_ARGUMENT;
        }
    }
    memcpy(run->state, initial_state, graph->state_count * sizeof(lc_real_t));
    memcpy(run->t_last, t_last, graph->node_count * sizeof(lc_time_t));
    memcpy(run->active_nodes, graph->nodes, graph->node_count * sizeof(lc_mixed_node));
    if (graph->parameter_count > 0U) {
        memcpy(
            run->active_parameters, graph->parameters,
            graph->parameter_count * sizeof(lc_real_t)
        );
    }
    if (graph->plastic_edge_count > 0U) {
        uint32_t slot;
        memset(
            run->plasticity, 0,
            graph->plastic_edge_count * sizeof(lc_plasticity_state)
        );
        for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
            uint32_t edge = graph->plastic_edges[slot];
            lc_plasticity_state *state = &run->plasticity[slot];
            state->edge = edge;
            state->kind = graph->learning_program_count > 0U
                ? graph->learning_programs[
                      graph->learning_bindings[edge].program
                  ].compatibility_kind
                : graph->plasticity[edge].kind;
            state->weight = graph->edges[edge].weight;
            state->t_pre_fast = t_last[0];
            state->t_post_fast = t_last[0];
            state->t_pre_slow = t_last[0];
            state->t_post_slow = t_last[0];
            state->t_eligibility_plus = t_last[0];
            state->t_eligibility_minus = t_last[0];
        }
        if (run->learning_weight_deltas != NULL) {
            memset(
                run->learning_weight_deltas, 0,
                graph->plastic_edge_count * sizeof(lc_real_t)
            );
            memset(
                run->learning_weight_touched, 0,
                graph->plastic_edge_count * sizeof(uint8_t)
            );
            run->learning_weight_master_count = 0U;
        }
    }
    if (run->learning_observers != NULL) {
        memset(
            run->learning_observers, 0,
            graph->node_count * sizeof(lc_learning_observer_state)
        );
        for (node = 0U; node < graph->node_count; ++node) {
            const lc_mixed_node *descriptor = &graph->nodes[node];
            run->learning_observers[node].slow_voltage = initial_state[
                descriptor->state_offset + descriptor->readout
            ];
            run->learning_observers[node].t_activity = t_last[node];
        }
    }
    lc_runtime_reset_caches(run->runtime, graph->node_count);
    memset(run->state_deposits, 0, graph->state_count * sizeof(lc_real_t));
    memset(run->program_deposits, 0, graph->node_count * sizeof(lc_real_t));
    memset(run->affected, 0, graph->node_count * sizeof(uint8_t));
    memset(run->deposit_affected, 0, graph->node_count * sizeof(uint8_t));
    memset(run->timestamp_reset, 0, graph->node_count * sizeof(uint8_t));
    memset(run->fired, 0, graph->node_count * sizeof(uint8_t));
    free(run->pending_modulations);
    run->pending_modulations = NULL;
    run->pending_modulation_count = 0U;
    run->heap.size = 0U;
    run->heap.logical_size = 0U;
    run->heap.next_seq = 0U;
    run->heap.peak = 0U;
    memset(&run->incremental_config, 0, sizeof(run->incremental_config));
    run->frontier = t_last[0];
    run->trace_sequence = 0U;
    run->next_input_subject = 0U;
    run->incremental_active = 0;
    run->incremental_failed = 0;
    run->ready = 1;
    return LC_OK;
}

/* Allocate one independent mutable execution for a compiled graph. */
lc_status lc_mixed_run_create(
    lc_compiled_graph *compiled,
    const lc_real_t *initial_state,
    uint32_t state_count,
    const lc_time_t *t_last,
    uint32_t node_count,
    lc_mixed_run **run
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        if (run != NULL) *run = NULL;
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_mixed_run *result;
    lc_status status;
    if (run == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    *run = NULL;
    if (compiled == NULL || compiled->references == UINT64_MAX ||
        lc_allocation_size_overflows(compiled->state_count, sizeof(lc_real_t)) ||
        lc_allocation_size_overflows(compiled->node_count, sizeof(lc_time_t)) ||
        lc_allocation_size_overflows(compiled->node_count, sizeof(lc_mixed_node)) ||
        lc_allocation_size_overflows(compiled->node_count, sizeof(lc_node_runtime)) ||
        lc_allocation_size_overflows(
            compiled->node_count, sizeof(lc_learning_observer_state)
        ) ||
        lc_allocation_size_overflows(
            compiled->plastic_edge_count, sizeof(lc_plasticity_state)
        ) ||
        lc_allocation_size_overflows(
            compiled->plastic_edge_count, sizeof(lc_real_t)
        ) ||
        lc_allocation_size_overflows(
            compiled->plastic_edge_count, sizeof(uint8_t)
        ) ||
        lc_allocation_size_overflows(
            compiled->plastic_edge_count, sizeof(uint32_t)
        ) ||
        lc_allocation_size_overflows(compiled->workspace_count, sizeof(lc_real_t))) {
        return LC_INVALID_ARGUMENT;
    }
    result = calloc(1U, sizeof(lc_mixed_run));
    if (result == NULL) {
        return LC_ALLOCATION_FAILED;
    }
    result->graph = compiled;
    compiled->references++;
    result->state = calloc(compiled->state_count, sizeof(lc_real_t));
    result->t_last = calloc(compiled->node_count, sizeof(lc_time_t));
    result->active_nodes = calloc(compiled->node_count, sizeof(lc_mixed_node));
    result->runtime = calloc(compiled->node_count, sizeof(lc_node_runtime));
    result->learning_observers = calloc(
        compiled->node_count, sizeof(lc_learning_observer_state)
    );
    result->state_deposits = calloc(compiled->state_count, sizeof(lc_real_t));
    result->program_deposits = calloc(compiled->node_count, sizeof(lc_real_t));
    result->affected = calloc(compiled->node_count, sizeof(uint8_t));
    result->deposit_affected = calloc(compiled->node_count, sizeof(uint8_t));
    result->timestamp_reset = calloc(compiled->node_count, sizeof(uint8_t));
    result->fired = calloc(compiled->node_count, sizeof(uint8_t));
    result->affected_nodes = calloc(compiled->node_count, sizeof(uint32_t));
    result->fired_nodes = calloc(compiled->node_count, sizeof(uint32_t));
    result->timestamp_reset_nodes = calloc(
        compiled->node_count, sizeof(uint32_t)
    );
    result->workspace = calloc(compiled->workspace_count, sizeof(lc_real_t));
    if (compiled->plastic_edge_count > 0U) {
        result->plasticity = calloc(
            compiled->plastic_edge_count, sizeof(lc_plasticity_state)
        );
        if (compiled->learning_program_count > 0U &&
            compiled->modulator_count > 0U) {
            result->learning_weight_deltas = calloc(
                compiled->plastic_edge_count, sizeof(lc_real_t)
            );
            result->learning_weight_touched = calloc(
                compiled->plastic_edge_count, sizeof(uint8_t)
            );
            result->learning_weight_masters = calloc(
                compiled->plastic_edge_count, sizeof(uint32_t)
            );
        }
    }
    if (compiled->parameter_count > 0U) {
        result->active_parameters = calloc(compiled->parameter_count, sizeof(lc_real_t));
    }
    if (result->state == NULL || result->t_last == NULL ||
        result->active_nodes == NULL || result->runtime == NULL ||
        result->learning_observers == NULL ||
        result->state_deposits == NULL || result->program_deposits == NULL ||
        result->affected == NULL || result->deposit_affected == NULL ||
        result->timestamp_reset == NULL ||
        result->fired == NULL || result->affected_nodes == NULL ||
        result->fired_nodes == NULL || result->timestamp_reset_nodes == NULL ||
        result->workspace == NULL ||
        (compiled->plastic_edge_count > 0U && result->plasticity == NULL) ||
        (compiled->plastic_edge_count > 0U &&
         compiled->learning_program_count > 0U &&
         compiled->modulator_count > 0U &&
         (result->learning_weight_deltas == NULL ||
          result->learning_weight_touched == NULL ||
          result->learning_weight_masters == NULL)) ||
        (compiled->parameter_count > 0U && result->active_parameters == NULL)) {
        lc_mixed_run_destroy(result);
        return LC_ALLOCATION_FAILED;
    }
    status = lc_mixed_run_reset(
        result, initial_state, state_count, t_last, node_count
    );
    if (status != LC_OK) {
        lc_mixed_run_destroy(result);
        return status;
    }
    *run = result;
    return LC_OK;
}

/* Initialize a resumable event frontier with a fixed final horizon. */
lc_status lc_mixed_run_begin_incremental(
    lc_mixed_run *run,
    const lc_run_config *config,
    lc_network_error *error
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_graph *graph;
    lc_event *resized;
    uint32_t node;
    lc_status status;
    if (run == NULL || run->graph == NULL || !run->ready || config == NULL ||
        error == NULL || config->queue_capacity == 0U ||
        config->same_time_cascade_limit == 0U || !lc_isfinite(config->t_end) ||
        config->t_end < LC_REAL_C(0.0) ||
        lc_allocation_size_overflows(config->queue_capacity, sizeof(lc_event))) {
        return LC_INVALID_ARGUMENT;
    }
    graph = run->graph;
    for (node = 0U; node < graph->node_count; ++node) {
        if (run->t_last[node] != run->t_last[0] ||
            run->t_last[node] > config->t_end) {
            return LC_INVALID_ARGUMENT;
        }
    }
    if (config->queue_capacity > run->heap_storage_capacity) {
        resized = realloc(
            run->heap.items,
            (size_t)config->queue_capacity * sizeof(lc_event)
        );
        if (resized == NULL) {
            return LC_ALLOCATION_FAILED;
        }
        run->heap.items = resized;
        run->heap_storage_capacity = config->queue_capacity;
    }
    memset(error, 0, sizeof(*error));
    lc_runtime_reset_caches(run->runtime, graph->node_count);
    run->heap.size = 0U;
    run->heap.logical_size = 0U;
    run->heap.capacity = config->queue_capacity;
    run->heap.next_seq = 0U;
    run->heap.peak = 0U;
    run->incremental_config = *config;
    run->frontier = run->t_last[0];
    run->trace_sequence = 0U;
    run->next_input_subject = 0U;
    run->incremental_active = 1;
    run->incremental_failed = 0;
    run->ready = 0;
    for (node = 0U; node < graph->node_count; ++node) {
        run->runtime[node].generation = 1U;
        status = lc_mixed_schedule_prediction_planned(
            run->active_nodes, run->state, run->t_last,
            run->active_parameters, run->runtime, node,
            &run->incremental_config, &run->heap, error, run->variables,
            run->workspace, graph->workspace_count, graph->node_eval_plans
        );
        if (status != LC_OK) {
            run->incremental_active = 0;
            run->incremental_failed = 1;
            return status;
        }
    }
    return LC_OK;
}

/* Reset neuronal episode state while retaining learned weights. */
lc_status lc_mixed_run_reset_episode(
    lc_mixed_run *run,
    const lc_real_t *initial_state,
    uint32_t state_count,
    lc_network_error *error
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_graph *graph;
    uint32_t node;
    uint32_t local;
    lc_status status;
    if (run == NULL || run->graph == NULL || initial_state == NULL ||
        error == NULL || !run->incremental_active || run->incremental_failed) {
        return LC_INVALID_ARGUMENT;
    }
    graph = run->graph;
    if (state_count != graph->state_count || !lc_isfinite(run->frontier) ||
        run->frontier > run->incremental_config.t_end) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0U; node < graph->node_count; ++node) {
        const lc_mixed_node *descriptor = &graph->nodes[node];
        for (local = 0U; local < descriptor->state_count; ++local) {
            if (!lc_isfinite(initial_state[descriptor->state_offset + local])) {
                return LC_INVALID_ARGUMENT;
            }
        }
        if (descriptor->crossing_kind != LC_CROSSING_REACTIVE &&
            descriptor->crossing_kind != LC_CROSSING_INTEGRATED_HAZARD &&
            initial_state[descriptor->state_offset + descriptor->readout] >=
                descriptor->threshold) {
            return LC_INVALID_ARGUMENT;
        }
    }
    memset(error, 0, sizeof(*error));
    memcpy(run->state, initial_state, graph->state_count * sizeof(lc_real_t));
    for (node = 0U; node < graph->node_count; ++node) {
        run->t_last[node] = run->frontier;
    }
    memcpy(
        run->active_nodes, graph->nodes,
        graph->node_count * sizeof(lc_mixed_node)
    );
    if (graph->parameter_count > 0U) {
        memcpy(
            run->active_parameters, graph->parameters,
            graph->parameter_count * sizeof(lc_real_t)
        );
    }
    if (graph->plastic_edge_count > 0U) {
        uint32_t slot;
        for (slot = 0U; slot < graph->plastic_edge_count; ++slot) {
            lc_plasticity_state *plasticity = &run->plasticity[slot];
            uint32_t edge = plasticity->edge;
            uint32_t kind = plasticity->kind;
            lc_real_t weight = plasticity->weight;
            memset(plasticity, 0, sizeof(*plasticity));
            plasticity->edge = edge;
            plasticity->kind = kind;
            plasticity->weight = weight;
            plasticity->t_pre_fast = run->frontier;
            plasticity->t_post_fast = run->frontier;
            plasticity->t_pre_slow = run->frontier;
            plasticity->t_post_slow = run->frontier;
            plasticity->t_eligibility_plus = run->frontier;
            plasticity->t_eligibility_minus = run->frontier;
        }
        if (run->learning_weight_deltas != NULL) {
            memset(
                run->learning_weight_deltas, 0,
                graph->plastic_edge_count * sizeof(lc_real_t)
            );
            memset(
                run->learning_weight_touched, 0,
                graph->plastic_edge_count * sizeof(uint8_t)
            );
            run->learning_weight_master_count = 0U;
        }
    }
    if (run->learning_observers != NULL) {
        memset(
            run->learning_observers, 0,
            graph->node_count * sizeof(lc_learning_observer_state)
        );
        for (node = 0U; node < graph->node_count; ++node) {
            const lc_mixed_node *descriptor = &graph->nodes[node];
            run->learning_observers[node].slow_voltage = initial_state[
                descriptor->state_offset + descriptor->readout
            ];
            run->learning_observers[node].t_activity = run->frontier;
        }
    }
    for (node = 0U; node < graph->node_count; ++node) {
        uint64_t hazard_draw_index = run->runtime[node].hazard_draw_index;
        lc_step_cache *cache = run->runtime[node].step_cache;
        lc_step_cache_reset(cache);
        memset(&run->runtime[node], 0, sizeof(lc_node_runtime));
        run->runtime[node].step_cache = cache;
        run->runtime[node].generation = 1U;
        run->runtime[node].hazard_draw_index = hazard_draw_index;
    }
    memset(run->state_deposits, 0, graph->state_count * sizeof(lc_real_t));
    memset(run->program_deposits, 0, graph->node_count * sizeof(lc_real_t));
    memset(run->affected, 0, graph->node_count * sizeof(uint8_t));
    memset(run->deposit_affected, 0, graph->node_count * sizeof(uint8_t));
    memset(run->timestamp_reset, 0, graph->node_count * sizeof(uint8_t));
    memset(run->fired, 0, graph->node_count * sizeof(uint8_t));
    free(run->pending_modulations);
    run->pending_modulations = NULL;
    run->pending_modulation_count = 0U;
    run->heap.size = 0U;
    run->heap.logical_size = 0U;
    run->heap.next_seq = 0U;
    run->heap.peak = 0U;
    run->next_input_subject = 0U;
    for (node = 0U; node < graph->node_count; ++node) {
        status = lc_mixed_schedule_prediction_planned(
            run->active_nodes, run->state, run->t_last,
            run->active_parameters, run->runtime, node,
            &run->incremental_config, &run->heap, error, run->variables,
            run->workspace, graph->workspace_count, graph->node_eval_plans
        );
        if (status != LC_OK) {
            run->incremental_active = 0;
            run->incremental_failed = 1;
            return status;
        }
    }
    return LC_OK;
}

/* Incremental runs retain the queue, state, learning traces, and decoders. */
static lc_status lc_mixed_run_execute_internal(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders,
    lc_trace_config *trace,
    lc_state_inspection_config *inspections,
    int resume,
    lc_time_t until,
    int seal
) {
    lc_compiled_graph *graph;
    lc_heap *heap;
    lc_status status = LC_OK;
    uint32_t node;
    uint32_t input;
    uint32_t input_cursor = 0U;
    uint32_t drive_update;
    uint32_t inspection_cursor = 0U;
    uint64_t input_subject_base;
    int cursor_inputs = 1;
    int static_add_deliveries;
    lc_event *resized;
    lc_profile_t kernel_start;

    if (run == NULL || run->graph == NULL || config == NULL ||
        (!resume && !run->ready) ||
        (resume && (!run->incremental_active || run->incremental_failed)) ||
        output_count == NULL || stats == NULL || error == NULL ||
        config->queue_capacity == 0U || config->same_time_cascade_limit == 0U ||
        !lc_isfinite(config->t_end) || config->t_end < LC_REAL_C(0.0) ||
        (input_count > 0U && inputs == NULL) ||
        (drive_update_count > 0U && drive_updates == NULL) ||
        (config->output_capacity > 0U && outputs == NULL) ||
        lc_allocation_size_overflows(config->queue_capacity, sizeof(lc_event)) ||
        !lc_isfinite(until) || until < LC_REAL_C(0.0) || until > config->t_end ||
        (resume && until < run->frontier) ||
        (resume && seal && until != config->t_end) ||
        (!resume && (until != config->t_end || !seal))) {
        return LC_INVALID_ARGUMENT;
    }
    graph = run->graph;
    static_add_deliveries =
        graph->plastic_edge_count == 0U && trace == NULL;
    memset(error, 0, sizeof(*error));
    if (!lc_trace_config_valid(trace, graph->node_count)) {
        return LC_INVALID_ARGUMENT;
    }
    status = lc_inspection_config_status(
        inspections, graph->node_count, run->t_last, until, seal
    );
    if (status != LC_OK) {
        if (status == LC_INSPECTION_OVERFLOW && inspections != NULL) {
            uint32_t request_index = (uint32_t)inspections->capacity;
            const lc_state_inspection_request *request =
                &inspections->requests[request_index];
            lc_set_resource_error(
                error, LC_RESOURCE_INSPECTION, inspections->capacity,
                inspections->capacity, inspections->capacity, NULL
            );
            error->has_event = 1U;
            error->event_index = request_index;
            error->node = request->node;
            error->t = request->t;
        }
        return status;
    }
    if (trace != NULL) {
        *trace->count = 0U;
        trace->sequence_base = resume ? run->trace_sequence : 0U;
    }
    if (inspections != NULL) {
        *inspections->count = 0U;
    }
    if (decoders != NULL &&
        lc_decoder_run_validate_for_execution(
            decoders, graph->node_count, config->t_end
        ) != LC_OK) {
        return LC_INVALID_ARGUMENT;
    }
    heap = &run->heap;
    for (node = 0; node < graph->node_count; ++node) {
        if (run->t_last[node] > until) {
            return LC_INVALID_ARGUMENT;
        }
    }
    for (input = 0; input < input_count; ++input) {
        const lc_mixed_node *target;
        if (inputs[input].node >= graph->node_count || !lc_isfinite(inputs[input].t) ||
            !lc_isfinite(inputs[input].value) ||
            inputs[input].t < run->t_last[inputs[input].node] ||
            (resume && inputs[input].t < run->frontier) ||
            (resume && (inputs[input].t > until ||
                        (!seal && inputs[input].t == until)))) {
            return LC_INVALID_ARGUMENT;
        }
        target = &graph->nodes[inputs[input].node];
        if (!lc_mixed_deposit_valid(
                target, inputs[input].deposit_kind, inputs[input].target
            )) {
            return LC_INVALID_ARGUMENT;
        }
        if (input > 0U && inputs[input].t < inputs[input - 1U].t) {
            cursor_inputs = 0;
        }
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        const lc_mixed_node *target;
        if (drive_updates[drive_update].node >= graph->node_count ||
            !lc_isfinite(drive_updates[drive_update].t) ||
            !lc_isfinite(drive_updates[drive_update].value) ||
            drive_updates[drive_update].t <
                run->t_last[drive_updates[drive_update].node] ||
            (resume && drive_updates[drive_update].t < run->frontier) ||
            (resume && (drive_updates[drive_update].t > until ||
                        (!seal && drive_updates[drive_update].t == until)))) {
            return LC_INVALID_ARGUMENT;
        }
        target = &graph->nodes[drive_updates[drive_update].node];
        if (drive_updates[drive_update].binding >= target->parameter_count) {
            return LC_INVALID_ARGUMENT;
        }
    }
    if (resume &&
        (run->next_input_subject > UINT32_MAX ||
         input_count > UINT32_MAX - run->next_input_subject)) {
        return LC_NUMERIC_ERROR;
    }
    if (resume &&
        (uint64_t)(cursor_inputs ? 0U : input_count) + drive_update_count >
            heap->capacity - heap->logical_size) {
        uint64_t available = heap->capacity - heap->logical_size;
        uint64_t queued_inputs = cursor_inputs ? 0U : input_count;
        lc_event overflowed;
        memset(&overflowed, 0, sizeof(overflowed));
        if (available < queued_inputs) {
            overflowed.t = inputs[available].t;
            overflowed.phase = LC_PHASE_DEPOSIT;
            overflowed.kind = LC_EVENT_INPUT_SPIKE;
            overflowed.index = inputs[available].node;
            overflowed.subject = (uint32_t)(run->next_input_subject + available);
        } else {
            uint64_t drive_index = available - queued_inputs;
            overflowed.t = drive_updates[drive_index].t;
            overflowed.phase = LC_PHASE_BOUNDARY;
            overflowed.kind = LC_EVENT_DRIVE_UPDATE;
            overflowed.index = drive_updates[drive_index].node;
            overflowed.subject = (uint32_t)drive_index;
        }
        lc_set_resource_error(
            error, LC_RESOURCE_QUEUE, heap->capacity, heap->capacity,
            heap->peak > heap->capacity ? heap->peak : heap->capacity,
            &overflowed
        );
        return LC_QUEUE_OVERFLOW;
    }
    if (!resume && config->queue_capacity > run->heap_storage_capacity) {
        resized = realloc(heap->items, (size_t)config->queue_capacity * sizeof(lc_event));
        if (resized == NULL) {
            return LC_ALLOCATION_FAILED;
        }
        heap->items = resized;
        run->heap_storage_capacity = config->queue_capacity;
    }

    *output_count = 0U;
    input_subject_base = resume ? run->next_input_subject : 0U;
    memset(stats, 0, sizeof(*stats));
    kernel_start = lc_monotonic_seconds();
    if (!resume) {
        run->ready = 0;
        heap->size = 0U;
        heap->logical_size = 0U;
        heap->capacity = config->queue_capacity;
        heap->next_seq = 0U;
        heap->peak = 0U;
        lc_runtime_reset_caches(run->runtime, graph->node_count);
        for (node = 0; node < graph->node_count; ++node) {
            run->runtime[node].generation = 1U;
            status = lc_mixed_schedule_prediction_planned(
                run->active_nodes, run->state, run->t_last,
                run->active_parameters, run->runtime, node, config, heap,
                error, run->variables, run->workspace,
                graph->workspace_count, graph->node_eval_plans
            );
            if (status != LC_OK) {
                goto compiled_cleanup;
            }
        }
        for (input = 0U; input < run->pending_modulation_count; ++input) {
            lc_event event;
            const lc_modulation_event *source = &run->pending_modulations[input];
            if (source->t > config->t_end) {
                continue;
            }
            memset(&event, 0, sizeof(event));
            event.t = source->t;
            event.phase = LC_PHASE_BOUNDARY;
            event.kind = LC_EVENT_MODULATION;
            event.index = source->modulator;
            event.subject = input;
            event.value = source->value;
            status = lc_heap_push_report(heap, event, error);
            if (status != LC_OK) {
                goto compiled_cleanup;
            }
        }
        free(run->pending_modulations);
        run->pending_modulations = NULL;
        run->pending_modulation_count = 0U;
    }
    if (!cursor_inputs) {
        for (input = 0; input < input_count; ++input) {
            lc_event event;
            if (inputs[input].t > config->t_end) {
                continue;
            }
            memset(&event, 0, sizeof(event));
            event.t = inputs[input].t;
            event.phase = LC_PHASE_DEPOSIT;
            event.kind = LC_EVENT_INPUT_SPIKE;
            event.index = inputs[input].node;
            event.subject = (uint32_t)(input_subject_base + input);
            event.target = inputs[input].target;
            event.auxiliary = inputs[input].deposit_kind;
            event.value = inputs[input].value;
            status = lc_heap_push_report(heap, event, error);
            if (status != LC_OK) {
                goto compiled_cleanup;
            }
        }
        input_cursor = input_count;
    }
    for (drive_update = 0; drive_update < drive_update_count; ++drive_update) {
        lc_event event;
        if (drive_updates[drive_update].t > config->t_end) {
            continue;
        }
        memset(&event, 0, sizeof(event));
        event.t = drive_updates[drive_update].t;
        event.phase = LC_PHASE_BOUNDARY;
        event.kind = LC_EVENT_DRIVE_UPDATE;
        event.index = drive_updates[drive_update].node;
        event.subject = drive_update;
        event.auxiliary = drive_updates[drive_update].binding;
        event.value = drive_updates[drive_update].value;
        status = lc_heap_push_report(heap, event, error);
        if (status != LC_OK) {
            goto compiled_cleanup;
        }
    }
    if (resume) {
        run->next_input_subject += input_count;
    }

    for (;;) {
        int heap_ready = heap->size > 0U &&
            (heap->items[0].t < until ||
             (seal && heap->items[0].t == until));
        int input_ready = cursor_inputs && input_cursor < input_count &&
            (inputs[input_cursor].t < until ||
             (seal && inputs[input_cursor].t == until));
        lc_time_t t;
        if (!heap_ready && !input_ready) {
            break;
        }
        if (input_ready &&
            (!heap_ready || inputs[input_cursor].t < heap->items[0].t)) {
            t = inputs[input_cursor].t;
        } else {
            t = heap->items[0].t;
        }
        uint32_t cascade_depth = 0U;
        uint32_t affected_count = 0U;
        uint32_t fired_count = 0U;
        uint32_t timestamp_reset_count = 0U;
        status = lc_mixed_capture_inspections_before(
            run, inspections, &inspection_cursor, t, error
        );
        if (status != LC_OK) {
            goto compiled_cleanup;
        }
        if (decoders != NULL) {
            status = lc_decoder_run_advance(decoders, t);
            if (status != LC_OK) {
                if (status == LC_DECODER_OUTPUT_OVERFLOW) {
                    lc_event event;
                    memset(&event, 0, sizeof(event));
                    event.t = t;
                    event.phase = LC_PHASE_BOUNDARY;
                    event.kind = LC_EVENT_DECODER_EVENT;
                    event.index = UINT32_MAX;
                    lc_set_decoder_resource_error(error, decoders, &event);
                }
                goto compiled_cleanup;
            }
        }
        while (lc_at(heap, t, LC_PHASE_BOUNDARY)) {
            lc_event event = lc_heap_pop(heap);
            stats->events_popped++;
            if (event.kind == LC_EVENT_DRIVE_UPDATE) {
                const lc_mixed_node *descriptor = &run->active_nodes[event.index];
                lc_real_t before[LC_ANALYTICAL_MAX_STATES];
                int capture = lc_trace_wants_state(
                    trace, LC_TRACE_DRIVE_UPDATE, event.index
                );
                stats->drive_updates_processed++;
                status = lc_mixed_advance_node_planned(
                    run->active_nodes, run->state, run->t_last,
                    run->active_parameters, run->runtime, event.index, t,
                    run->variables, run->workspace, graph->workspace_count,
                    graph->node_eval_plans, graph, run->learning_observers
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                if (capture) {
                    lc_trace_copy_state(
                        before, &run->state[descriptor->state_offset],
                        descriptor->state_count
                    );
                }
                run->active_parameters[
                    descriptor->parameter_offset + event.auxiliary
                ] = event.value;
                run->runtime[event.index].affine_disabled = 1;
                status = lc_trace_emit(
                    trace, LC_TRACE_DRIVE_UPDATE, LC_TRACE_PHASE_BOUNDARY,
                    t, event.index, event.auxiliary,
                    run->runtime[event.index].generation, event.value,
                    capture ? before : NULL,
                    capture ? &run->state[descriptor->state_offset] : NULL,
                    descriptor->state_count, error
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                lc_sparse_node_add(
                    run->affected, run->affected_nodes, &affected_count,
                    event.index
                );
            } else if (event.kind == LC_EVENT_MODULATION) {
                uint64_t position = graph->modulator_offsets[event.index];
                uint32_t slot = graph->modulator_slots[position];
                uint32_t plastic_edge = graph->plastic_edges[slot];
                uint32_t representative_node = graph->edges[plastic_edge].post;
                status = lc_plastic_modulate(
                    graph, run->plasticity, event.index, event.value, t,
                    run->variables, run->workspace, graph->workspace_count,
                    run->learning_weight_deltas,
                    run->learning_weight_touched,
                    run->learning_weight_masters,
                    &run->learning_weight_master_count
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                status = lc_trace_emit(
                    trace, LC_TRACE_MODULATION, LC_TRACE_PHASE_BOUNDARY, t,
                    representative_node, event.index,
                    run->runtime[representative_node].generation, event.value,
                    NULL, NULL, 0U, error
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
            } else if (event.kind == LC_EVENT_REFRACTORY_RELEASE &&
                event.generation == run->runtime[event.index].refractory_generation &&
                run->runtime[event.index].clamped) {
                const lc_mixed_node *descriptor = &run->active_nodes[event.index];
                lc_real_t before[LC_ANALYTICAL_MAX_STATES];
                int capture = lc_trace_wants_state(
                    trace, LC_TRACE_REFRACTORY_RELEASE, event.index
                );
                stats->refractory_releases_processed++;
                status = lc_mixed_advance_node_planned(
                    run->active_nodes, run->state, run->t_last,
                    run->active_parameters, run->runtime, event.index, t,
                    run->variables, run->workspace, graph->workspace_count,
                    graph->node_eval_plans, graph, run->learning_observers
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                if (capture) {
                    lc_trace_copy_state(
                        before, &run->state[descriptor->state_offset],
                        descriptor->state_count
                    );
                }
                run->runtime[event.index].clamped = 0;
                status = lc_trace_emit(
                    trace, LC_TRACE_REFRACTORY_RELEASE,
                    LC_TRACE_PHASE_BOUNDARY, t, event.index, UINT32_MAX,
                    event.generation, LC_REAL_C(0.0), capture ? before : NULL,
                    capture ? &run->state[descriptor->state_offset] : NULL,
                    descriptor->state_count, error
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                lc_sparse_node_add(
                    run->affected, run->affected_nodes, &affected_count,
                    event.index
                );
            }
        }
        if (graph->learning_program_count > 0U &&
            run->learning_weight_master_count > 0U) {
            status = lc_learning_commit_modulation_weights(
                graph, run->plasticity, run->learning_weight_deltas,
                run->learning_weight_touched, run->learning_weight_masters,
                &run->learning_weight_master_count
            );
            if (status != LC_OK) {
                goto compiled_cleanup;
            }
        }

        for (;;) {
            fired_count = 0U;
            while ((cursor_inputs && input_cursor < input_count &&
                    inputs[input_cursor].t == t) ||
                   lc_at(heap, t, LC_PHASE_DEPOSIT)) {
                lc_event event;
                if (cursor_inputs && input_cursor < input_count &&
                    inputs[input_cursor].t == t) {
                    const lc_mixed_input_spike *source = &inputs[input_cursor];
                    memset(&event, 0, sizeof(event));
                    event.t = source->t;
                    event.phase = LC_PHASE_DEPOSIT;
                    event.kind = LC_EVENT_INPUT_SPIKE;
                    event.index = source->node;
                    event.subject =
                        (uint32_t)(input_subject_base + input_cursor);
                    event.target = source->target;
                    event.auxiliary = source->deposit_kind;
                    event.value = source->value;
                    input_cursor++;
                } else {
                    event = lc_heap_pop(heap);
                }
                if (event.kind == LC_EVENT_DELIVERY) {
                    if (event.subject == 0U) {
                        stats->events_popped++;
                        stats->deliveries_processed++;
                        if (static_add_deliveries &&
                            graph->edges[event.index].deposit_kind ==
                                LC_DEPOSIT_STATE_ADD) {
                            status = lc_mixed_accumulate_static_add(
                                graph, run->active_nodes, event.index,
                                run->state_deposits, run->affected,
                                run->deposit_affected, run->affected_nodes,
                                &affected_count
                            );
                        } else {
                            status = lc_mixed_process_delivery(
                                graph, run->plasticity,
                                run->learning_observers, run->active_nodes,
                                run->state, run->t_last,
                                run->active_parameters, run->runtime,
                                event.index, t,
                                event.generation, trace, error,
                                run->state_deposits, run->program_deposits,
                                run->affected, run->deposit_affected,
                                run->affected_nodes, &affected_count,
                                run->variables, run->workspace,
                                graph->workspace_count
                            );
                        }
                        if (status != LC_OK) {
                            goto compiled_cleanup;
                        }
                    } else {
                        const lc_delivery_group *group;
                        uint64_t position;
                        if (event.index >= graph->delivery_group_count) {
                            status = LC_INVALID_ARGUMENT;
                            goto compiled_cleanup;
                        }
                        group = &graph->delivery_groups[event.index];
                        stats->events_popped += group->edge_count;
                        stats->deliveries_processed += group->edge_count;
                        for (position = group->edge_offset;
                             position < group->edge_offset + group->edge_count;
                             ++position) {
                            uint32_t edge =
                                graph->delivery_group_edges[position];
                            if (static_add_deliveries &&
                                graph->edges[edge].deposit_kind ==
                                    LC_DEPOSIT_STATE_ADD) {
                                status = lc_mixed_accumulate_static_add(
                                    graph, run->active_nodes, edge,
                                    run->state_deposits, run->affected,
                                    run->deposit_affected,
                                    run->affected_nodes, &affected_count
                                );
                            } else {
                                status = lc_mixed_process_delivery(
                                    graph, run->plasticity,
                                    run->learning_observers,
                                    run->active_nodes, run->state,
                                    run->t_last, run->active_parameters,
                                    run->runtime, edge, t,
                                    event.generation, trace, error,
                                    run->state_deposits,
                                    run->program_deposits, run->affected,
                                    run->deposit_affected,
                                    run->affected_nodes, &affected_count,
                                    run->variables, run->workspace,
                                    graph->workspace_count
                                );
                            }
                            if (status != LC_OK) {
                                goto compiled_cleanup;
                            }
                        }
                    }
                } else {
                    uint32_t target_node = event.index;
                    uint32_t deposit_kind = event.auxiliary;
                    uint32_t target = event.target;
                    lc_real_t value = event.value;
                    const lc_mixed_node *descriptor =
                        &run->active_nodes[target_node];
                    stats->events_popped++;
                    stats->input_spikes_processed++;
                    status = lc_trace_emit(
                        trace, LC_TRACE_INPUT_SPIKE, LC_TRACE_PHASE_DEPOSIT, t,
                        target_node, event.subject, event.generation, value,
                        NULL, NULL, 0U, error
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    if (deposit_kind == LC_DEPOSIT_PROGRAM) {
                        run->program_deposits[target_node] += value;
                        if (!lc_isfinite(run->program_deposits[target_node])) {
                            status = LC_NUMERIC_ERROR;
                            goto compiled_cleanup;
                        }
                    } else {
                        uint32_t state_index = descriptor->state_offset + target;
                        run->state_deposits[state_index] += value;
                        if (!lc_isfinite(run->state_deposits[state_index])) {
                            status = LC_NUMERIC_ERROR;
                            goto compiled_cleanup;
                        }
                    }
                    lc_sparse_node_add(
                        run->affected, run->affected_nodes, &affected_count,
                        target_node
                    );
                    run->deposit_affected[target_node] = 1U;
                }
            }

            {
                uint32_t affected_cursor;
                if (affected_count > 1U) {
                    qsort(
                        run->affected_nodes, affected_count, sizeof(uint32_t),
                        lc_uint32_before
                    );
                }
                for (affected_cursor = 0U;
                     affected_cursor < affected_count; ++affected_cursor) {
                const lc_mixed_node *descriptor;
                lc_real_t before[LC_ANALYTICAL_MAX_STATES];
                lc_real_t pre_deposit_readout;
                int has_deposit;
                int capture;
                uint32_t local;
                node = run->affected_nodes[affected_cursor];
                descriptor = &run->active_nodes[node];
                if (static_add_deliveries &&
                    lc_mixed_scalar_affine_enabled(
                        descriptor, &run->runtime[node]) &&
                    descriptor->crossing_kind !=
                        LC_CROSSING_INTEGRATED_HAZARD &&
                    descriptor->deposit_node_count == 0U &&
                    !descriptor->reset_before_deposit) {
                    uint32_t state_index = descriptor->state_offset;
                    status = lc_mixed_scalar_affine_advance(
                        descriptor, &run->runtime[node],
                        &run->state[state_index], &run->t_last[node], t
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    if (!run->runtime[node].clamped) {
                        run->state[state_index] +=
                            run->state_deposits[state_index];
                        if (!lc_isfinite(run->state[state_index])) {
                            status = LC_NUMERIC_ERROR;
                            goto compiled_cleanup;
                        }
                    }
                    run->state_deposits[state_index] = LC_REAL_C(0.0);
                    run->program_deposits[node] = LC_REAL_C(0.0);
                    run->deposit_affected[node] = 0U;
                    run->affected[node] = 0U;
                    if (run->runtime[node].clamped) {
                        continue;
                    }
                    if (run->runtime[node].generation == UINT64_MAX) {
                        status = LC_NUMERIC_ERROR;
                        goto compiled_cleanup;
                    }
                    run->runtime[node].generation++;
                    if (descriptor->crossing_kind !=
                            LC_CROSSING_INTEGRATED_HAZARD &&
                        run->state[state_index] >= descriptor->threshold) {
                        lc_sparse_node_add(
                            run->fired, run->fired_nodes, &fired_count, node
                        );
                    } else {
                        status = lc_mixed_schedule_prediction_planned(
                            run->active_nodes, run->state, run->t_last,
                            run->active_parameters, run->runtime, node,
                            config, heap, error, run->variables,
                            run->workspace, graph->workspace_count,
                            graph->node_eval_plans
                        );
                        if (status != LC_OK) {
                            goto compiled_cleanup;
                        }
                    }
                    continue;
                }
                has_deposit = run->deposit_affected[node] != 0U;
                capture = has_deposit && lc_trace_wants_state(
                    trace, LC_TRACE_DEPOSIT_APPLY, node
                );
                status = lc_mixed_advance_node_planned(
                    run->active_nodes, run->state, run->t_last,
                    run->active_parameters, run->runtime, node, t,
                    run->variables, run->workspace, graph->workspace_count,
                    graph->node_eval_plans, graph, run->learning_observers
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                if (capture) {
                    lc_trace_copy_state(
                        before, &run->state[descriptor->state_offset],
                        descriptor->state_count
                    );
                }
                pre_deposit_readout = run->state[
                    descriptor->state_offset + descriptor->readout
                ];
                if (has_deposit && descriptor->reset_before_deposit &&
                    !run->timestamp_reset[node]) {
                    const lc_expr_node *reset_nodes = graph->node_eval_plans == NULL
                        ? descriptor->program_nodes
                        : graph->node_eval_plans[node].reset_nodes;
                    status = lc_expr_state_map(
                        reset_nodes, descriptor->program_node_count,
                        descriptor->parameter_count > 0U
                            ? &run->active_parameters[descriptor->parameter_offset]
                            : NULL,
                        descriptor->parameter_count, descriptor->reset_roots,
                        descriptor->state_count,
                        &run->state[descriptor->state_offset], run->variables,
                        descriptor->state_count + 1U, run->workspace,
                        graph->workspace_count
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    run->t_last[node] = t;
                    run->timestamp_reset[node] = 1U;
                    run->timestamp_reset_nodes[timestamp_reset_count++] = node;
                }
                for (local = 0; local < descriptor->state_count; ++local) {
                    uint32_t state_index = descriptor->state_offset + local;
                    if (!(run->runtime[node].clamped &&
                          local == descriptor->readout)) {
                        run->state[state_index] +=
                            run->state_deposits[state_index];
                        if (!lc_isfinite(run->state[state_index])) {
                            status = LC_NUMERIC_ERROR;
                            goto compiled_cleanup;
                        }
                    }
                    run->state_deposits[state_index] = LC_REAL_C(0.0);
                }
                if (run->program_deposits[node] != LC_REAL_C(0.0) &&
                    !(run->runtime[node].clamped &&
                      descriptor->deposit_target == descriptor->readout)) {
                    const lc_expr_node *deposit_nodes =
                        graph->node_eval_plans == NULL
                            ? descriptor->deposit_nodes
                            : graph->node_eval_plans[node].deposit_nodes;
                    status = lc_expr_state_deposit(
                        deposit_nodes, descriptor->deposit_node_count,
                        descriptor->parameter_count > 0U
                            ? &run->active_parameters[descriptor->parameter_offset]
                            : NULL,
                        descriptor->parameter_count, descriptor->deposit_root,
                        run->program_deposits[node],
                        &run->state[descriptor->state_offset],
                        descriptor->state_count, descriptor->deposit_target,
                        run->variables, 1U, run->workspace,
                        graph->workspace_count
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                }
                if (has_deposit) {
                    status = lc_trace_emit(
                        trace, LC_TRACE_DEPOSIT_APPLY, LC_TRACE_PHASE_DEPOSIT,
                        t, node, UINT32_MAX, run->runtime[node].generation,
                        run->program_deposits[node], capture ? before : NULL,
                        capture ? &run->state[descriptor->state_offset] : NULL,
                        descriptor->state_count, error
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    if (!run->runtime[node].clamped &&
                        graph->learning_observer_program_by_node != NULL &&
                        graph->learning_observer_program_by_node[node] !=
                            UINT32_MAX) {
                        lc_real_t post_readout = run->state[
                            descriptor->state_offset + descriptor->readout
                        ];
                        lc_real_t amplitude;
                        status = lc_learning_soft_observation_amplitude(
                            graph, descriptor,
                            graph->node_eval_plans == NULL
                                ? descriptor->program_nodes
                                : graph->node_eval_plans[node].crossing_nodes,
                            node,
                            &run->state[descriptor->state_offset],
                            descriptor->parameter_count > 0U
                                ? &run->active_parameters[
                                    descriptor->parameter_offset
                                ] : NULL,
                            pre_deposit_readout, post_readout,
                            run->variables, run->workspace,
                            graph->workspace_count, &amplitude
                        );
                        if (status != LC_OK) {
                            goto compiled_cleanup;
                        }
                        if (amplitude > LC_REAL_C(0.0)) {
                            status = lc_learning_observe(
                                graph, run->plasticity,
                                run->learning_observers, node, t,
                                post_readout, amplitude, run->variables,
                                run->workspace, graph->workspace_count
                            );
                            if (status != LC_OK) {
                                goto compiled_cleanup;
                            }
                        }
                    }
                }
                run->program_deposits[node] = LC_REAL_C(0.0);
                run->deposit_affected[node] = 0U;
                run->affected[node] = 0U;
                if (run->runtime[node].clamped) {
                    continue;
                }
                if (run->runtime[node].generation == UINT64_MAX) {
                    status = LC_NUMERIC_ERROR;
                    goto compiled_cleanup;
                }
                run->runtime[node].generation++;
                if (descriptor->crossing_kind !=
                        LC_CROSSING_INTEGRATED_HAZARD &&
                    run->state[descriptor->state_offset + descriptor->readout] >=
                        descriptor->threshold) {
                    lc_sparse_node_add(
                        run->fired, run->fired_nodes, &fired_count, node
                    );
                } else {
                    status = lc_mixed_schedule_prediction_planned(
                        run->active_nodes, run->state, run->t_last,
                        run->active_parameters, run->runtime, node, config, heap,
                        error, run->variables, run->workspace,
                        graph->workspace_count, graph->node_eval_plans
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                }
                }
            }
            affected_count = 0U;

            while (lc_at(heap, t, LC_PHASE_PREDICTION)) {
                lc_event event = lc_heap_pop(heap);
                stats->events_popped++;
                if (event.generation != run->runtime[event.index].generation ||
                    run->runtime[event.index].clamped) {
                    stats->stale_predictions++;
                    status = lc_trace_emit(
                        trace, LC_TRACE_STALE_PREDICTION,
                        LC_TRACE_PHASE_PREDICTION, t, event.index, UINT32_MAX,
                        event.generation, LC_REAL_C(0.0), NULL, NULL, 0U, error
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    continue;
                }
                {
                    const lc_mixed_node *descriptor = &run->active_nodes[event.index];
                    status = lc_mixed_advance_node_planned(
                        run->active_nodes, run->state, run->t_last,
                        run->active_parameters, run->runtime, event.index, t,
                        run->variables, run->workspace,
                        graph->workspace_count, graph->node_eval_plans,
                        graph, run->learning_observers
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                    if (event.kind == LC_EVENT_NUMERICAL_CONTINUATION) {
                        status = lc_trace_emit(trace, LC_TRACE_NUMERICAL_CONTINUATION,
                            LC_TRACE_PHASE_PREDICTION, t, event.index, UINT32_MAX,
                            event.generation, LC_REAL_C(0.0), NULL, NULL, 0U, error);
                        if (status != LC_OK) goto compiled_cleanup;
                        status = lc_mixed_schedule_prediction_planned(
                            run->active_nodes, run->state, run->t_last,
                            run->active_parameters, run->runtime, event.index,
                            config, heap, error, run->variables, run->workspace,
                            graph->workspace_count, graph->node_eval_plans);
                        if (status != LC_OK) goto compiled_cleanup;
                        continue;
                    }
                    status = lc_trace_emit(
                        trace, LC_TRACE_PREDICTION_CONFIRMED,
                        LC_TRACE_PHASE_PREDICTION, t, event.index, UINT32_MAX,
                        event.generation, LC_REAL_C(0.0),
                        &run->state[descriptor->state_offset],
                        &run->state[descriptor->state_offset],
                        descriptor->state_count, error
                    );
                    if (status != LC_OK) {
                        goto compiled_cleanup;
                    }
                }
                lc_sparse_node_add(
                    run->fired, run->fired_nodes, &fired_count, event.index
                );
                stats->autonomous_spikes_confirmed++;
            }

            if (fired_count > 0U) {
                uint32_t fired_cursor;
                if (fired_count > 1U) {
                    qsort(
                        run->fired_nodes, fired_count, sizeof(uint32_t),
                        lc_uint32_before
                    );
                }
                cascade_depth++;
                if (cascade_depth > config->same_time_cascade_limit) {
                    status = LC_CASCADE_LIMIT;
                    goto compiled_cleanup;
                }
                if (cascade_depth > stats->max_same_time_cascade_depth) {
                    stats->max_same_time_cascade_depth = cascade_depth;
                }
                status = lc_mixed_fire_nodes(
                    run->active_nodes, run->state, run->t_last,
                    run->active_parameters, run->runtime, graph->node_count,
                    graph->edges, graph->outgoing_offsets,
                    graph->delivery_group_edges,
                    config, heap, run->fired, run->fired_nodes, fired_count, t,
                    outputs, output_count, decoders, stats, error, trace,
                    run->variables, run->workspace, graph->workspace_count,
                    graph, run->plasticity, run->learning_observers
                );
                if (status != LC_OK) {
                    goto compiled_cleanup;
                }
                for (fired_cursor = 0U;
                     fired_cursor < fired_count; ++fired_cursor) {
                    run->fired[run->fired_nodes[fired_cursor]] = 0U;
                }
                fired_count = 0U;
            }
            if (!lc_at(heap, t, LC_PHASE_DEPOSIT)) {
                break;
            }
        }
        for (node = 0U; node < timestamp_reset_count; ++node) {
            run->timestamp_reset[run->timestamp_reset_nodes[node]] = 0U;
        }
        status = lc_mixed_capture_inspections_through(
            run, inspections, &inspection_cursor, t, error
        );
        if (status != LC_OK) {
            goto compiled_cleanup;
        }
    }

    status = seal
                 ? lc_mixed_capture_inspections_through(
                       run, inspections, &inspection_cursor, until, error
                   )
                 : lc_mixed_capture_inspections_before(
                       run, inspections, &inspection_cursor, until, error
                   );
    if (status != LC_OK) {
        goto compiled_cleanup;
    }

    if (decoders != NULL) {
        status = seal
                     ? lc_decoder_run_advance(decoders, until)
                     : lc_decoder_run_advance_before(decoders, until);
        if (status != LC_OK) {
            if (status == LC_DECODER_OUTPUT_OVERFLOW) {
                lc_event event;
                memset(&event, 0, sizeof(event));
                event.t = until;
                event.phase = LC_PHASE_BOUNDARY;
                event.kind = LC_EVENT_DECODER_EVENT;
                event.index = UINT32_MAX;
                lc_set_decoder_resource_error(error, decoders, &event);
            }
            goto compiled_cleanup;
        }
    }

    for (node = 0; node < graph->node_count; ++node) {
        const lc_mixed_node *descriptor = &run->active_nodes[node];
        status = lc_mixed_advance_node_planned(
            run->active_nodes, run->state, run->t_last, run->active_parameters,
            run->runtime, node, until, run->variables, run->workspace,
            graph->workspace_count, graph->node_eval_plans, graph,
            run->learning_observers
        );
        if (status != LC_OK) {
            goto compiled_cleanup;
        }
        if (seal) {
            status = lc_trace_emit(
                trace, LC_TRACE_FINAL_STATE, LC_TRACE_PHASE_FINAL, until,
                node, UINT32_MAX, run->runtime[node].generation, LC_REAL_C(0.0),
                &run->state[descriptor->state_offset],
                &run->state[descriptor->state_offset],
                descriptor->state_count, error
            );
            if (status != LC_OK) {
                goto compiled_cleanup;
            }
        }
    }

compiled_cleanup:
    stats->peak_queue_occupancy = heap->peak;
    stats->kernel_seconds = lc_monotonic_seconds() - kernel_start;
    if (resume) {
        if (trace != NULL && *trace->count <= UINT64_MAX - run->trace_sequence) {
            run->trace_sequence += *trace->count;
        }
        if (status != LC_OK) {
            run->incremental_failed = 1;
            run->incremental_active = 0;
        } else {
            run->frontier = until;
            if (seal) {
                run->incremental_active = 0;
                heap->size = 0U;
                heap->logical_size = 0U;
            }
        }
    } else {
        heap->size = 0U;
        heap->logical_size = 0U;
    }
    return status;
}

/* Execute one complete mixed run without optional observers. */
lc_status lc_mixed_run_execute(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, NULL, NULL, NULL, 0,
        config->t_end, 1
    );
}

/* Execute one complete mixed run with causal trace recording. */
lc_status lc_mixed_run_execute_recorded(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_trace_config *trace
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (trace == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, NULL, trace, NULL, 0,
        config->t_end, 1
    );
}

/* Execute one complete mixed run with streaming decoders. */
lc_status lc_mixed_run_execute_with_decoders(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (decoders == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, decoders, NULL, NULL, 0,
        config->t_end, 1
    );
}

/* Execute with streaming decoders and causal trace recording. */
lc_status lc_mixed_run_execute_with_decoders_recorded(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders,
    lc_trace_config *trace
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (decoders == NULL || trace == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, decoders, trace, NULL, 0,
        config->t_end, 1
    );
}

/* Execute with any supported combination of read-only observers. */
lc_status lc_mixed_run_execute_observed(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders,
    lc_trace_config *trace,
    lc_state_inspection_config *inspections
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (decoders == NULL && trace == NULL && inspections == NULL) {
        return LC_INVALID_ARGUMENT;
    }
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, decoders, trace, inspections, 0,
        config->t_end, 1
    );
}

/* Advance a resumable run to an open or sealed boundary. */
lc_status lc_mixed_run_advance_incremental(
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    lc_time_t until,
    uint32_t seal,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders,
    lc_trace_config *trace,
    lc_state_inspection_config *inspections
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    if (run == NULL || seal > 1U) {
        return LC_INVALID_ARGUMENT;
    }
    return lc_mixed_run_execute_internal(
        run, inputs, input_count, drive_updates, drive_update_count,
        &run->incremental_config, outputs, output_count, stats, error,
        decoders, trace, inspections, 1, until, seal != 0U
    );
}

/* Compile, execute, record, and destroy a temporary mixed graph. */
lc_status lc_mixed_network_run_recorded(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    lc_real_t *state,
    uint32_t state_count,
    lc_time_t *t_last,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    uint32_t edge_count,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    const lc_run_config *config,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_trace_config *trace
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    lc_compiled_graph *compiled = NULL;
    lc_mixed_run *run = NULL;
    lc_status status;
    lc_status copy_status;
    status = lc_mixed_graph_compile(
        nodes, node_count, state_count, parameters, parameter_count, edges,
        edge_count, &compiled
    );
    if (status != LC_OK) {
        return status;
    }
    status = lc_mixed_run_create(
        compiled, state, state_count, t_last, node_count, &run
    );
    lc_mixed_graph_destroy(compiled);
    if (status != LC_OK) {
        return status;
    }
    status = lc_mixed_run_execute_recorded(
        run, inputs, input_count, drive_updates, drive_update_count, config,
        outputs, output_count, stats, error, trace
    );
    copy_status = lc_mixed_run_copy_state(
        run, state, state_count, t_last, node_count
    );
    lc_mixed_run_destroy(run);
    return status == LC_OK ? copy_status : status;
}

/* Copy current node state and update times without changing the frontier. */
lc_status lc_mixed_run_copy_state(
    const lc_mixed_run *run,
    lc_real_t *state,
    uint32_t state_count,
    lc_time_t *t_last,
    uint32_t node_count
) {
    if (run == NULL || run->graph == NULL || state == NULL || t_last == NULL ||
        state_count != run->graph->state_count ||
        node_count != run->graph->node_count) {
        return LC_INVALID_ARGUMENT;
    }
    memcpy(state, run->state, state_count * sizeof(lc_real_t));
    memcpy(t_last, run->t_last, node_count * sizeof(lc_time_t));
    return LC_OK;
}

/* Queue chronological third-factor events before execution advances. */
lc_status lc_mixed_run_schedule_modulations(
    lc_mixed_run *run,
    const lc_modulation_event *events,
    uint32_t event_count
) {
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) {
        return LC_NUMERIC_ERROR;
    }
#endif
    uint32_t index;
    if (run == NULL || run->graph == NULL ||
        (event_count > 0U && events == NULL) ||
        run->incremental_failed ||
        lc_allocation_size_overflows(event_count, sizeof(lc_modulation_event))) {
        return LC_INVALID_ARGUMENT;
    }
    for (index = 0U; index < event_count; ++index) {
        if (!lc_isfinite(events[index].t) || !lc_isfinite(events[index].value) ||
            events[index].modulator >= run->graph->modulator_count ||
            events[index].t < (run->incremental_active ? run->frontier : run->t_last[0]) ||
            (run->incremental_active &&
             events[index].t > run->incremental_config.t_end)) {
            return LC_INVALID_ARGUMENT;
        }
    }
    if (run->incremental_active) {
        for (index = 0U; index < event_count; ++index) {
            lc_event event;
            lc_status status;
            memset(&event, 0, sizeof(event));
            event.t = events[index].t;
            event.phase = LC_PHASE_BOUNDARY;
            event.kind = LC_EVENT_MODULATION;
            event.index = events[index].modulator;
            event.subject = index;
            event.value = events[index].value;
            status = lc_heap_push(&run->heap, event);
            if (status != LC_OK) {
                return status;
            }
        }
        return LC_OK;
    }
    if (!run->ready || run->pending_modulation_count != 0U) {
        return LC_INVALID_ARGUMENT;
    }
    if (event_count > 0U) {
        run->pending_modulations = calloc(
            event_count, sizeof(lc_modulation_event)
        );
        if (run->pending_modulations == NULL) {
            return LC_ALLOCATION_FAILED;
        }
        memcpy(
            run->pending_modulations, events,
            event_count * sizeof(lc_modulation_event)
        );
        run->pending_modulation_count = event_count;
    }
    return LC_OK;
}

/* Copy current edge weights in canonical edge order. */
lc_status lc_mixed_run_copy_weights(
    const lc_mixed_run *run,
    lc_real_t *weights,
    uint32_t edge_count
) {
    uint32_t edge;
    if (run == NULL || run->graph == NULL || weights == NULL ||
        edge_count != run->graph->edge_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (edge = 0U; edge < edge_count; ++edge) {
        weights[edge] = lc_mixed_edge_weight(run->graph, run->plasticity, edge);
    }
    return LC_OK;
}

/* Copy observable learning state for every plastic edge. */
lc_status lc_mixed_run_copy_plasticity(
    const lc_mixed_run *run,
    lc_plasticity_state *states,
    uint32_t capacity,
    uint32_t *count
) {
    if (run == NULL || run->graph == NULL || count == NULL ||
        (capacity > 0U && states == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *count = run->graph->plastic_edge_count;
    if (capacity < run->graph->plastic_edge_count) {
        return LC_OUTPUT_OVERFLOW;
    }
    if (run->graph->plastic_edge_count > 0U) {
        uint32_t slot;
        for (slot = 0U; slot < run->graph->plastic_edge_count; ++slot) {
            states[slot] = run->plasticity[slot];
            states[slot].weight = lc_plastic_slot_weight(
                run->graph, run->plasticity, slot
            );
        }
    }
    return LC_OK;
}

/* Copy shared neuron-local learning-observer state for inspection. */
lc_status lc_mixed_run_copy_learning_observers(
    const lc_mixed_run *run,
    lc_learning_observer_snapshot *states,
    uint32_t capacity,
    uint32_t *count
) {
    uint32_t node;
    if (run == NULL || run->graph == NULL || count == NULL ||
        (capacity > 0U && states == NULL)) {
        return LC_INVALID_ARGUMENT;
    }
    *count = run->graph->node_count;
    if (capacity < run->graph->node_count) {
        return LC_OUTPUT_OVERFLOW;
    }
    for (node = 0U; node < run->graph->node_count; ++node) {
        const lc_learning_observer_state *source =
            &run->learning_observers[node];
        lc_learning_observer_snapshot *target = &states[node];
        target->node = node;
        target->active =
            run->graph->learning_observer_program_by_node != NULL &&
            run->graph->learning_observer_program_by_node[node] != UINT32_MAX;
        target->slow_voltage = source->slow_voltage;
        target->fast_activity = source->fast_activity;
        target->slow_activity = source->slow_activity;
        target->sensitivity = source->sensitivity;
        target->t_activity = source->t_activity;
    }
    return LC_OK;
}

void lc_mixed_run_destroy(lc_mixed_run *run) {
    lc_compiled_graph *graph;
    if (run == NULL) {
        return;
    }
    graph = run->graph;
    free(run->state);
    free(run->t_last);
    free(run->active_nodes);
    free(run->active_parameters);
    lc_runtime_clear_caches(run->runtime, graph == NULL ? 0U : graph->node_count);
    free(run->runtime);
    free(run->learning_observers);
    free(run->state_deposits);
    free(run->program_deposits);
    free(run->affected);
    free(run->deposit_affected);
    free(run->timestamp_reset);
    free(run->fired);
    free(run->affected_nodes);
    free(run->fired_nodes);
    free(run->timestamp_reset_nodes);
    free(run->workspace);
    free(run->plasticity);
    free(run->learning_weight_deltas);
    free(run->learning_weight_touched);
    free(run->learning_weight_masters);
    free(run->pending_modulations);
    free(run->heap.items);
    free(run);
    lc_compiled_graph_release(graph);
}
