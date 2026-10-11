#ifndef LACUNA_STEP_INTERNAL_H
#define LACUNA_STEP_INTERNAL_H

#include "lacuna.h"

/* Bounded, run-owned speculative history; never part of a compiled image/ABI.
 * Float64 network execution only. Reduced precision retains its audited path. */
#define LC_STEP_CACHE_SEGMENTS 8U
typedef struct lc_step_rhs_plan lc_step_rhs_plan;
typedef struct lc_step_segment {
    lc_time_t start, step;
    lc_real_t (*coefficients)[5];
    lc_real_t *endpoint;
} lc_step_segment;
typedef struct lc_step_cache {
    uint32_t count, state_count;
    lc_time_t end;
    lc_step_segment segments[LC_STEP_CACHE_SEGMENTS];
    /* Valid only at the exact uninterrupted accepted endpoint. The owning
     * network must invalidate on every state/parameter-generation change. */
    uint32_t resume_valid, disable_reuse;
    lc_time_t resume_time, next_step;
    lc_real_t *resume_state, *derivative;
    /* One right-sized slab: eight segments plus endpoint continuation data. */
    lc_real_t *storage;
    uint32_t storage_states;
    /* Equation-derived, parameter-bound derivative view. Private run storage;
     * never serialized. The test switch leaves the integrator unchanged. */
    lc_step_rhs_plan *rhs_plan;
    uint32_t disable_rhs_plan;
    uint32_t disable_validated_rhs;
} lc_step_cache;

/* Release owned trajectory/derivative storage, not the containing cache itself. */
void lc_step_cache_release(lc_step_cache *cache);
/* Invalidate trajectories/FSAL but retain owned storage and immutable plans. */
void lc_step_cache_reset(lc_step_cache *cache);

lc_status lc_expr_step_predict_cached(
    const lc_expr_node *nodes, uint32_t node_count,
    const lc_real_t *parameters, uint32_t parameter_count,
    const uint32_t *rhs_roots, uint32_t state_count, uint32_t readout,
    lc_real_t threshold, const lc_step_config *config, const lc_real_t *state,
    lc_time_t t_last, lc_time_t horizon, lc_real_t *variables,
    uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count,
    lc_step_result *result, lc_step_cache *cache);

/* Reads only; partial observations do not move integration boundaries. */
int lc_step_cache_sample(const lc_step_cache *cache, lc_time_t t,
                         lc_real_t *values, uint32_t state_count);
#endif
