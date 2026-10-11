/* Private-plan regressions compare the identical integrator with/without the
 * derivative view. No tolerance-based comparison of trajectories is used. */
#include "../../c/src/step_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static const lc_step_config config = {
    1e-12, 1e-14, .25, 1e-12, .25, 1e-10, 100000U, 700000U
};

static lc_status predict(const lc_expr_node *nodes, uint32_t n,
    lc_real_t *parameters, uint32_t parameter_count, const uint32_t *roots,
    uint32_t states, lc_real_t threshold, const lc_real_t *state, lc_time_t t,
    lc_step_cache *cache, lc_step_result *result) {
    lc_real_t variables[LC_ANALYTICAL_MAX_STATES + 1U];
    lc_real_t workspace[128];
    /* Shared scratch is intentionally poisoned between every prediction. */
    memset(workspace, 0xff, sizeof(workspace));
    return lc_expr_step_predict_cached(nodes, n, parameters, parameter_count,
        roots, states, 0U, threshold, &config, state, t, t + .25,
        variables, states + 1U, workspace, 128U, result, cache);
}

static void compare(const lc_step_cache *a, const lc_step_cache *b,
                    const lc_step_result *x, const lc_step_result *y) {
    assert(a->count == b->count && a->state_count == b->state_count);
    for (uint32_t i = 0U; i < a->count; ++i) {
        assert(a->segments[i].start == b->segments[i].start);
        assert(a->segments[i].step == b->segments[i].step);
        assert(memcmp(a->segments[i].coefficients, b->segments[i].coefficients,
                      a->state_count * 5U * sizeof(lc_real_t)) == 0);
        assert(memcmp(a->segments[i].endpoint, b->segments[i].endpoint,
                      a->state_count * sizeof(lc_real_t)) == 0);
    }
    assert(memcmp(x, y, sizeof(*x)) == 0);
}

static void test_persistent_plan_and_parameter_refresh(void) {
    /* y'=y*y+exp(p), plus time-dependent second state; an unused expression
     * is structurally valid and finite here. Parameter p changes in place. */
    const lc_expr_node nodes[] = {
        {LC_EXPR_VAR,0,0,1,0}, {LC_EXPR_PARAM,0,0,0,0},
        {LC_EXPR_EXP,1,0,0,0}, {LC_EXPR_MUL,0,0,0,0},
        {LC_EXPR_ADD,3,2,0,0}, {LC_EXPR_VAR,0,0,0,0},
        {LC_EXPR_SIN,5,0,0,0}, {LC_EXPR_VAR,0,0,2,0},
        {LC_EXPR_SUB,6,7,0,0}, {LC_EXPR_COS,0,0,0,0}
    };
    uint32_t roots[] = {4,8};
    lc_real_t parameters[] = {0.};
    lc_real_t state[] = {0.,-.0};
    lc_step_cache planned = {0}, full = {0};
    lc_step_result a, b;
    lc_time_t t = 0.;
    unsigned i, rejected = 0U;
    full.disable_rhs_plan = 1U;
    for (i = 0; i < 4; ++i) {
        lc_status sa, sb;
        if (i == 2) {
            parameters[0] = -1.;
            /* The original execution policy invalidates on parameter updates;
             * the new cache also detects changes independently. */
            full.resume_valid = 0U;
        }
        sa = predict(nodes,10,parameters,1,roots,2,100.,state,t,&planned,&a);
        sb = predict(nodes,10,parameters,1,roots,2,100.,state,t,&full,&b);
        assert(sa == LC_NO_CROSSING && sb == sa);
        assert(planned.rhs_plan != NULL && full.rhs_plan == NULL);
        compare(&planned,&full,&a,&b);
        rejected += a.rejected_steps;
        assert(lc_step_cache_sample(&planned,a.t_reached,state,2));
        t = a.t_reached;
    }
    assert(rejected > 0);
    /* Changing the root set rebinds the plan and invalidates FSAL as well. */
    roots[0] = 8; full.resume_valid = 0;
    assert(predict(nodes,10,parameters,1,roots,2,100.,state,t,&planned,&a) == LC_NO_CROSSING);
    assert(predict(nodes,10,parameters,1,roots,2,100.,state,t,&full,&b) == LC_NO_CROSSING);
    compare(&planned,&full,&a,&b);
    lc_step_cache_release(&planned); lc_step_cache_release(&planned);
    assert(planned.rhs_plan == NULL);
    lc_step_cache_release(&full);
}

static void test_all_expression_operations(void) {
    /* Each supported op appears once with dynamic dependencies and once with
     * parameter-only dependencies, exercising folding and index remapping. */
    uint32_t op;
    for (op = LC_EXPR_NEG; op <= LC_EXPR_MAX; ++op) {
        lc_expr_node nodes[] = {
            {LC_EXPR_VAR,0,0,1,0}, {LC_EXPR_PARAM,0,0,0,0},
            {LC_EXPR_CONST,0,0,0,2.}, {0,0,2,0,0},
            {0,1,2,0,0}, {LC_EXPR_ADD,3,4,0,0},
            {LC_EXPR_CONST,0,0,0,.001}, {LC_EXPR_MUL,5,6,0,0}
        };
        uint32_t root = 7;
        lc_real_t parameter = .7, state = .5;
        lc_step_cache planned = {0}, full = {0}, checked_plan = {0};
        lc_step_result a, b, c;
        full.disable_rhs_plan = 1;
        checked_plan.disable_validated_rhs = 1;
        nodes[3].op = op; nodes[4].op = op;
        assert(predict(nodes,8,&parameter,1,&root,1,10.,&state,0,&planned,&a) == LC_NO_CROSSING);
        assert(predict(nodes,8,&parameter,1,&root,1,10.,&state,0,&full,&b) == LC_NO_CROSSING);
        compare(&planned,&full,&a,&b);
        assert(predict(nodes,8,&parameter,1,&root,1,10.,&state,0,&checked_plan,&c) == LC_NO_CROSSING);
        compare(&planned,&checked_plan,&a,&c);
        lc_step_cache_release(&planned);
        lc_step_cache_release(&full);
        lc_step_cache_release(&checked_plan);
    }
}

static void test_domains_and_unused_roots(void) {
    lc_expr_node nodes[] = {
        {LC_EXPR_PARAM,0,0,0,0}, {LC_EXPR_LOG,0,0,0,0},
        {LC_EXPR_VAR,0,0,1,0}, {LC_EXPR_LOG,2,0,0,0}
    };
    uint32_t root = 1;
    lc_real_t parameter = 2., state = -1.;
    lc_step_cache planned = {0};
    lc_step_result a;
    /* A reset-only log(state) is not a derivative: do not evaluate it at
     * intermediate RK stages. Its own operation remains responsible for it. */
    assert(predict(nodes,4,&parameter,1,&root,1,100.,&state,0,&planned,&a) == LC_NO_CROSSING);
    parameter = -1.;
    assert(predict(nodes,4,&parameter,1,&root,1,100.,&state,0,&planned,&a) == LC_NUMERIC_ERROR);
    parameter = NAN;
    assert(predict(nodes,4,&parameter,1,&root,1,100.,&state,0,&planned,&a) == LC_NUMERIC_ERROR);
    parameter = 2.; root = 3;
    assert(predict(nodes,4,&parameter,1,&root,1,100.,&state,0,&planned,&a) == LC_NUMERIC_ERROR);
    root = 99;
    assert(predict(nodes,4,&parameter,1,&root,1,100.,&state,0,&planned,&a) == LC_INVALID_ARGUMENT);
    lc_step_cache_release(&planned);
}

static void test_crossing_coefficients_and_signed_zero(void) {
    const lc_expr_node nodes[] = {{LC_EXPR_PARAM,0,0,0,0}};
    uint32_t root = 0;
    lc_real_t parameter = 1., state = -.0, variables[2], workspace[1];
    lc_step_cache cache = {0};
    lc_step_result cached, uncached;
    assert(predict(nodes,1,&parameter,1,&root,1,.125,&state,0,&cache,&cached) == LC_OK);
    /* One constant-derivative step: uncached detection computes coefficients
     * independently, while cached detection reuses them. */
    assert(lc_expr_step_predict(nodes,1,&parameter,1,&root,1,0,.125,&config,
        &state,0,.25,variables,2,workspace,1,&uncached) == LC_OK);
    assert(memcmp(&cached,&uncached,sizeof(cached)) == 0);
    assert(signbit(cache.segments[0].coefficients[0][0]));
    lc_step_cache_release(&cache);
}

static void test_compact_storage_reset_and_resize(void) {
    lc_expr_node nodes[LC_ANALYTICAL_MAX_STATES * 2U];
    uint32_t roots[LC_ANALYTICAL_MAX_STATES];
    lc_real_t state[LC_ANALYTICAL_MAX_STATES], sample[LC_ANALYTICAL_MAX_STATES];
    lc_step_cache reused = {0};
    for (uint32_t size = 1; size <= LC_ANALYTICAL_MAX_STATES; ++size) {
        lc_step_result first, after_reset, fresh_result;
        lc_step_cache fresh = {0};
        lc_real_t *storage;
        lc_step_rhs_plan *plan;
        for (uint32_t j = 0; j < size; ++j) {
            nodes[2*j] = (lc_expr_node){LC_EXPR_VAR,0,0,j+1,0};
            nodes[2*j+1] = (lc_expr_node){LC_EXPR_NEG,2*j,0,0,0};
            roots[j] = 2*j+1;
            state[j] = j+.25;
        }
        /* Changing the program/layout explicitly; a previous plan must not
         * survive an in-place source edit (internal programs are immutable). */
        lc_step_cache_release(&reused);
        assert(predict(nodes,2*size,NULL,0,roots,size,100.,state,0,&reused,&first) == LC_NO_CROSSING);
        assert(reused.storage_states == size);
        storage = reused.storage; plan = reused.rhs_plan;
        assert(reused.derivative + size == storage + 50U*size);
        for (uint32_t i = 0; i < LC_STEP_CACHE_SEGMENTS; ++i) {
            assert((lc_real_t *)reused.segments[i].coefficients == storage + 6U*size*i);
            assert(reused.segments[i].endpoint == storage + 6U*size*i + 5U*size);
        }
        lc_step_cache_reset(&reused);
        assert(reused.storage == storage && reused.rhs_plan == plan);
        assert(reused.count == 0 && !reused.resume_valid);
        assert(!lc_step_cache_sample(&reused,.01,sample,size));
        for (uint32_t j = 0; j < size; ++j) state[j] = j+.75;
        assert(predict(nodes,2*size,NULL,0,roots,size,100.,state,0,&reused,&after_reset) == LC_NO_CROSSING);
        assert(reused.storage == storage && reused.rhs_plan == plan);
        assert(predict(nodes,2*size,NULL,0,roots,size,100.,state,0,&fresh,&fresh_result) == LC_NO_CROSSING);
        compare(&reused,&fresh,&after_reset,&fresh_result);
        lc_step_cache_release(&fresh);
        /* Resize the retained slab using a smaller root/layout on the same
         * immutable source, then validate that it cannot read the old history. */
        if (size > 1) {
            const lc_expr_node constant = {LC_EXPR_CONST,0,0,0,0.1};
            uint32_t root = 0;
            assert(predict(&constant,1,NULL,0,&root,1,100.,state,0,&reused,&fresh_result) == LC_NO_CROSSING);
            assert(reused.storage_states == 1);
        }
    }
    lc_step_cache_release(&reused);
    lc_step_cache_release(&reused);
    assert(reused.storage == NULL && reused.rhs_plan == NULL && reused.count == 0);
}

static void test_validation_boundary_and_dynamic_failures(void) {
    const lc_expr_node bad_ref[] = {{LC_EXPR_NEG,99,0,0,0}};
    const lc_expr_node bad_var[] = {{LC_EXPR_VAR,0,0,99,0}};
    const lc_expr_node bad_op[] = {{99,0,0,0,0}};
    const lc_expr_node bad_param[] = {{LC_EXPR_PARAM,0,0,0,0}};
    const lc_expr_node *invalid[] = {bad_ref,bad_var,bad_op,bad_param};
    lc_real_t state = .5;
    uint32_t root = 0;
    lc_step_result result;
    lc_step_cache cache = {0};
    for (uint32_t i = 0; i < 4; ++i) {
        assert(predict(invalid[i],1,NULL,0,&root,1,10.,&state,0,&cache,&result) == LC_INVALID_ARGUMENT);
        assert(cache.rhs_plan == NULL);
    }
    lc_step_cache_release(&cache);
    /* Active division-by-zero, invalid power and exponential overflow are
     * still caught, identically to the checked derivative evaluator. */
    for (uint32_t op = 0; op < 3; ++op) {
        lc_expr_node nodes[] = {{LC_EXPR_VAR,0,0,1,0},
            {LC_EXPR_CONST,0,0,0,0}, {LC_EXPR_DIV,0,1,0,0}};
        lc_step_cache fast = {0}, checked = {0};
        lc_step_result a,b;
        checked.disable_validated_rhs = 1;
        root = 2;
        if (op == 1) { nodes[2].op = LC_EXPR_POW; nodes[1].value = .5; state = -1.; }
        else if (op == 2) { nodes[2].op = LC_EXPR_EXP; state = 1000.; }
        else state = 1.;
        assert(predict(nodes,3,NULL,0,&root,1,2000.,&state,0,&fast,&a) == LC_NUMERIC_ERROR);
        assert(predict(nodes,3,NULL,0,&root,1,2000.,&state,0,&checked,&b) == LC_NUMERIC_ERROR);
        lc_step_cache_release(&fast); lc_step_cache_release(&checked);
    }
}

int main(void) {
    test_persistent_plan_and_parameter_refresh();
    test_all_expression_operations();
    test_domains_and_unused_roots();
    test_crossing_coefficients_and_signed_zero();
    test_compact_storage_reset_and_resize();
    test_validation_boundary_and_dynamic_failures();
    printf("cache metadata %zu bytes; 1-state total %zu; 2-state total %zu\n",
        sizeof(lc_step_cache), sizeof(lc_step_cache)+50U*sizeof(lc_real_t),
        sizeof(lc_step_cache)+100U*sizeof(lc_real_t));
    return 0;
}
