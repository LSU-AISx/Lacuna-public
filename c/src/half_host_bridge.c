/* Host-only bit transport for ctypes. Not part of the numerical runtime. */
#include "lacuna.h"
#include <string.h>

#if LACUNA_REAL_BITS != 16 || LACUNA_TIME_BITS != 16
#error "The half host bridge requires uniform binary16 core types"
#endif

typedef char lc_host_half_size_check[(sizeof(lc_real_t) == sizeof(uint16_t)) ? 1 : -1];

static lc_real_t lc_host_half_from_bits(uint16_t bits) {
    lc_real_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef lc_status (*lc_host_profile_check_fn)(uint32_t, uint32_t, uint32_t, uint32_t);
typedef char lc_host_pointer_size_check[(sizeof(lc_host_profile_check_fn) == sizeof(void *)) ? 1 : -1];

LC_API uint32_t lc_host_half_transport_version(void) {
    return 1U;
}

LC_API lc_status lc_host_half_profile_check(void *function) {
    lc_host_profile_check_fn native_call;
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(4U, 16U, 16U, 1U);
}

LC_API lc_status lc_host_half_encoder_run_create(
    void *function,
    const lc_encoder_spec *specs,
    uint32_t spec_count,
    uint64_t seed,
    uint16_t initial_frontier,
    lc_encoder_run **run
) {
    lc_status (*native_call)(const lc_encoder_spec *specs, uint32_t spec_count, uint64_t seed, lc_time_t initial_frontier, lc_encoder_run **run);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        specs,
        spec_count,
        seed,
        lc_host_half_from_bits(initial_frontier),
        run
    );
}

LC_API lc_status lc_host_half_encoder_run_advance(
    void *function,
    lc_encoder_run *run,
    const lc_presentation *presentations,
    uint32_t presentation_count,
    uint16_t until,
    uint32_t seal,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
) {
    lc_status (*native_call)(lc_encoder_run *run, const lc_presentation *presentations, uint32_t presentation_count, lc_time_t until, uint32_t seal, lc_encoded_spike *spikes, uint64_t spike_capacity, uint64_t *spike_count, lc_encoded_drive *drives, uint64_t drive_capacity, uint64_t *drive_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        run,
        presentations,
        presentation_count,
        lc_host_half_from_bits(until),
        seal,
        spikes,
        spike_capacity,
        spike_count,
        drives,
        drive_capacity,
        drive_count
    );
}

LC_API lc_status lc_host_half_decoder_run_advance(
    void *function,
    lc_decoder_run *run,
    uint16_t observed_through
) {
    lc_status (*native_call)(lc_decoder_run *run, lc_time_t observed_through);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        run,
        lc_host_half_from_bits(observed_through)
    );
}

LC_API lc_status lc_host_half_decoder_run_advance_before(
    void *function,
    lc_decoder_run *run,
    uint16_t observed_before
) {
    lc_status (*native_call)(lc_decoder_run *run, lc_time_t observed_before);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        run,
        lc_host_half_from_bits(observed_before)
    );
}

LC_API lc_status lc_host_half_scalar_advance(
    void *function,
    const lc_scalar_lif_model *model,
    lc_scalar_state *state,
    uint16_t t
) {
    lc_status (*native_call)(const lc_scalar_lif_model *model, lc_scalar_state *state, lc_time_t t);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        model,
        state,
        lc_host_half_from_bits(t)
    );
}

LC_API lc_status lc_host_half_expr_state_advance(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *roots,
    uint32_t state_count,
    lc_real_t *state,
    lc_time_t *t_last,
    uint16_t t,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const uint32_t *roots, uint32_t state_count, lc_real_t *state, lc_time_t *t_last, lc_time_t t, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        roots,
        state_count,
        state,
        t_last,
        lc_host_half_from_bits(t),
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_expr_state_deposit(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    uint32_t root,
    uint16_t weight,
    lc_real_t *state,
    uint32_t state_count,
    uint32_t target,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, uint32_t root, lc_real_t weight, lc_real_t *state, uint32_t state_count, uint32_t target, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        root,
        lc_host_half_from_bits(weight),
        state,
        state_count,
        target,
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_expr_step_advance(
    void *function,
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
    uint16_t t,
    uint32_t clamped,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_step_result *result
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const uint32_t *rhs_roots, uint32_t state_count, uint32_t readout, const lc_step_config *config, lc_real_t *state, lc_time_t *t_last, lc_time_t t, uint32_t clamped, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count, lc_step_result *result);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        rhs_roots,
        state_count,
        readout,
        config,
        state,
        t_last,
        lc_host_half_from_bits(t),
        clamped,
        variables,
        variable_count,
        workspace,
        workspace_count,
        result
    );
}

LC_API lc_status lc_host_half_expr_step_predict(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const uint32_t *rhs_roots,
    uint32_t state_count,
    uint32_t readout,
    uint16_t threshold,
    const lc_step_config *config,
    const lc_real_t *state,
    uint16_t t_last,
    uint16_t horizon,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count,
    lc_step_result *result
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const uint32_t *rhs_roots, uint32_t state_count, uint32_t readout, lc_real_t threshold, const lc_step_config *config, const lc_real_t *state, lc_time_t t_last, lc_time_t horizon, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count, lc_step_result *result);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        rhs_roots,
        state_count,
        readout,
        lc_host_half_from_bits(threshold),
        config,
        state,
        lc_host_half_from_bits(t_last),
        lc_host_half_from_bits(horizon),
        variables,
        variable_count,
        workspace,
        workspace_count,
        result
    );
}

LC_API lc_status lc_host_half_expr_alpha_predict(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_root_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    uint16_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const lc_root_hint *hint, const lc_real_t *state, uint32_t state_count, lc_time_t t_last, lc_root_result *result, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        hint,
        state,
        state_count,
        lc_host_half_from_bits(t_last),
        result,
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_expr_two_exp_predict(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_two_exp_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    uint16_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const lc_two_exp_hint *hint, const lc_real_t *state, uint32_t state_count, lc_time_t t_last, lc_root_result *result, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        hint,
        state,
        state_count,
        lc_host_half_from_bits(t_last),
        result,
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_expr_multi_exp_predict(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_multi_exp_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    uint16_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const lc_multi_exp_hint *hint, const lc_real_t *state, uint32_t state_count, lc_time_t t_last, lc_root_result *result, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        hint,
        state,
        state_count,
        lc_host_half_from_bits(t_last),
        result,
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_expr_exp_poly_predict(
    void *function,
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_exp_poly_hint *hint,
    const lc_real_t *state,
    uint32_t state_count,
    uint16_t t_last,
    lc_root_result *result,
    lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
) {
    lc_status (*native_call)(const lc_expr_node *nodes, uint32_t node_count, const lc_real_t *parameters, uint32_t parameter_count, const lc_exp_poly_hint *hint, const lc_real_t *state, uint32_t state_count, lc_time_t t_last, lc_root_result *result, lc_real_t *variables, uint32_t variable_count, lc_real_t *workspace, uint32_t workspace_count);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        nodes,
        node_count,
        parameters,
        parameter_count,
        hint,
        state,
        state_count,
        lc_host_half_from_bits(t_last),
        result,
        variables,
        variable_count,
        workspace,
        workspace_count
    );
}

LC_API lc_status lc_host_half_mixed_run_advance_incremental(
    void *function,
    lc_mixed_run *run,
    const lc_mixed_input_spike *inputs,
    uint32_t input_count,
    const lc_mixed_drive_update *drive_updates,
    uint32_t drive_update_count,
    uint16_t until,
    uint32_t seal,
    lc_output_spike *outputs,
    uint64_t *output_count,
    lc_run_stats *stats,
    lc_network_error *error,
    lc_decoder_run *decoders,
    lc_trace_config *trace,
    lc_state_inspection_config *inspections
) {
    lc_status (*native_call)(lc_mixed_run *run, const lc_mixed_input_spike *inputs, uint32_t input_count, const lc_mixed_drive_update *drive_updates, uint32_t drive_update_count, lc_time_t until, uint32_t seal, lc_output_spike *outputs, uint64_t *output_count, lc_run_stats *stats, lc_network_error *error, lc_decoder_run *decoders, lc_trace_config *trace, lc_state_inspection_config *inspections);
    if (function == NULL) return LC_INVALID_ARGUMENT;
    memcpy(&native_call, &function, sizeof(native_call));
    return native_call(
        run,
        inputs,
        input_count,
        drive_updates,
        drive_update_count,
        lc_host_half_from_bits(until),
        seal,
        outputs,
        output_count,
        stats,
        error,
        decoders,
        trace,
        inspections
    );
}
