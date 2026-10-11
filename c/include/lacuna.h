#ifndef LACUNA_H
#define LACUNA_H

#include <stdint.h>
#include "lacuna_numeric.h"

#if defined(_WIN32)
#define LC_API __declspec(dllexport)
#else
#define LC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Status values are stable across the public C ABI. */
typedef enum lc_status {
    LC_OK = 0,
    LC_NO_CROSSING = 1,
    LC_INVALID_ARGUMENT = 2,
    LC_TIME_REVERSED = 3,
    LC_NUMERIC_ERROR = 4,
    LC_UNSUPPORTED_MODEL = 5,
    LC_QUEUE_OVERFLOW = 6,
    LC_OUTPUT_OVERFLOW = 7,
    LC_CASCADE_LIMIT = 8,
    LC_ALLOCATION_FAILED = 9,
    LC_ROOT_NONCONVERGENCE = 10,
    LC_DECODER_OUTPUT_OVERFLOW = 11,
    LC_TRACE_OVERFLOW = 12,
    LC_INSPECTION_OVERFLOW = 13,
    LC_STEP_LIMIT = 14,
    LC_RHS_EVALUATION_LIMIT = 15,
    LC_IMAGE_INVALID = 16,
    LC_IMAGE_INCOMPATIBLE = 17,
    LC_IMAGE_CHECKSUM_MISMATCH = 18
} lc_status;

#define LC_ROOT_MAX_ITERATIONS 192U
#define LC_ANALYTICAL_MAX_STATES 8U
#if LACUNA_REAL_BITS == 16
#define LC_ABI_VERSION 19U
#define LC_COMPILED_GRAPH_IMAGE_VERSION 2U
#elif LACUNA_REAL_BITS == 64 && LACUNA_TIME_BITS == 64
#define LC_ABI_VERSION 17U
#define LC_COMPILED_GRAPH_IMAGE_VERSION 1U
#else
#define LC_ABI_VERSION 18U
#define LC_COMPILED_GRAPH_IMAGE_VERSION 2U
#endif

typedef enum lc_dispatch_form {
    LC_REACTIVE = 0,
    LC_CLOSED_FORM = 1,
    LC_ROOT_FIND = 2,
    LC_STEPPED = 3
} lc_dispatch_form;

/* Polarity selects fixed-sign magnitudes or explicit signed edge weights. */
typedef enum lc_neuron_polarity {
    LC_EXCITATORY = 0,
    LC_INHIBITORY = 1,
    LC_MIXED = 2
} lc_neuron_polarity;

/* Homogeneous scalar LIF graph descriptors. */
typedef struct lc_scalar_lif_model {
    lc_real_t a;
    lc_real_t b;
    lc_real_t threshold;
    lc_real_t reset;
    lc_time_t refractory;
    uint32_t polarity;
} lc_scalar_lif_model;

typedef struct lc_scalar_state {
    lc_real_t value;
    lc_time_t t_last;
} lc_scalar_state;

typedef struct lc_delta_edge {
    uint32_t pre;
    uint32_t post;
    lc_real_t weight;
    lc_time_t delay;
} lc_delta_edge;

typedef struct lc_input_spike {
    lc_time_t t;
    uint32_t node;
    lc_real_t value;
} lc_input_spike;

typedef struct lc_drive_update {
    lc_time_t t;
    uint32_t node;
    lc_real_t b;
} lc_drive_update;

typedef struct lc_output_spike {
    lc_time_t t;
    uint32_t node;
} lc_output_spike;

/* Host-side encoders convert normalized presentations into input events. */
typedef enum lc_encoder_kind {
    LC_ENCODER_NATIVE_EVENT = 0,
    LC_ENCODER_REGULAR_RATE = 1,
    LC_ENCODER_POISSON_RATE = 2,
    LC_ENCODER_TTFS = 3,
    LC_ENCODER_BURST = 4,
    LC_ENCODER_LATENCY_BURST = 5,
    LC_ENCODER_HELD_CURRENT = 6
} lc_encoder_kind;

typedef struct lc_encoder_spec {
    uint32_t kind;
    uint64_t stream;
    lc_real_t amplitude;
    lc_real_t rate_min;
    lc_real_t rate_max;
    lc_time_t latency_min;
    lc_time_t latency_max;
    lc_time_t duration;
    lc_real_t silence_threshold;
    lc_real_t gain;
    lc_real_t offset;
    lc_real_t baseline;
} lc_encoder_spec;

typedef struct lc_encoder_state {
    lc_real_t phase_remaining;
    lc_real_t poisson_remaining;
    lc_time_t last_end;
    uint64_t draw_index;
    uint64_t seed;
    uint32_t initialized;
    uint32_t poisson_ready;
} lc_encoder_state;

typedef struct lc_presentation {
    lc_time_t t_start;
    lc_time_t t_end;
    uint32_t encoder;
    lc_real_t value;
} lc_presentation;

typedef struct lc_encoded_spike {
    lc_time_t t;
    uint32_t encoder;
    lc_real_t value;
} lc_encoded_spike;

typedef struct lc_encoded_drive {
    lc_time_t t;
    uint32_t encoder;
    lc_real_t value;
} lc_encoded_drive;

/* Decoders observe output spikes without changing network state. */
typedef enum lc_decoder_kind {
    LC_DECODER_RATE = 0,
    LC_DECODER_TTFS = 1,
    LC_DECODER_TEMPORAL_WEIGHT = 2
} lc_decoder_kind;

typedef enum lc_rate_mode {
    LC_RATE_FINITE = 0,
    LC_RATE_SLIDING = 1,
    LC_RATE_CUMULATIVE = 2
} lc_rate_mode;

typedef enum lc_emission_policy {
    LC_EMIT_ON_EVENT = 0,
    LC_EMIT_ON_WINDOW_CLOSE = 1,
    LC_EMIT_ON_EVENT_AND_WINDOW_CLOSE = 2,
    LC_EMIT_ON_QUERY = 3
} lc_emission_policy;

typedef enum lc_decoded_event_kind {
    LC_DECODE_UPDATE = 0,
    LC_DECODE_FINAL = 1,
    LC_DECODE_NO_SPIKE = 2,
    LC_DECODE_QUERY = 3
} lc_decoded_event_kind;

typedef struct lc_decoder_spec {
    uint32_t kind;
    uint32_t node;
    uint32_t mode;
    uint32_t first_only;
    uint32_t normalize;
    uint32_t emission;
    lc_time_t width;
    lc_time_t origin;
    lc_real_t tau;
} lc_decoder_spec;

typedef struct lc_decode_window {
    lc_time_t t_start;
    lc_time_t t_end;
} lc_decode_window;

typedef struct lc_decoder_window_binding {
    uint32_t decoder;
    uint32_t window;
    lc_time_t t_start;
    lc_time_t t_end;
} lc_decoder_window_binding;

typedef struct lc_decoder_query_binding {
    uint32_t decoder;
    uint32_t window;
    lc_time_t t;
} lc_decoder_query_binding;

typedef struct lc_decode_result {
    uint32_t decoder;
    uint32_t window;
    uint32_t valid;
    uint64_t count;
    lc_real_t value;
    lc_time_t first_spike;
    lc_time_t window_start;
    lc_time_t window_end;
} lc_decode_result;

typedef struct lc_decoded_event {
    uint32_t decoder;
    uint32_t window;
    uint32_t kind;
    uint32_t valid;
    uint32_t has_source;
    uint32_t has_first_spike;
    uint64_t count;
    lc_time_t emitted_at;
    lc_time_t source_spike_time;
    lc_time_t window_start;
    lc_time_t window_end;
    lc_time_t observed_through;
    lc_real_t value;
    lc_time_t first_spike;
} lc_decoded_event;

typedef struct lc_run_config {
    lc_time_t t_end;
    uint64_t queue_capacity;
    uint64_t output_capacity;
    uint32_t same_time_cascade_limit;
    uint64_t stochastic_seed;
} lc_run_config;

typedef struct lc_run_stats {
    uint64_t events_popped;
    uint64_t stale_predictions;
    uint64_t peak_queue_occupancy;
    uint64_t deliveries_scheduled;
    uint64_t deliveries_processed;
    uint64_t input_spikes_processed;
    uint64_t drive_updates_processed;
    uint64_t autonomous_spikes_confirmed;
    uint64_t output_spikes;
    uint64_t refractory_releases_processed;
    uint32_t max_same_time_cascade_depth;
    lc_profile_t kernel_seconds; /* Zero when native profiling is disabled. */
} lc_run_stats;

/* Trace records expose causal event phases and optional state snapshots. */
typedef enum lc_trace_kind {
    LC_TRACE_INPUT_SPIKE = 0,
    LC_TRACE_DELIVERY = 1,
    LC_TRACE_DRIVE_UPDATE = 2,
    LC_TRACE_REFRACTORY_RELEASE = 3,
    LC_TRACE_STALE_PREDICTION = 4,
    LC_TRACE_PREDICTION_CONFIRMED = 5,
    LC_TRACE_DEPOSIT_APPLY = 6,
    LC_TRACE_SPIKE = 7,
    LC_TRACE_RESET = 8,
    LC_TRACE_REFRACTORY_ENTER = 9,
    LC_TRACE_FINAL_STATE = 10,
    LC_TRACE_MODULATION = 11,
    LC_TRACE_NUMERICAL_CONTINUATION = 12,
    LC_TRACE_KIND_COUNT = 13
} lc_trace_kind;

typedef enum lc_trace_phase {
    LC_TRACE_PHASE_BOUNDARY = 0,
    LC_TRACE_PHASE_DEPOSIT = 1,
    LC_TRACE_PHASE_PREDICTION = 2,
    LC_TRACE_PHASE_FIRE = 3,
    LC_TRACE_PHASE_FINAL = 4
} lc_trace_phase;

typedef struct lc_trace_record {
    lc_time_t t;
    uint64_t sequence;
    uint64_t generation;
    uint32_t kind;
    uint32_t phase;
    uint32_t node;
    uint32_t subject;
    uint32_t state_count;
    lc_real_t value;
    lc_real_t before[LC_ANALYTICAL_MAX_STATES];
    lc_real_t after[LC_ANALYTICAL_MAX_STATES];
} lc_trace_record;

typedef lc_status (*lc_trace_consumer)(
    const lc_trace_record *record,
    void *context
);

typedef struct lc_trace_config {
    uint64_t kind_mask;
    const uint8_t *node_mask; /* NULL selects every node. */
    uint32_t node_mask_count;
    uint32_t capture_state;
    lc_trace_record *records;
    uint64_t capacity;
    uint64_t *count;
    lc_trace_consumer consumer;
    void *consumer_context;
    uint64_t sequence_base;
} lc_trace_config;

typedef struct lc_state_inspection_request {
    lc_time_t t;
    uint32_t node;
} lc_state_inspection_request;

typedef struct lc_state_inspection_result {
    lc_time_t t;
    uint64_t generation;
    uint32_t node;
    uint32_t state_count;
    uint32_t clamped;
    lc_real_t values[LC_ANALYTICAL_MAX_STATES];
} lc_state_inspection_result;

typedef struct lc_state_inspection_config {
    const lc_state_inspection_request *requests;
    uint32_t request_count;
    lc_state_inspection_result *results;
    uint64_t capacity;
    uint64_t *count;
} lc_state_inspection_config;

/* Expression nodes form a topologically ordered equation program. */
typedef enum lc_expr_op {
    LC_EXPR_CONST = 0,
    LC_EXPR_PARAM = 1,
    LC_EXPR_VAR = 2,
    LC_EXPR_NEG = 3,
    LC_EXPR_ADD = 4,
    LC_EXPR_SUB = 5,
    LC_EXPR_MUL = 6,
    LC_EXPR_DIV = 7,
    LC_EXPR_POW = 8,
    LC_EXPR_EXP = 9,
    LC_EXPR_LOG = 10,
    LC_EXPR_PHI1 = 11,
    LC_EXPR_PHI1_DERIV = 12,
    LC_EXPR_SIN = 13,
    LC_EXPR_COS = 14,
    LC_EXPR_TANH = 15,
    LC_EXPR_MAX = 16
} lc_expr_op;

typedef struct lc_expr_node {
    uint32_t op;
    uint32_t lhs;
    uint32_t rhs;
    uint32_t binding;
    lc_real_t value;
} lc_expr_node;

/* Crossing hints bind expression roots used by analytical solvers. */
typedef struct lc_root_hint {
    uint32_t g_root;
    uint32_t g_prime_root;
    uint32_t extremum_root;
    uint32_t extremum_prime_root;
    uint32_t asymptote_root;
    uint32_t membrane_coefficient_root;
    uint32_t synapse_constant_root;
    uint32_t synapse_linear_root;
    uint32_t membrane_rate_root;
    uint32_t synapse_rate_root;
    uint32_t threshold_root;
    lc_real_t relative_tolerance;
    lc_time_t fastest_time_constant;
} lc_root_hint;

typedef struct lc_two_exp_hint {
    uint32_t g_root;
    uint32_t g_prime_root;
    uint32_t limit_root;
    uint32_t coefficient_one_root;
    uint32_t coefficient_two_root;
    uint32_t rate_one_root;
    uint32_t rate_two_root;
    lc_real_t relative_tolerance;
    lc_time_t fastest_time_constant;
} lc_two_exp_hint;

typedef struct lc_scalar_log_hint {
    uint32_t decay_root;
    uint32_t affine_root;
    uint32_t threshold_root;
} lc_scalar_log_hint;

typedef struct lc_multi_exp_hint {
    uint32_t limit_root;
    uint32_t coefficient_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t rate_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t mode_count;
    uint32_t iteration_cap;
    lc_real_t relative_tolerance;
    lc_time_t fastest_time_constant;
} lc_multi_exp_hint;

/* Flattened P_k(t) coefficients for limit + sum P_k(t) exp(rate_k t). */
typedef struct lc_exp_poly_hint {
    uint32_t limit_root;
    uint32_t rate_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t coefficient_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t coefficient_offsets[LC_ANALYTICAL_MAX_STATES + 1U];
    uint32_t block_count;
    uint32_t coefficient_count;
    uint32_t iteration_cap;
    lc_real_t relative_tolerance;
    lc_time_t fastest_time_constant;
} lc_exp_poly_hint;

typedef struct lc_root_result {
    lc_time_t t_spike;
    lc_time_t horizon;
    lc_time_t bracket_low;
    lc_time_t bracket_high;
    lc_real_t residual;
    lc_time_t tolerance;
    uint32_t iterations;
    uint32_t extrema_count;
} lc_root_result;

typedef struct lc_step_config {
    lc_real_t relative_tolerance;
    lc_real_t absolute_tolerance;
    lc_time_t initial_step;
    lc_time_t minimum_step;
    lc_time_t maximum_step;
    lc_time_t event_tolerance;
    uint32_t maximum_steps;
    uint32_t maximum_rhs_evaluations;
} lc_step_config;

typedef struct lc_step_result {
    lc_time_t t_reached;
    lc_time_t t_crossing;
    lc_time_t last_step;
    lc_real_t error_norm;
    uint32_t accepted_steps;
    uint32_t rejected_steps;
    uint32_t rhs_evaluations;
    uint32_t event_iterations;
} lc_step_result;

/* Crossing kind selects the exact or controlled numerical execution method. */
typedef enum lc_crossing_kind {
    LC_CROSSING_SCALAR_LOG = 0,
    LC_CROSSING_ALPHA_REAL = 1,
    LC_CROSSING_TWO_REAL_EXP = 2,
    LC_CROSSING_REPEATED_REAL_MODE = 3,
    LC_CROSSING_MULTI_REAL_EXP = 4,
    LC_CROSSING_MULTI_EXP_POLY = 5,
    LC_CROSSING_NUMERICAL = 6,
    LC_CROSSING_REACTIVE = 7,
    LC_CROSSING_INTEGRATED_HAZARD = 8
} lc_crossing_kind;

typedef enum lc_hazard_kind {
    LC_HAZARD_NONE = 0,
    LC_HAZARD_EXPONENTIAL_VOLTAGE = 1
} lc_hazard_kind;

typedef struct lc_hazard_config {
    uint32_t kind;
    uint32_t trajectory_mode_count;
    uint32_t trajectory_limit_root;
    uint32_t trajectory_coefficient_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t trajectory_rate_roots[LC_ANALYTICAL_MAX_STATES];
    lc_real_t log_scale;
    lc_real_t voltage_gain;
    lc_real_t relative_tolerance;
    lc_real_t absolute_tolerance;
    lc_time_t time_tolerance;
    uint32_t maximum_quadrature_depth;
    uint32_t maximum_root_iterations;
} lc_hazard_config;

typedef enum lc_deposit_kind {
    LC_DEPOSIT_STATE_ADD = 0,
    LC_DEPOSIT_PROGRAM = 1,
    LC_DEPOSIT_FOLDED_ALPHA = LC_DEPOSIT_PROGRAM
} lc_deposit_kind;

typedef enum lc_node_arithmetic_kind {
    LC_NODE_ARITHMETIC_EXPRESSIONS = 0,
    LC_NODE_ARITHMETIC_SCALAR_AFFINE = 1
} lc_node_arithmetic_kind;

typedef struct lc_mixed_node {
    uint32_t dispatch;
    uint32_t crossing_kind;
    uint32_t state_offset;
    uint32_t state_count;
    uint32_t readout;
    uint32_t parameter_offset;
    uint32_t parameter_count;
    lc_real_t threshold;
    lc_time_t refractory;
    const lc_expr_node *program_nodes;
    uint32_t program_node_count;
    uint32_t normal_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t clamped_roots[LC_ANALYTICAL_MAX_STATES];
    uint32_t reset_roots[LC_ANALYTICAL_MAX_STATES];
    lc_scalar_log_hint scalar_log_hint;
    lc_root_hint root_hint;
    lc_two_exp_hint two_exp_hint;
    lc_multi_exp_hint multi_exp_hint;
    lc_exp_poly_hint exp_poly_hint;
    lc_step_config step_config;
    const lc_expr_node *deposit_nodes;
    uint32_t deposit_node_count;
    uint32_t deposit_root;
    uint32_t deposit_target;
    uint32_t polarity;
    uint32_t reset_before_deposit;
    uint32_t arithmetic_kind;
    lc_real_t scalar_affine_decay;
    lc_real_t scalar_affine_drive;
    lc_real_t scalar_affine_reset;
    lc_hazard_config hazard;
} lc_mixed_node;

/* Typed neurons use magnitudes. Mixed neurons use signed edge weights. */
typedef struct lc_mixed_edge {
    uint32_t pre;
    uint32_t post;
    uint32_t deposit_kind;
    uint32_t target;
    lc_real_t weight;
    lc_time_t delay;
    lc_real_t deposit_scale;
} lc_mixed_edge;

typedef enum lc_plasticity_kind {
    LC_PLASTICITY_NONE = 0,
    LC_PLASTICITY_PAIR = 1,
    LC_PLASTICITY_TRIPLET = 2,
    LC_PLASTICITY_MODULATED = 3
} lc_plasticity_kind;

/* Edge-aligned immutable rule descriptor. Unused fields must be zero. */
typedef struct lc_plasticity_rule {
    uint32_t kind;
    uint32_t modulator;
    uint32_t consume_on_modulation;
    /* Zero means edge-local. Positive values identify a shared weight group. */
    uint32_t weight_group;
    lc_real_t tau_pre;
    lc_real_t tau_post;
    lc_real_t tau_pre_slow;
    lc_real_t tau_post_slow;
    lc_real_t a2_plus;
    lc_real_t a2_minus;
    lc_real_t a3_plus;
    lc_real_t a3_minus;
    lc_real_t tau_eligibility_plus;
    lc_real_t tau_eligibility_minus;
    lc_real_t positive_plus;
    lc_real_t positive_minus;
    lc_real_t negative_plus;
    lc_real_t negative_minus;
    lc_real_t learning_rate;
    lc_real_t weight_min;
    lc_real_t weight_max;
} lc_plasticity_rule;

typedef struct lc_modulation_event {
    lc_time_t t;
    uint32_t modulator;
    lc_real_t value;
} lc_modulation_event;

typedef struct lc_plasticity_state {
    uint32_t edge;
    uint32_t kind;
    lc_real_t weight;
    lc_real_t pre_fast;
    lc_real_t post_fast;
    lc_real_t pre_slow;
    lc_real_t post_slow;
    lc_real_t eligibility_plus;
    lc_real_t eligibility_minus;
    lc_time_t t_pre_fast;
    lc_time_t t_post_fast;
    lc_time_t t_pre_slow;
    lc_time_t t_post_slow;
    lc_time_t t_eligibility_plus;
    lc_time_t t_eligibility_minus;
} lc_plasticity_state;

/*
 * Read-only neuron-local state retained by an equation-derived learning
 * observer. ``sensitivity`` is the cumulative sum of the observer's local
 * gain over the current episode. It is cleared by the episode reset together
 * with the voltage/activity traces, while learned weights are preserved.
 */
typedef struct lc_learning_observer_snapshot {
    uint32_t node;
    uint32_t active;
    lc_real_t slow_voltage;
    lc_real_t fast_activity;
    lc_real_t slow_activity;
    lc_real_t sensitivity;
    lc_time_t t_activity;
} lc_learning_observer_snapshot;

/*
 * Equation-derived edge-learning programs. These descriptors deliberately
 * describe traces and event maps rather than named plasticity models. The
 * base variable layout is [weight, modulation, learning_scale,
 * post_readout, trace[0], ..., trace[trace_count-1]]. Programs with a
 * neuron-local observer insert [observation_gain, event_amplitude] before the
 * edge-local traces. Either layout may additionally insert input_accepted
 * immediately before the traces. During PRE_SPIKE this is 0 when a hard clamp
 * discards the deposit to the membrane readout, and 1 otherwise; it is 0 for
 * other learning events. Deposits to non-clamped state remain accepted. This
 * flag describes the executed deposit, not a spike surrogate or full state
 * sensitivity. Rules that do not reference it retain their arrival semantics.
 * UINT32_MAX denotes an absent root, program, or modulator.
 */
#define LC_LEARNING_EVENT_COUNT 5U
#define LC_LEARNING_MAX_TRACES 6U
/* Maximum base layout, including the observer and optional input context. */
#define LC_LEARNING_BASE_VARIABLE_COUNT 7U

typedef enum lc_learning_event_kind {
    LC_LEARNING_PRE_SPIKE = 0,
    LC_LEARNING_POST_SPIKE = 1,
    LC_LEARNING_MODULATION_POSITIVE = 2,
    LC_LEARNING_MODULATION_NEGATIVE = 3,
    LC_LEARNING_OBSERVATION = 4
} lc_learning_event_kind;

typedef struct lc_learning_event_program {
    const lc_expr_node *nodes;
    uint32_t node_count;
    uint32_t advance_mask;
    uint32_t variable_mask;
    uint32_t weight_root;
    uint32_t trace_update_count;
    uint32_t trace_indices[LC_LEARNING_MAX_TRACES];
    uint32_t trace_roots[LC_LEARNING_MAX_TRACES];
} lc_learning_event_program;

/*
 * Optional neuron-local observer shared by all qualifying incoming edges.
 * Its variable layout is [post_readout, threshold, event_amplitude,
 * slow_voltage, fast_activity, slow_activity].
 */
typedef struct lc_learning_observer_program {
    const lc_expr_node *nodes;
    uint32_t node_count;
    uint32_t variable_mask;
    uint32_t parameter_mask;
    uint32_t voltage_tau_parameter;
    uint32_t fast_activity_tau_parameter;
    uint32_t slow_activity_tau_parameter;
    uint32_t band_width_parameter;
    uint32_t fast_activity_root;
    uint32_t slow_activity_root;
    uint32_t gain_root;
} lc_learning_observer_program;

typedef struct lc_learning_program {
    uint32_t parameter_count;
    uint32_t variable_count;
    uint32_t trace_count;
    uint32_t clamp_normalized_weight;
    /* Observer-only label used by the compatibility plasticity-state API. */
    uint32_t compatibility_kind;
    uint32_t trace_tau_parameters[LC_LEARNING_MAX_TRACES];
    uint32_t trace_storage_slots[LC_LEARNING_MAX_TRACES];
    lc_learning_event_program events[LC_LEARNING_EVENT_COUNT];
    lc_learning_observer_program observer;
} lc_learning_program;

typedef struct lc_learning_binding {
    uint32_t program;
    uint32_t parameter_offset;
    uint32_t modulator;
    /* Zero means edge-local. Positive values identify a shared weight group. */
    uint32_t weight_group;
    lc_real_t weight_min;
    lc_real_t weight_max;
} lc_learning_binding;

typedef struct lc_mixed_input_spike {
    lc_time_t t;
    uint32_t node;
    uint32_t deposit_kind;
    uint32_t target;
    lc_real_t value;
} lc_mixed_input_spike;

typedef struct lc_mixed_drive_update {
    lc_time_t t;
    uint32_t node;
    uint32_t binding;
    lc_real_t value;
} lc_mixed_drive_update;

typedef enum lc_network_event_kind {
    LC_EVENT_REFRACTORY_RELEASE = 0,
    LC_EVENT_DRIVE_UPDATE = 1,
    LC_EVENT_DELIVERY = 2,
    LC_EVENT_INPUT_SPIKE = 3,
    LC_EVENT_AUTONOMOUS_SPIKE = 4,
    LC_EVENT_OUTPUT_SPIKE = 5,
    LC_EVENT_DECODER_EVENT = 6,
    LC_EVENT_MODULATION = 7,
    LC_EVENT_NONE = 8,
    LC_EVENT_NUMERICAL_CONTINUATION = 9
} lc_network_event_kind;

typedef enum lc_network_event_phase {
    LC_PHASE_BOUNDARY = 0,
    LC_PHASE_DEPOSIT = 1,
    LC_PHASE_PREDICTION = 2,
    LC_PHASE_NONE = 3
} lc_network_event_phase;

typedef enum lc_network_resource {
    LC_RESOURCE_NONE = 0,
    LC_RESOURCE_QUEUE = 1,
    LC_RESOURCE_OUTPUT = 2,
    LC_RESOURCE_DECODER_OUTPUT = 3,
    LC_RESOURCE_TRACE = 4,
    LC_RESOURCE_INSPECTION = 5
} lc_network_resource;

typedef struct lc_network_error {
    uint32_t node;
    lc_time_t t;
    lc_root_result root;
    uint32_t resource;
    uint32_t event_kind;
    uint32_t event_phase;
    uint32_t has_event;
    uint32_t event_index;
    uint64_t capacity;
    uint64_t occupancy;
    uint64_t peak;
} lc_network_error;

typedef struct lc_compiled_graph lc_compiled_graph;
typedef struct lc_mixed_run lc_mixed_run;
typedef struct lc_compiled_decoders lc_compiled_decoders;
typedef struct lc_decoder_run lc_decoder_run;
typedef struct lc_encoder_run lc_encoder_run;

typedef struct lc_compiled_graph_info {
    uint32_t node_count;
    uint32_t state_count;
    uint32_t parameter_count;
    uint32_t edge_count;
    uint32_t plastic_edge_count;
} lc_compiled_graph_info;

typedef struct lc_compiled_node_layout {
    uint32_t state_offset;
    uint32_t state_count;
} lc_compiled_node_layout;

LC_API const char *lc_status_string(lc_status status);
LC_API uint32_t lc_abi_version(void);
LC_API uint64_t lc_sizeof_network_error(void);

/*
 * Integer-only precision handshake, safe before passing any numeric descriptor
 * or calling a function with precision-dependent arguments. Property values
 * describe the loaded library, not the consumer's header settings.
 * Unknown property keys return zero. An unsupported representation
 * has profile zero.
 * Bits report storage width, including padding. Mantissa digits and exponent
 * limits describe the actual representation. Wide guards may vary by platform.
 * Only binary64 model values and time are currently implemented.
 */
LC_API uint32_t lc_numeric_property(uint32_t field);
LC_API lc_status lc_numeric_profile_check(
    uint32_t expected_profile,
    uint32_t expected_real_bits,
    uint32_t expected_time_bits,
    uint32_t expected_arithmetic_revision
);

/* Host-side scalar-to-event encoders. State is caller-owned and reset per run. */
LC_API lc_status lc_encoder_state_reset(
    lc_encoder_state *states,
    uint32_t state_count,
    uint64_t seed
);

LC_API lc_status lc_encode_presentations(
    const lc_encoder_spec *specs,
    uint32_t spec_count,
    lc_encoder_state *states,
    uint32_t state_count,
    const lc_presentation *presentations,
    uint32_t presentation_count,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
);

/*
 * Create a live scalar encoder run at an open simulation frontier. The run
 * owns phase, random-stream, arming, burst, and held-current presentation
 * state. Specifications are copied and may be released after this call.
 */
LC_API lc_status lc_encoder_run_create(
    const lc_encoder_spec *specs,
    uint32_t spec_count,
    uint64_t seed,
    lc_time_t initial_frontier,
    lc_encoder_run **run
);

/*
 * Submit presentations beginning at or after the current frontier and emit
 * primitives before `until`. If `seal` is nonzero, primitives exactly at
 * `until` are also emitted and the run becomes finished. A new presentation
 * replaces the unelapsed portion of the active presentation on its encoder.
 * The call is transactional: output overflow leaves run state unchanged.
 */
LC_API lc_status lc_encoder_run_advance(
    lc_encoder_run *run,
    const lc_presentation *presentations,
    uint32_t presentation_count,
    lc_time_t until,
    uint32_t seal,
    lc_encoded_spike *spikes,
    uint64_t spike_capacity,
    uint64_t *spike_count,
    lc_encoded_drive *drives,
    uint64_t drive_capacity,
    uint64_t *drive_count
);

/*
 * Start a new presentation episode at the current frontier. Active encoder
 * phase/presentation state is cleared while counter-based random streams
 * continue from their next unused draw.
 */
LC_API lc_status lc_encoder_run_reset_episode(lc_encoder_run *run);

LC_API void lc_encoder_run_destroy(lc_encoder_run *run);

/* Host-side spike decoders over a half-open observation window. */
LC_API lc_status lc_decode_spikes(
    const lc_decoder_spec *specs,
    uint32_t spec_count,
    const lc_output_spike *spikes,
    uint64_t spike_count,
    const lc_decode_window *window,
    lc_decode_result *results,
    uint32_t result_count
);

/* Compile immutable decoder specifications and their canonical node index. */
LC_API lc_status lc_decoder_bank_compile(
    const lc_decoder_spec *specs,
    uint32_t spec_count,
    uint32_t node_count,
    lc_compiled_decoders **compiled
);

LC_API void lc_decoder_bank_destroy(lc_compiled_decoders *compiled);

/* Create/reset mutable streaming state for one independent execution. */
LC_API lc_status lc_decoder_run_create(
    lc_compiled_decoders *compiled,
    lc_decoder_run **run
);

/* Configure bounded decoded-event retention. Capacity zero disables retention. */
LC_API lc_status lc_decoder_run_reserve_events(
    lc_decoder_run *run,
    uint64_t capacity
);

LC_API lc_status lc_decoder_run_reset(
    lc_decoder_run *run,
    const lc_decode_window *window
);

/* Reset with one or more windows ordered by nondecreasing end time. */
LC_API lc_status lc_decoder_run_reset_windows(
    lc_decoder_run *run,
    const lc_decode_window *windows,
    uint32_t window_count
);

/* Reset with explicit sparse decoder/window assignments. */
LC_API lc_status lc_decoder_run_reset_schedule(
    lc_decoder_run *run,
    const lc_decoder_window_binding *bindings,
    uint64_t binding_count
);

/* Set exact-time snapshots for decoders configured with ON_QUERY emission. */
LC_API lc_status lc_decoder_run_set_queries(
    lc_decoder_run *run,
    const lc_decoder_query_binding *bindings,
    uint64_t binding_count
);

/* Emit due queries and close due windows before observations at this time. */
LC_API lc_status lc_decoder_run_advance(
    lc_decoder_run *run,
    lc_time_t observed_through
);

/* Advance observation strictly before an open right boundary. */
LC_API lc_status lc_decoder_run_advance_before(
    lc_decoder_run *run,
    lc_time_t observed_before
);

LC_API lc_status lc_decoder_run_consume(
    lc_decoder_run *run,
    const lc_output_spike *spike
);

LC_API lc_status lc_decoder_run_validate_for_nodes(
    const lc_decoder_run *run,
    uint32_t node_count
);

LC_API lc_status lc_decoder_run_validate_for_execution(
    const lc_decoder_run *run,
    uint32_t node_count,
    lc_time_t t_end
);

LC_API lc_status lc_decoder_run_result_count(
    const lc_decoder_run *run,
    uint64_t *result_count
);

LC_API lc_status lc_decoder_run_finalize(
    lc_decoder_run *run,
    lc_decode_result *results,
    uint64_t result_count
);

LC_API lc_status lc_decoder_run_copy_events(
    const lc_decoder_run *run,
    lc_decoded_event *events,
    uint64_t event_capacity,
    uint64_t *event_count
);

/* Inspect bounded decoded-event retention without exposing decoder internals. */
LC_API lc_status lc_decoder_run_event_usage(
    const lc_decoder_run *run,
    uint64_t *event_count,
    uint64_t *event_capacity
);

LC_API void lc_decoder_run_destroy(lc_decoder_run *run);

/* Advance dx/dt = a*x + b from state->t_last to t. */
LC_API lc_status lc_scalar_advance(
    const lc_scalar_lif_model *model,
    lc_scalar_state *state,
    lc_time_t t
);

/*
 * Predict the next continuous rising threshold crossing from the current state.
 * LC_NO_CROSSING returns LC_REACTIVE and writes no scheduled time.
 */
LC_API lc_status lc_scalar_predict(
    const lc_scalar_lif_model *model,
    const lc_scalar_state *state,
    lc_time_t *t_spike,
    lc_dispatch_form *dispatch
);

/*
 * Run a flat scalar-LIF graph with delta edges and external input spikes.
 * states is both input and output. Output spikes are written chronologically.
 */
LC_API lc_status lc_delta_network_run(
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
);

/*
 * Prepare an immutable mixed graph. The compiled graph owns deep copies of all
 * descriptors, expression programs, parameters, edges, and its CSR index.
 */
LC_API lc_status lc_mixed_graph_compile(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    uint32_t edge_count,
    lc_compiled_graph **compiled
);

LC_API lc_status lc_mixed_graph_compile_plastic(
    const lc_mixed_node *nodes,
    uint32_t node_count,
    uint32_t state_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_mixed_edge *edges,
    const lc_plasticity_rule *plasticity,
    uint32_t edge_count,
    lc_compiled_graph **compiled
);

LC_API lc_status lc_mixed_graph_compile_learning(
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
);

LC_API void lc_mixed_graph_destroy(lc_compiled_graph *compiled);

/* Measure and serialize one immutable compiled graph into a portable image. */
LC_API lc_status lc_compiled_graph_image_size(
    const lc_compiled_graph *compiled,
    uint64_t *size
);

LC_API lc_status lc_compiled_graph_serialize(
    const lc_compiled_graph *compiled,
    uint8_t *destination,
    uint64_t capacity,
    uint64_t *written
);

/* Reconstruct the ordinary runtime graph without compiling model equations. */
LC_API lc_status lc_compiled_graph_deserialize(
    const uint8_t *image,
    uint64_t size,
    lc_compiled_graph **compiled
);

LC_API lc_status lc_compiled_graph_get_info(
    const lc_compiled_graph *compiled,
    lc_compiled_graph_info *info
);

LC_API lc_status lc_compiled_graph_copy_node_layouts(
    const lc_compiled_graph *compiled,
    lc_compiled_node_layout *layouts,
    uint32_t layout_count
);

/* Create or reset independent mutable state for a compiled graph. */
LC_API lc_status lc_mixed_run_create(
    lc_compiled_graph *compiled,
    const lc_real_t *initial_state,
    uint32_t state_count,
    const lc_time_t *t_last,
    uint32_t node_count,
    lc_mixed_run **run
);

LC_API lc_status lc_mixed_run_reset(
    lc_mixed_run *run,
    const lc_real_t *initial_state,
    uint32_t state_count,
    const lc_time_t *t_last,
    uint32_t node_count
);

/*
 * Begin a resumable execution with a fixed final horizon. Subsequent open
 * advances process timestamps strictly before their boundary. The sealing
 * advance processes the final boundary and emits final state records.
 */
LC_API lc_status lc_mixed_run_begin_incremental(
    lc_mixed_run *run,
    const lc_run_config *config,
    lc_network_error *error
);

LC_API lc_status lc_mixed_run_advance_incremental(
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
);

/*
 * Restore authored neuronal state at the current incremental frontier, clear
 * queued events and all plastic traces, and preserve learned edge weights.
 */
LC_API lc_status lc_mixed_run_reset_episode(
    lc_mixed_run *run,
    const lc_real_t *initial_state,
    uint32_t state_count,
    lc_network_error *error
);

LC_API lc_status lc_mixed_run_execute(
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
);

/* Execute with an optional bounded buffer or streaming causal trace sink. */
LC_API lc_status lc_mixed_run_execute_recorded(
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
);

/* Execute while streaming each emitted spike into a C decoder run. */
LC_API lc_status lc_mixed_run_execute_with_decoders(
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
);

LC_API lc_status lc_mixed_run_execute_with_decoders_recorded(
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
);

/*
 * Execute with any combination of decoder, causal trace, and explicit-time
 * state-inspection observers. Observer pointers are independently nullable.
 * Inspections are read-only and observe the settled post-event state at their
 * timestamp. Request times must be nondecreasing at this C boundary.
 */
LC_API lc_status lc_mixed_run_execute_observed(
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
);

LC_API lc_status lc_mixed_run_copy_state(
    const lc_mixed_run *run,
    lc_real_t *state,
    uint32_t state_count,
    lc_time_t *t_last,
    uint32_t node_count
);

/* Schedule boundary-phase third-factor events before the relevant advance. */
LC_API lc_status lc_mixed_run_schedule_modulations(
    lc_mixed_run *run,
    const lc_modulation_event *events,
    uint32_t event_count
);

/* Inspect edge weights or complete plastic observer state. */
LC_API lc_status lc_mixed_run_copy_weights(
    const lc_mixed_run *run,
    lc_real_t *weights,
    uint32_t edge_count
);

LC_API lc_status lc_mixed_run_copy_plasticity(
    const lc_mixed_run *run,
    lc_plasticity_state *states,
    uint32_t capacity,
    uint32_t *count
);

LC_API lc_status lc_mixed_run_copy_learning_observers(
    const lc_mixed_run *run,
    lc_learning_observer_snapshot *states,
    uint32_t capacity,
    uint32_t *count
);

LC_API void lc_mixed_run_destroy(lc_mixed_run *run);

/* Compatibility wrapper that prepares, executes, and destroys a temporary graph. */
LC_API lc_status lc_mixed_network_run(
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
);

LC_API lc_status lc_mixed_network_run_recorded(
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
);

/* Evaluate a topologically ordered expression DAG into caller-owned workspace. */
LC_API lc_status lc_expr_evaluate(
    const lc_expr_node *nodes,
    uint32_t node_count,
    const lc_real_t *parameters,
    uint32_t parameter_count,
    const lc_real_t *variables,
    uint32_t variable_count,
    lc_real_t *workspace,
    uint32_t workspace_count
);

/*
 * Evaluate only the dependency closure of selected roots. Unreachable DAG nodes
 * and their variable bindings are deliberately not inspected.
 */
LC_API lc_status lc_expr_evaluate_selected(
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
);

/*
 * Advance a flat analytical state using expression roots produced by the resolver.
 * The expression variable layout is [Delta, state[0], ..., state[state_count-1]].
 * Scratch buffers are caller-owned so this entry point remains allocation-free.
 */
LC_API lc_status lc_expr_state_advance(
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
);

/* Apply a simultaneous state map stored as roots in an analytical program. */
LC_API lc_status lc_expr_state_map(
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
);

/* Evaluate one weight-dependent deposit root and add it to a state component. */
LC_API lc_status lc_expr_state_deposit(
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
);

/*
 * Advance a generic ODE state with an adaptive Dormand-Prince 5(4) method.
 * The expression variable layout is [time, state[0], ..., state[state_count-1]].
 */
LC_API lc_status lc_expr_step_advance(
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
);

/* Replay bounded network prediction/dense output from an event anchor to t.
 * horizon is the original simulation end, not the requested sample time.
 * There must be no reset, deposit or parameter change between the anchor and t.
 * Reduced-precision and clamped nodes use the ordinary advance path. */
LC_API lc_status lc_expr_step_replay(
    const lc_expr_node *nodes, uint32_t node_count,
    const lc_real_t *parameters, uint32_t parameter_count,
    const uint32_t *rhs_roots, uint32_t state_count, uint32_t readout,
    const lc_step_config *config, lc_real_t *state, lc_time_t *t_last,
    lc_time_t t, uint32_t clamped, lc_real_t *variables, uint32_t variable_count,
    lc_real_t *workspace, uint32_t workspace_count, lc_step_result *result,
    lc_real_t threshold, lc_time_t horizon
);

/* The v2 replay policy preserves adaptive step size and FSAL derivatives. */
LC_API lc_status lc_expr_step_replay_v2(
    const lc_expr_node *nodes, uint32_t node_count,
    const lc_real_t *parameters, uint32_t parameter_count,
    const uint32_t *rhs_roots, uint32_t state_count, uint32_t readout,
    const lc_step_config *config, lc_real_t *state, lc_time_t *t_last,
    lc_time_t t, uint32_t clamped, lc_real_t *variables, uint32_t variable_count,
    lc_real_t *workspace, uint32_t workspace_count, lc_step_result *result,
    lc_real_t threshold, lc_time_t horizon
);

/* Predict the first rising fixed-threshold crossing without mutating state. */
LC_API lc_status lc_expr_step_predict(
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
);

/* Predict a scalar logarithmic crossing from a generic analytical program. */
LC_API lc_status lc_expr_scalar_log_predict(
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
);

/* Predict the first rising crossing for the certified one-alpha trajectory family. */
LC_API lc_status lc_expr_alpha_predict(
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
);

LC_API lc_status lc_expr_two_exp_predict(
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
);

/* Predict the first rising crossing of a bounded stable real-exponential sum. */
LC_API lc_status lc_expr_multi_exp_predict(
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
);

/* Predict the first rising crossing of a bounded stable exp-polynomial sum. */
LC_API lc_status lc_expr_exp_poly_predict(
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
);

#ifdef __cplusplus
}
#endif

#endif
