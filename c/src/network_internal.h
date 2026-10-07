#ifndef LACUNA_NETWORK_INTERNAL_H
#define LACUNA_NETWORK_INTERNAL_H

#include "lacuna.h"

#include <stdint.h>

typedef struct lc_delivery_group {
    uint64_t edge_offset;
    uint32_t edge_count;
    uint32_t first_edge;
    lc_time_t delay;
} lc_delivery_group;

typedef struct lc_learning_batch {
    uint64_t slot_offset;
    uint32_t slot_count;
    uint32_t program;
} lc_learning_batch;

typedef struct lc_node_eval_plan {
    const lc_expr_node *normal_nodes;
    const lc_expr_node *clamped_nodes;
    const lc_expr_node *reset_nodes;
    const lc_expr_node *crossing_nodes;
    const lc_expr_node *deposit_nodes;
} lc_node_eval_plan;

struct lc_compiled_graph {
    uint64_t references;
    uint32_t node_count;
    uint32_t state_count;
    uint32_t parameter_count;
    uint32_t edge_count;
    uint32_t workspace_count;
    lc_mixed_node *nodes;
    lc_expr_node *expression_nodes;
    lc_expr_node *specialized_expression_nodes;
    lc_node_eval_plan *node_eval_plans;
    lc_real_t *parameters;
    lc_mixed_edge *edges;
    uint32_t delivery_group_count;
    lc_delivery_group *delivery_groups;
    uint32_t *delivery_group_edges;
    uint64_t *outgoing_offsets;
    lc_plasticity_rule *plasticity;
    uint32_t learning_program_count;
    lc_learning_program *learning_programs;
    lc_expr_node *learning_expression_nodes;
    uint32_t learning_parameter_count;
    lc_real_t *learning_parameters;
    lc_learning_binding *learning_bindings;
    uint32_t *learning_observer_program_by_node;
    uint32_t *learning_observer_parameter_offset_by_node;
    uint32_t plastic_edge_count;
    uint32_t *plastic_slot_by_edge;
    uint32_t *plastic_edges;
    uint32_t *weight_master_slots;
    lc_real_t *weight_learning_scales;
    uint64_t *incoming_plastic_offsets;
    uint32_t *incoming_plastic_slots;
    uint32_t incoming_learning_batch_count;
    uint64_t *incoming_learning_batch_offsets;
    lc_learning_batch *incoming_learning_batches;
    uint32_t modulator_count;
    uint64_t *modulator_offsets;
    uint32_t *modulator_slots;
    uint32_t modulation_learning_batch_count;
    uint64_t *modulation_learning_batch_offsets;
    lc_learning_batch *modulation_learning_batches;
};

int lc_compiled_graph_validate_loaded(const lc_compiled_graph *graph);

#endif
