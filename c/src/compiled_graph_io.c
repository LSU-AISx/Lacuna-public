#include "lacuna.h"
#include "network_internal.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Persist compiled graphs without exposing their process-local pointers. */

/* Keep the binary64 version-1 image byte-for-byte compatible. */
#if LACUNA_REAL_BITS == 64 && LACUNA_TIME_BITS == 64
#define LC_IMAGE_HEADER_SIZE UINT64_C(40)
#define LC_IMAGE_VERSION 1U
#define LC_IMAGE_PROFILE LC_NUMERIC_PROFILE_BINARY64
#else
#define LC_IMAGE_HEADER_SIZE UINT64_C(56)
#define LC_IMAGE_VERSION 2U
#if LACUNA_REAL_BITS == 16
#define LC_IMAGE_PROFILE LC_NUMERIC_PROFILE_BINARY16
#elif LACUNA_TIME_BITS == 64
#define LC_IMAGE_PROFILE LC_NUMERIC_PROFILE_BINARY32_TIME64
#else
#define LC_IMAGE_PROFILE LC_NUMERIC_PROFILE_BINARY32
#endif
#endif
#define LC_IMAGE_REAL_SIZE ((uint64_t)(LACUNA_REAL_BITS / 8U))
#define LC_IMAGE_EXPR_SIZE (UINT64_C(16) + LC_IMAGE_REAL_SIZE)
#define LC_IMAGE_NULL_REF UINT64_MAX
#define LC_IMAGE_SPECIALIZED_REF (UINT64_C(1) << 63U)

#define LC_IMAGE_HAS_EVAL_PLANS (UINT64_C(1) << 0U)
#define LC_IMAGE_HAS_LEGACY_PLASTICITY (UINT64_C(1) << 1U)
#define LC_IMAGE_HAS_LEARNING_BINDINGS (UINT64_C(1) << 2U)
#define LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS (UINT64_C(1) << 3U)
#define LC_IMAGE_HAS_PLASTIC_SLOT_MAP (UINT64_C(1) << 4U)
#define LC_IMAGE_HAS_WEIGHT_GROUPS (UINT64_C(1) << 5U)
#define LC_IMAGE_HAS_INCOMING_PLASTIC (UINT64_C(1) << 6U)
#define LC_IMAGE_HAS_INCOMING_BATCHES (UINT64_C(1) << 7U)
#define LC_IMAGE_HAS_MODULATOR_INDEX (UINT64_C(1) << 8U)
#define LC_IMAGE_HAS_MODULATION_BATCHES (UINT64_C(1) << 9U)
#define LC_IMAGE_KNOWN_FLAGS ((UINT64_C(1) << 10U) - 1U)

static const uint8_t lc_image_magic[8] = {
    'L', 'C', 'G', 'I', 'M', 'G', '0', '1'
};

/* No numeric casts, including on strict-float32 targets. */
static int lc_image_numeric_supported(void) {
    lc_real_t real_one = LC_REAL_C(1.0);
    lc_time_t time_one = LC_TIME_C(1.0);
#if LACUNA_REAL_BITS == 16
    uint16_t real_bits = 0U;
#elif LACUNA_REAL_BITS == 32
    uint32_t real_bits = 0U;
#else
    uint64_t real_bits = 0U;
#endif
#if LACUNA_TIME_BITS == 16
    uint16_t time_bits = 0U;
#elif LACUNA_TIME_BITS == 32
    uint32_t time_bits = 0U;
#else
    uint64_t time_bits = 0U;
#endif
    if (lc_numeric_profile_check(
            LC_IMAGE_PROFILE, LACUNA_REAL_BITS, LACUNA_TIME_BITS,
            LC_NUMERIC_ARITHMETIC_REVISION
        ) != LC_OK ||
        sizeof(real_one) != sizeof(real_bits) ||
        sizeof(time_one) != sizeof(time_bits)) return 0;
    memcpy(&real_bits, &real_one, sizeof(real_bits));
    memcpy(&time_bits, &time_one, sizeof(time_bits));
#if LACUNA_REAL_BITS == 16
    if (LC_REAL_MANT_DIG != 11 || LC_REAL_MAX_EXP != 16 ||
        real_bits != UINT16_C(0x3c00)) return 0;
#elif LACUNA_REAL_BITS == 32
    if (FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        real_bits != UINT32_C(0x3f800000)) return 0;
#else
    if (DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 ||
        real_bits != UINT64_C(0x3ff0000000000000)) return 0;
#endif
#if LACUNA_TIME_BITS == 16
    if (LC_TIME_MANT_DIG != 11 || LC_TIME_MAX_EXP != 16 ||
        time_bits != UINT16_C(0x3c00)) return 0;
#elif LACUNA_TIME_BITS == 32
    if (FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 ||
        time_bits != UINT32_C(0x3f800000)) return 0;
#else
    if (DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 ||
        time_bits != UINT64_C(0x3ff0000000000000)) return 0;
#endif
    return FLT_RADIX == 2;
}

typedef struct lc_image_writer {
    uint8_t *data;
    uint64_t capacity;
    uint64_t position;
    int failed;
} lc_image_writer;

typedef struct lc_image_reader {
    const uint8_t *data;
    uint64_t size;
    uint64_t position;
    int failed;
} lc_image_reader;

typedef struct lc_image_expression_block {
    const lc_expr_node *nodes;
    uint32_t count;
    uint64_t offset;
} lc_image_expression_block;

typedef struct lc_image_expression_pool {
    lc_image_expression_block *blocks;
    uint32_t count;
    uint32_t capacity;
    uint64_t expression_count;
} lc_image_expression_pool;

typedef struct lc_image_layout {
    lc_image_expression_pool base;
    lc_image_expression_pool specialized;
    uint64_t learning_expression_count;
    uint64_t flags;
} lc_image_layout;

static int lc_image_allocation_overflows(
    uint64_t count,
    size_t element_size
) {
    return element_size != 0U && count > (uint64_t)(SIZE_MAX / element_size);
}

static void lc_image_write_bytes(
    lc_image_writer *writer,
    const uint8_t *bytes,
    uint64_t count
) {
    if (writer->failed || count > UINT64_MAX - writer->position) {
        writer->failed = 1;
        return;
    }
    if (writer->data != NULL) {
        if (writer->position > writer->capacity ||
            count > writer->capacity - writer->position) {
            writer->failed = 1;
            return;
        }
        if (count > 0U) {
            memcpy(&writer->data[writer->position], bytes, (size_t)count);
        }
    }
    writer->position += count;
}

#if LACUNA_REAL_BITS == 16 || LACUNA_TIME_BITS == 16
static void lc_image_write_u16(lc_image_writer *writer, uint16_t value) {
    uint8_t bytes[2];
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8U);
    lc_image_write_bytes(writer, bytes, sizeof(bytes));
}
#endif

static void lc_image_write_u32(lc_image_writer *writer, uint32_t value) {
    uint8_t bytes[4];
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8U);
    bytes[2] = (uint8_t)(value >> 16U);
    bytes[3] = (uint8_t)(value >> 24U);
    lc_image_write_bytes(writer, bytes, sizeof(bytes));
}

static void lc_image_write_u64(lc_image_writer *writer, uint64_t value) {
    uint8_t bytes[8];
    uint32_t index;
    for (index = 0U; index < 8U; ++index) {
        bytes[index] = (uint8_t)(value >> (index * 8U));
    }
    lc_image_write_bytes(writer, bytes, sizeof(bytes));
}

static void lc_image_write_real(lc_image_writer *writer, lc_real_t value) {
#if LACUNA_REAL_BITS == 16
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u16(writer, bits);
#elif LACUNA_REAL_BITS == 32
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u32(writer, bits);
#else
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u64(writer, bits);
#endif
}

static void lc_image_write_time(lc_image_writer *writer, lc_time_t value) {
#if LACUNA_TIME_BITS == 16
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u16(writer, bits);
#elif LACUNA_TIME_BITS == 32
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u32(writer, bits);
#else
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    lc_image_write_u64(writer, bits);
#endif
}

static void lc_image_read_bytes(
    lc_image_reader *reader,
    uint8_t *bytes,
    uint64_t count
) {
    if (reader->failed || reader->position > reader->size ||
        count > reader->size - reader->position) {
        reader->failed = 1;
        return;
    }
    if (count > 0U) {
        memcpy(bytes, &reader->data[reader->position], (size_t)count);
    }
    reader->position += count;
}

#if LACUNA_REAL_BITS == 16 || LACUNA_TIME_BITS == 16
static uint16_t lc_image_read_u16(lc_image_reader *reader) {
    uint8_t bytes[2] = {0U, 0U};
    lc_image_read_bytes(reader, bytes, sizeof(bytes));
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8U));
}
#endif

static uint32_t lc_image_read_u32(lc_image_reader *reader) {
    uint8_t bytes[4] = {0U, 0U, 0U, 0U};
    lc_image_read_bytes(reader, bytes, sizeof(bytes));
    return (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8U) |
        ((uint32_t)bytes[2] << 16U) |
        ((uint32_t)bytes[3] << 24U);
}

static uint64_t lc_image_read_u64(lc_image_reader *reader) {
    uint8_t bytes[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    uint64_t value = 0U;
    uint32_t index;
    lc_image_read_bytes(reader, bytes, sizeof(bytes));
    for (index = 0U; index < 8U; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8U);
    }
    return value;
}

static lc_real_t lc_image_read_real(lc_image_reader *reader) {
#if LACUNA_REAL_BITS == 16
    uint16_t bits = lc_image_read_u16(reader);
#elif LACUNA_REAL_BITS == 32
    uint32_t bits = lc_image_read_u32(reader);
#else
    uint64_t bits = lc_image_read_u64(reader);
#endif
    lc_real_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static lc_time_t lc_image_read_time(lc_image_reader *reader) {
#if LACUNA_TIME_BITS == 16
    uint16_t bits = lc_image_read_u16(reader);
#elif LACUNA_TIME_BITS == 32
    uint32_t bits = lc_image_read_u32(reader);
#else
    uint64_t bits = lc_image_read_u64(reader);
#endif
    lc_time_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t lc_image_crc32(const uint8_t *data, uint64_t count) {
    uint32_t crc = UINT32_MAX;
    uint64_t position;
    for (position = 0U; position < count; ++position) {
        uint32_t bit;
        crc ^= data[position];
        for (bit = 0U; bit < 8U; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1U) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return ~crc;
}

static void lc_image_patch_u32(uint8_t *data, uint64_t offset, uint32_t value) {
    data[offset] = (uint8_t)value;
    data[offset + 1U] = (uint8_t)(value >> 8U);
    data[offset + 2U] = (uint8_t)(value >> 16U);
    data[offset + 3U] = (uint8_t)(value >> 24U);
}

static void lc_image_pool_release(lc_image_expression_pool *pool) {
    free(pool->blocks);
    memset(pool, 0, sizeof(*pool));
}

static int lc_image_pool_find(
    const lc_image_expression_pool *pool,
    const lc_expr_node *nodes,
    uint32_t count,
    uint64_t *offset
) {
    uint32_t index;
    for (index = 0U; index < pool->count; ++index) {
        if (pool->blocks[index].nodes == nodes &&
            pool->blocks[index].count == count) {
            *offset = pool->blocks[index].offset;
            return 1;
        }
    }
    return 0;
}

static int lc_image_pool_add(
    lc_image_expression_pool *pool,
    const lc_expr_node *nodes,
    uint32_t count,
    uint64_t *offset
) {
    lc_image_expression_block *block;
    uint32_t index;
    if (count == 0U) {
        if (nodes != NULL) return 0;
        *offset = LC_IMAGE_NULL_REF;
        return 1;
    }
    if (nodes == NULL || lc_image_pool_find(pool, nodes, count, offset)) {
        return nodes != NULL;
    }
    for (index = 0U; index < pool->count; ++index) {
        if (pool->blocks[index].nodes == nodes &&
            pool->blocks[index].count != count) {
            return 0;
        }
    }
    if (pool->count >= pool->capacity ||
        pool->expression_count > UINT64_MAX - count) {
        return 0;
    }
    block = &pool->blocks[pool->count++];
    block->nodes = nodes;
    block->count = count;
    block->offset = pool->expression_count;
    *offset = block->offset;
    pool->expression_count += count;
    return 1;
}

static int lc_image_plan_reference(
    lc_image_layout *layout,
    const lc_expr_node *nodes,
    uint32_t count,
    uint64_t *reference
) {
    uint64_t offset;
    if (count == 0U) {
        if (nodes != NULL) return 0;
        *reference = LC_IMAGE_NULL_REF;
        return 1;
    }
    if (nodes == NULL) return 0;
    if (lc_image_pool_find(&layout->base, nodes, count, &offset)) {
        *reference = offset;
        return 1;
    }
    if (!lc_image_pool_add(&layout->specialized, nodes, count, &offset) ||
        offset >= LC_IMAGE_SPECIALIZED_REF) {
        return 0;
    }
    *reference = LC_IMAGE_SPECIALIZED_REF | offset;
    return 1;
}

static int lc_image_frozen_plan_reference(
    const lc_image_layout *layout,
    const lc_expr_node *nodes,
    uint32_t count,
    uint64_t *reference
) {
    uint64_t offset;
    if (count == 0U) {
        if (nodes != NULL) return 0;
        *reference = LC_IMAGE_NULL_REF;
        return 1;
    }
    if (nodes == NULL) return 0;
    if (lc_image_pool_find(&layout->base, nodes, count, &offset)) {
        *reference = offset;
        return 1;
    }
    if (!lc_image_pool_find(&layout->specialized, nodes, count, &offset) ||
        offset >= LC_IMAGE_SPECIALIZED_REF) {
        return 0;
    }
    *reference = LC_IMAGE_SPECIALIZED_REF | offset;
    return 1;
}

static void lc_image_layout_release(lc_image_layout *layout) {
    lc_image_pool_release(&layout->base);
    lc_image_pool_release(&layout->specialized);
    memset(layout, 0, sizeof(*layout));
}

static int lc_image_layout_build(
    const lc_compiled_graph *graph,
    lc_image_layout *layout
) {
    uint64_t base_capacity;
    uint64_t specialized_capacity;
    uint32_t node;
    uint32_t program;
    if (graph == NULL || graph->nodes == NULL || graph->node_count == 0U ||
        graph->state_count == 0U || graph->outgoing_offsets == NULL) {
        return 0;
    }
    memset(layout, 0, sizeof(*layout));
    base_capacity = (uint64_t)graph->node_count * 2U;
    specialized_capacity = (uint64_t)graph->node_count * 5U;
    if (base_capacity > UINT32_MAX || specialized_capacity > UINT32_MAX ||
        lc_image_allocation_overflows(
            base_capacity, sizeof(lc_image_expression_block)
        ) || lc_image_allocation_overflows(
            specialized_capacity, sizeof(lc_image_expression_block)
        )) {
        return 0;
    }
    layout->base.capacity = (uint32_t)base_capacity;
    layout->specialized.capacity = (uint32_t)specialized_capacity;
    layout->base.blocks = calloc(
        layout->base.capacity, sizeof(lc_image_expression_block)
    );
    layout->specialized.blocks = calloc(
        layout->specialized.capacity, sizeof(lc_image_expression_block)
    );
    if (layout->base.blocks == NULL || layout->specialized.blocks == NULL) {
        lc_image_layout_release(layout);
        return 0;
    }
    for (node = 0U; node < graph->node_count; ++node) {
        uint64_t ignored;
        const lc_mixed_node *descriptor = &graph->nodes[node];
        if (!lc_image_pool_add(
                &layout->base, descriptor->program_nodes,
                descriptor->program_node_count, &ignored
            ) || !lc_image_pool_add(
                &layout->base, descriptor->deposit_nodes,
                descriptor->deposit_node_count, &ignored
            )) {
            lc_image_layout_release(layout);
            return 0;
        }
    }
    if (graph->node_eval_plans != NULL) {
        layout->flags |= LC_IMAGE_HAS_EVAL_PLANS;
        for (node = 0U; node < graph->node_count; ++node) {
            const lc_mixed_node *descriptor = &graph->nodes[node];
            const lc_node_eval_plan *plan = &graph->node_eval_plans[node];
            uint64_t ignored;
            if (!lc_image_plan_reference(
                    layout, plan->normal_nodes,
                    descriptor->program_node_count, &ignored
                ) || !lc_image_plan_reference(
                    layout, plan->clamped_nodes,
                    descriptor->program_node_count, &ignored
                ) || !lc_image_plan_reference(
                    layout, plan->reset_nodes,
                    descriptor->program_node_count, &ignored
                ) || !lc_image_plan_reference(
                    layout, plan->crossing_nodes,
                    descriptor->program_node_count, &ignored
                ) || !lc_image_plan_reference(
                    layout, plan->deposit_nodes,
                    descriptor->deposit_node_count, &ignored
                )) {
                lc_image_layout_release(layout);
                return 0;
            }
        }
    }
    for (program = 0U; program < graph->learning_program_count; ++program) {
        uint32_t event;
        const lc_learning_program *item = &graph->learning_programs[program];
        for (event = 0U; event < LC_LEARNING_EVENT_COUNT; ++event) {
            if (layout->learning_expression_count >
                UINT64_MAX - item->events[event].node_count) {
                lc_image_layout_release(layout);
                return 0;
            }
            layout->learning_expression_count +=
                item->events[event].node_count;
        }
        if (layout->learning_expression_count >
            UINT64_MAX - item->observer.node_count) {
            lc_image_layout_release(layout);
            return 0;
        }
        layout->learning_expression_count += item->observer.node_count;
    }
    if (graph->plasticity != NULL)
        layout->flags |= LC_IMAGE_HAS_LEGACY_PLASTICITY;
    if (graph->learning_bindings != NULL ||
        graph->learning_program_count > 0U ||
        graph->learning_observer_program_by_node != NULL ||
        graph->learning_observer_parameter_offset_by_node != NULL)
        layout->flags |= LC_IMAGE_HAS_LEARNING_BINDINGS;
    if (graph->learning_observer_program_by_node != NULL ||
        graph->learning_observer_parameter_offset_by_node != NULL)
        layout->flags |= LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS;
    if (graph->plastic_slot_by_edge != NULL)
        layout->flags |= LC_IMAGE_HAS_PLASTIC_SLOT_MAP;
    if (graph->weight_master_slots != NULL ||
        graph->weight_learning_scales != NULL)
        layout->flags |= LC_IMAGE_HAS_WEIGHT_GROUPS;
    if (graph->incoming_plastic_offsets != NULL ||
        graph->incoming_plastic_slots != NULL)
        layout->flags |= LC_IMAGE_HAS_INCOMING_PLASTIC;
    if (graph->incoming_learning_batch_offsets != NULL ||
        graph->incoming_learning_batches != NULL)
        layout->flags |= LC_IMAGE_HAS_INCOMING_BATCHES;
    if (graph->modulator_offsets != NULL || graph->modulator_slots != NULL)
        layout->flags |= LC_IMAGE_HAS_MODULATOR_INDEX;
    if (graph->modulation_learning_batch_offsets != NULL ||
        graph->modulation_learning_batches != NULL)
        layout->flags |= LC_IMAGE_HAS_MODULATION_BATCHES;
    return 1;
}

static void lc_image_write_expr(
    lc_image_writer *writer,
    const lc_expr_node *node
) {
    lc_image_write_u32(writer, node->op);
    lc_image_write_u32(writer, node->lhs);
    lc_image_write_u32(writer, node->rhs);
    lc_image_write_u32(writer, node->binding);
    lc_image_write_real(writer, node->value);
}

static void lc_image_read_expr(
    lc_image_reader *reader,
    lc_expr_node *node
) {
    node->op = lc_image_read_u32(reader);
    node->lhs = lc_image_read_u32(reader);
    node->rhs = lc_image_read_u32(reader);
    node->binding = lc_image_read_u32(reader);
    node->value = lc_image_read_real(reader);
}

static void lc_image_write_scalar_log_hint(
    lc_image_writer *writer,
    const lc_scalar_log_hint *hint
) {
    lc_image_write_u32(writer, hint->decay_root);
    lc_image_write_u32(writer, hint->affine_root);
    lc_image_write_u32(writer, hint->threshold_root);
}

static void lc_image_read_scalar_log_hint(
    lc_image_reader *reader,
    lc_scalar_log_hint *hint
) {
    hint->decay_root = lc_image_read_u32(reader);
    hint->affine_root = lc_image_read_u32(reader);
    hint->threshold_root = lc_image_read_u32(reader);
}

static void lc_image_write_root_hint(
    lc_image_writer *writer,
    const lc_root_hint *hint
) {
    lc_image_write_u32(writer, hint->g_root);
    lc_image_write_u32(writer, hint->g_prime_root);
    lc_image_write_u32(writer, hint->extremum_root);
    lc_image_write_u32(writer, hint->extremum_prime_root);
    lc_image_write_u32(writer, hint->asymptote_root);
    lc_image_write_u32(writer, hint->membrane_coefficient_root);
    lc_image_write_u32(writer, hint->synapse_constant_root);
    lc_image_write_u32(writer, hint->synapse_linear_root);
    lc_image_write_u32(writer, hint->membrane_rate_root);
    lc_image_write_u32(writer, hint->synapse_rate_root);
    lc_image_write_u32(writer, hint->threshold_root);
    lc_image_write_real(writer, hint->relative_tolerance);
    lc_image_write_time(writer, hint->fastest_time_constant);
}

static void lc_image_read_root_hint(
    lc_image_reader *reader,
    lc_root_hint *hint
) {
    hint->g_root = lc_image_read_u32(reader);
    hint->g_prime_root = lc_image_read_u32(reader);
    hint->extremum_root = lc_image_read_u32(reader);
    hint->extremum_prime_root = lc_image_read_u32(reader);
    hint->asymptote_root = lc_image_read_u32(reader);
    hint->membrane_coefficient_root = lc_image_read_u32(reader);
    hint->synapse_constant_root = lc_image_read_u32(reader);
    hint->synapse_linear_root = lc_image_read_u32(reader);
    hint->membrane_rate_root = lc_image_read_u32(reader);
    hint->synapse_rate_root = lc_image_read_u32(reader);
    hint->threshold_root = lc_image_read_u32(reader);
    hint->relative_tolerance = lc_image_read_real(reader);
    hint->fastest_time_constant = lc_image_read_time(reader);
}

static void lc_image_write_two_exp_hint(
    lc_image_writer *writer,
    const lc_two_exp_hint *hint
) {
    lc_image_write_u32(writer, hint->g_root);
    lc_image_write_u32(writer, hint->g_prime_root);
    lc_image_write_u32(writer, hint->limit_root);
    lc_image_write_u32(writer, hint->coefficient_one_root);
    lc_image_write_u32(writer, hint->coefficient_two_root);
    lc_image_write_u32(writer, hint->rate_one_root);
    lc_image_write_u32(writer, hint->rate_two_root);
    lc_image_write_real(writer, hint->relative_tolerance);
    lc_image_write_time(writer, hint->fastest_time_constant);
}

static void lc_image_read_two_exp_hint(
    lc_image_reader *reader,
    lc_two_exp_hint *hint
) {
    hint->g_root = lc_image_read_u32(reader);
    hint->g_prime_root = lc_image_read_u32(reader);
    hint->limit_root = lc_image_read_u32(reader);
    hint->coefficient_one_root = lc_image_read_u32(reader);
    hint->coefficient_two_root = lc_image_read_u32(reader);
    hint->rate_one_root = lc_image_read_u32(reader);
    hint->rate_two_root = lc_image_read_u32(reader);
    hint->relative_tolerance = lc_image_read_real(reader);
    hint->fastest_time_constant = lc_image_read_time(reader);
}

static void lc_image_write_multi_exp_hint(
    lc_image_writer *writer,
    const lc_multi_exp_hint *hint
) {
    uint32_t index;
    lc_image_write_u32(writer, hint->limit_root);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hint->coefficient_roots[index]);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hint->rate_roots[index]);
    lc_image_write_u32(writer, hint->mode_count);
    lc_image_write_u32(writer, hint->iteration_cap);
    lc_image_write_real(writer, hint->relative_tolerance);
    lc_image_write_time(writer, hint->fastest_time_constant);
}

static void lc_image_read_multi_exp_hint(
    lc_image_reader *reader,
    lc_multi_exp_hint *hint
) {
    uint32_t index;
    hint->limit_root = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hint->coefficient_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hint->rate_roots[index] = lc_image_read_u32(reader);
    hint->mode_count = lc_image_read_u32(reader);
    hint->iteration_cap = lc_image_read_u32(reader);
    hint->relative_tolerance = lc_image_read_real(reader);
    hint->fastest_time_constant = lc_image_read_time(reader);
}

static void lc_image_write_exp_poly_hint(
    lc_image_writer *writer,
    const lc_exp_poly_hint *hint
) {
    uint32_t index;
    lc_image_write_u32(writer, hint->limit_root);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hint->rate_roots[index]);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hint->coefficient_roots[index]);
    for (index = 0U; index <= LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hint->coefficient_offsets[index]);
    lc_image_write_u32(writer, hint->block_count);
    lc_image_write_u32(writer, hint->coefficient_count);
    lc_image_write_u32(writer, hint->iteration_cap);
    lc_image_write_real(writer, hint->relative_tolerance);
    lc_image_write_time(writer, hint->fastest_time_constant);
}

static void lc_image_read_exp_poly_hint(
    lc_image_reader *reader,
    lc_exp_poly_hint *hint
) {
    uint32_t index;
    hint->limit_root = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hint->rate_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hint->coefficient_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index <= LC_ANALYTICAL_MAX_STATES; ++index)
        hint->coefficient_offsets[index] = lc_image_read_u32(reader);
    hint->block_count = lc_image_read_u32(reader);
    hint->coefficient_count = lc_image_read_u32(reader);
    hint->iteration_cap = lc_image_read_u32(reader);
    hint->relative_tolerance = lc_image_read_real(reader);
    hint->fastest_time_constant = lc_image_read_time(reader);
}

static void lc_image_write_step_config(
    lc_image_writer *writer,
    const lc_step_config *config
) {
    lc_image_write_real(writer, config->relative_tolerance);
    lc_image_write_real(writer, config->absolute_tolerance);
    lc_image_write_time(writer, config->initial_step);
    lc_image_write_time(writer, config->minimum_step);
    lc_image_write_time(writer, config->maximum_step);
    lc_image_write_time(writer, config->event_tolerance);
    lc_image_write_u32(writer, config->maximum_steps);
    lc_image_write_u32(writer, config->maximum_rhs_evaluations);
}

static void lc_image_read_step_config(
    lc_image_reader *reader,
    lc_step_config *config
) {
    config->relative_tolerance = lc_image_read_real(reader);
    config->absolute_tolerance = lc_image_read_real(reader);
    config->initial_step = lc_image_read_time(reader);
    config->minimum_step = lc_image_read_time(reader);
    config->maximum_step = lc_image_read_time(reader);
    config->event_tolerance = lc_image_read_time(reader);
    config->maximum_steps = lc_image_read_u32(reader);
    config->maximum_rhs_evaluations = lc_image_read_u32(reader);
}

static void lc_image_write_hazard(
    lc_image_writer *writer,
    const lc_hazard_config *hazard
) {
    uint32_t index;
    lc_image_write_u32(writer, hazard->kind);
    lc_image_write_u32(writer, hazard->trajectory_mode_count);
    lc_image_write_u32(writer, hazard->trajectory_limit_root);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hazard->trajectory_coefficient_roots[index]);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, hazard->trajectory_rate_roots[index]);
    lc_image_write_real(writer, hazard->log_scale);
    lc_image_write_real(writer, hazard->voltage_gain);
    lc_image_write_real(writer, hazard->relative_tolerance);
    lc_image_write_real(writer, hazard->absolute_tolerance);
    lc_image_write_time(writer, hazard->time_tolerance);
    lc_image_write_u32(writer, hazard->maximum_quadrature_depth);
    lc_image_write_u32(writer, hazard->maximum_root_iterations);
}

static void lc_image_read_hazard(
    lc_image_reader *reader,
    lc_hazard_config *hazard
) {
    uint32_t index;
    hazard->kind = lc_image_read_u32(reader);
    hazard->trajectory_mode_count = lc_image_read_u32(reader);
    hazard->trajectory_limit_root = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hazard->trajectory_coefficient_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        hazard->trajectory_rate_roots[index] = lc_image_read_u32(reader);
    hazard->log_scale = lc_image_read_real(reader);
    hazard->voltage_gain = lc_image_read_real(reader);
    hazard->relative_tolerance = lc_image_read_real(reader);
    hazard->absolute_tolerance = lc_image_read_real(reader);
    hazard->time_tolerance = lc_image_read_time(reader);
    hazard->maximum_quadrature_depth = lc_image_read_u32(reader);
    hazard->maximum_root_iterations = lc_image_read_u32(reader);
}

static void lc_image_write_node(
    lc_image_writer *writer,
    const lc_mixed_node *node,
    uint64_t program_reference,
    uint64_t deposit_reference
) {
    uint32_t index;
    lc_image_write_u64(writer, program_reference);
    lc_image_write_u64(writer, deposit_reference);
    lc_image_write_u32(writer, node->dispatch);
    lc_image_write_u32(writer, node->crossing_kind);
    lc_image_write_u32(writer, node->state_offset);
    lc_image_write_u32(writer, node->state_count);
    lc_image_write_u32(writer, node->readout);
    lc_image_write_u32(writer, node->parameter_offset);
    lc_image_write_u32(writer, node->parameter_count);
    lc_image_write_real(writer, node->threshold);
    lc_image_write_time(writer, node->refractory);
    lc_image_write_u32(writer, node->program_node_count);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, node->normal_roots[index]);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, node->clamped_roots[index]);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        lc_image_write_u32(writer, node->reset_roots[index]);
    lc_image_write_scalar_log_hint(writer, &node->scalar_log_hint);
    lc_image_write_root_hint(writer, &node->root_hint);
    lc_image_write_two_exp_hint(writer, &node->two_exp_hint);
    lc_image_write_multi_exp_hint(writer, &node->multi_exp_hint);
    lc_image_write_exp_poly_hint(writer, &node->exp_poly_hint);
    lc_image_write_step_config(writer, &node->step_config);
    lc_image_write_u32(writer, node->deposit_node_count);
    lc_image_write_u32(writer, node->deposit_root);
    lc_image_write_u32(writer, node->deposit_target);
    lc_image_write_u32(writer, node->polarity);
    lc_image_write_u32(writer, node->reset_before_deposit);
    lc_image_write_u32(writer, node->arithmetic_kind);
    lc_image_write_real(writer, node->scalar_affine_decay);
    lc_image_write_real(writer, node->scalar_affine_drive);
    lc_image_write_real(writer, node->scalar_affine_reset);
    lc_image_write_hazard(writer, &node->hazard);
}

static void lc_image_read_node(
    lc_image_reader *reader,
    lc_mixed_node *node,
    uint64_t *program_reference,
    uint64_t *deposit_reference
) {
    uint32_t index;
    *program_reference = lc_image_read_u64(reader);
    *deposit_reference = lc_image_read_u64(reader);
    node->dispatch = lc_image_read_u32(reader);
    node->crossing_kind = lc_image_read_u32(reader);
    node->state_offset = lc_image_read_u32(reader);
    node->state_count = lc_image_read_u32(reader);
    node->readout = lc_image_read_u32(reader);
    node->parameter_offset = lc_image_read_u32(reader);
    node->parameter_count = lc_image_read_u32(reader);
    node->threshold = lc_image_read_real(reader);
    node->refractory = lc_image_read_time(reader);
    node->program_node_count = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        node->normal_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        node->clamped_roots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_ANALYTICAL_MAX_STATES; ++index)
        node->reset_roots[index] = lc_image_read_u32(reader);
    lc_image_read_scalar_log_hint(reader, &node->scalar_log_hint);
    lc_image_read_root_hint(reader, &node->root_hint);
    lc_image_read_two_exp_hint(reader, &node->two_exp_hint);
    lc_image_read_multi_exp_hint(reader, &node->multi_exp_hint);
    lc_image_read_exp_poly_hint(reader, &node->exp_poly_hint);
    lc_image_read_step_config(reader, &node->step_config);
    node->deposit_node_count = lc_image_read_u32(reader);
    node->deposit_root = lc_image_read_u32(reader);
    node->deposit_target = lc_image_read_u32(reader);
    node->polarity = lc_image_read_u32(reader);
    node->reset_before_deposit = lc_image_read_u32(reader);
    node->arithmetic_kind = lc_image_read_u32(reader);
    node->scalar_affine_decay = lc_image_read_real(reader);
    node->scalar_affine_drive = lc_image_read_real(reader);
    node->scalar_affine_reset = lc_image_read_real(reader);
    lc_image_read_hazard(reader, &node->hazard);
}

static void lc_image_write_edge(
    lc_image_writer *writer,
    const lc_mixed_edge *edge
) {
    lc_image_write_u32(writer, edge->pre);
    lc_image_write_u32(writer, edge->post);
    lc_image_write_u32(writer, edge->deposit_kind);
    lc_image_write_u32(writer, edge->target);
    lc_image_write_real(writer, edge->weight);
    lc_image_write_time(writer, edge->delay);
    lc_image_write_real(writer, edge->deposit_scale);
}

static void lc_image_read_edge(
    lc_image_reader *reader,
    lc_mixed_edge *edge
) {
    edge->pre = lc_image_read_u32(reader);
    edge->post = lc_image_read_u32(reader);
    edge->deposit_kind = lc_image_read_u32(reader);
    edge->target = lc_image_read_u32(reader);
    edge->weight = lc_image_read_real(reader);
    edge->delay = lc_image_read_time(reader);
    edge->deposit_scale = lc_image_read_real(reader);
}

static void lc_image_write_delivery_group(
    lc_image_writer *writer,
    const lc_delivery_group *group
) {
    lc_image_write_u64(writer, group->edge_offset);
    lc_image_write_u32(writer, group->edge_count);
    lc_image_write_u32(writer, group->first_edge);
    lc_image_write_time(writer, group->delay);
}

static void lc_image_read_delivery_group(
    lc_image_reader *reader,
    lc_delivery_group *group
) {
    group->edge_offset = lc_image_read_u64(reader);
    group->edge_count = lc_image_read_u32(reader);
    group->first_edge = lc_image_read_u32(reader);
    group->delay = lc_image_read_time(reader);
}

static void lc_image_write_plasticity_rule(
    lc_image_writer *writer,
    const lc_plasticity_rule *rule
) {
    lc_image_write_u32(writer, rule->kind);
    lc_image_write_u32(writer, rule->modulator);
    lc_image_write_u32(writer, rule->consume_on_modulation);
    lc_image_write_u32(writer, rule->weight_group);
    lc_image_write_real(writer, rule->tau_pre);
    lc_image_write_real(writer, rule->tau_post);
    lc_image_write_real(writer, rule->tau_pre_slow);
    lc_image_write_real(writer, rule->tau_post_slow);
    lc_image_write_real(writer, rule->a2_plus);
    lc_image_write_real(writer, rule->a2_minus);
    lc_image_write_real(writer, rule->a3_plus);
    lc_image_write_real(writer, rule->a3_minus);
    lc_image_write_real(writer, rule->tau_eligibility_plus);
    lc_image_write_real(writer, rule->tau_eligibility_minus);
    lc_image_write_real(writer, rule->positive_plus);
    lc_image_write_real(writer, rule->positive_minus);
    lc_image_write_real(writer, rule->negative_plus);
    lc_image_write_real(writer, rule->negative_minus);
    lc_image_write_real(writer, rule->learning_rate);
    lc_image_write_real(writer, rule->weight_min);
    lc_image_write_real(writer, rule->weight_max);
}

static void lc_image_read_plasticity_rule(
    lc_image_reader *reader,
    lc_plasticity_rule *rule
) {
    rule->kind = lc_image_read_u32(reader);
    rule->modulator = lc_image_read_u32(reader);
    rule->consume_on_modulation = lc_image_read_u32(reader);
    rule->weight_group = lc_image_read_u32(reader);
    rule->tau_pre = lc_image_read_real(reader);
    rule->tau_post = lc_image_read_real(reader);
    rule->tau_pre_slow = lc_image_read_real(reader);
    rule->tau_post_slow = lc_image_read_real(reader);
    rule->a2_plus = lc_image_read_real(reader);
    rule->a2_minus = lc_image_read_real(reader);
    rule->a3_plus = lc_image_read_real(reader);
    rule->a3_minus = lc_image_read_real(reader);
    rule->tau_eligibility_plus = lc_image_read_real(reader);
    rule->tau_eligibility_minus = lc_image_read_real(reader);
    rule->positive_plus = lc_image_read_real(reader);
    rule->positive_minus = lc_image_read_real(reader);
    rule->negative_plus = lc_image_read_real(reader);
    rule->negative_minus = lc_image_read_real(reader);
    rule->learning_rate = lc_image_read_real(reader);
    rule->weight_min = lc_image_read_real(reader);
    rule->weight_max = lc_image_read_real(reader);
}

static void lc_image_write_learning_event(
    lc_image_writer *writer,
    const lc_learning_event_program *event,
    uint64_t expression_offset
) {
    uint32_t index;
    lc_image_write_u64(writer, expression_offset);
    lc_image_write_u32(writer, event->node_count);
    lc_image_write_u32(writer, event->advance_mask);
    lc_image_write_u32(writer, event->variable_mask);
    lc_image_write_u32(writer, event->weight_root);
    lc_image_write_u32(writer, event->trace_update_count);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        lc_image_write_u32(writer, event->trace_indices[index]);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        lc_image_write_u32(writer, event->trace_roots[index]);
}

static void lc_image_read_learning_event(
    lc_image_reader *reader,
    lc_learning_event_program *event,
    uint64_t *expression_offset
) {
    uint32_t index;
    *expression_offset = lc_image_read_u64(reader);
    event->node_count = lc_image_read_u32(reader);
    event->advance_mask = lc_image_read_u32(reader);
    event->variable_mask = lc_image_read_u32(reader);
    event->weight_root = lc_image_read_u32(reader);
    event->trace_update_count = lc_image_read_u32(reader);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        event->trace_indices[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        event->trace_roots[index] = lc_image_read_u32(reader);
}

static void lc_image_write_learning_observer(
    lc_image_writer *writer,
    const lc_learning_observer_program *observer,
    uint64_t expression_offset
) {
    lc_image_write_u64(writer, expression_offset);
    lc_image_write_u32(writer, observer->node_count);
    lc_image_write_u32(writer, observer->variable_mask);
    lc_image_write_u32(writer, observer->parameter_mask);
    lc_image_write_u32(writer, observer->voltage_tau_parameter);
    lc_image_write_u32(writer, observer->fast_activity_tau_parameter);
    lc_image_write_u32(writer, observer->slow_activity_tau_parameter);
    lc_image_write_u32(writer, observer->band_width_parameter);
    lc_image_write_u32(writer, observer->fast_activity_root);
    lc_image_write_u32(writer, observer->slow_activity_root);
    lc_image_write_u32(writer, observer->gain_root);
}

static void lc_image_read_learning_observer(
    lc_image_reader *reader,
    lc_learning_observer_program *observer,
    uint64_t *expression_offset
) {
    *expression_offset = lc_image_read_u64(reader);
    observer->node_count = lc_image_read_u32(reader);
    observer->variable_mask = lc_image_read_u32(reader);
    observer->parameter_mask = lc_image_read_u32(reader);
    observer->voltage_tau_parameter = lc_image_read_u32(reader);
    observer->fast_activity_tau_parameter = lc_image_read_u32(reader);
    observer->slow_activity_tau_parameter = lc_image_read_u32(reader);
    observer->band_width_parameter = lc_image_read_u32(reader);
    observer->fast_activity_root = lc_image_read_u32(reader);
    observer->slow_activity_root = lc_image_read_u32(reader);
    observer->gain_root = lc_image_read_u32(reader);
}

static void lc_image_write_learning_program(
    lc_image_writer *writer,
    const lc_learning_program *program,
    uint64_t *expression_offset
) {
    uint32_t index;
    lc_image_write_u32(writer, program->parameter_count);
    lc_image_write_u32(writer, program->variable_count);
    lc_image_write_u32(writer, program->trace_count);
    lc_image_write_u32(writer, program->clamp_normalized_weight);
    lc_image_write_u32(writer, program->compatibility_kind);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        lc_image_write_u32(writer, program->trace_tau_parameters[index]);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        lc_image_write_u32(writer, program->trace_storage_slots[index]);
    for (index = 0U; index < LC_LEARNING_EVENT_COUNT; ++index) {
        const lc_learning_event_program *event = &program->events[index];
        uint64_t reference = event->node_count == 0U
            ? LC_IMAGE_NULL_REF : *expression_offset;
        lc_image_write_learning_event(writer, event, reference);
        *expression_offset += event->node_count;
    }
    {
        const lc_learning_observer_program *observer = &program->observer;
        uint64_t reference = observer->node_count == 0U
            ? LC_IMAGE_NULL_REF : *expression_offset;
        lc_image_write_learning_observer(writer, observer, reference);
        *expression_offset += observer->node_count;
    }
}

static void lc_image_read_learning_program(
    lc_image_reader *reader,
    lc_learning_program *program,
    uint64_t *expression_offsets
) {
    uint32_t index;
    program->parameter_count = lc_image_read_u32(reader);
    program->variable_count = lc_image_read_u32(reader);
    program->trace_count = lc_image_read_u32(reader);
    program->clamp_normalized_weight = lc_image_read_u32(reader);
    program->compatibility_kind = lc_image_read_u32(reader);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        program->trace_tau_parameters[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_LEARNING_MAX_TRACES; ++index)
        program->trace_storage_slots[index] = lc_image_read_u32(reader);
    for (index = 0U; index < LC_LEARNING_EVENT_COUNT; ++index)
        lc_image_read_learning_event(
            reader, &program->events[index], &expression_offsets[index]
        );
    lc_image_read_learning_observer(
        reader, &program->observer,
        &expression_offsets[LC_LEARNING_EVENT_COUNT]
    );
}

static void lc_image_write_learning_binding(
    lc_image_writer *writer,
    const lc_learning_binding *binding
) {
    lc_image_write_u32(writer, binding->program);
    lc_image_write_u32(writer, binding->parameter_offset);
    lc_image_write_u32(writer, binding->modulator);
    lc_image_write_u32(writer, binding->weight_group);
    lc_image_write_real(writer, binding->weight_min);
    lc_image_write_real(writer, binding->weight_max);
}

static void lc_image_read_learning_binding(
    lc_image_reader *reader,
    lc_learning_binding *binding
) {
    binding->program = lc_image_read_u32(reader);
    binding->parameter_offset = lc_image_read_u32(reader);
    binding->modulator = lc_image_read_u32(reader);
    binding->weight_group = lc_image_read_u32(reader);
    binding->weight_min = lc_image_read_real(reader);
    binding->weight_max = lc_image_read_real(reader);
}

static void lc_image_write_learning_batch(
    lc_image_writer *writer,
    const lc_learning_batch *batch
) {
    lc_image_write_u64(writer, batch->slot_offset);
    lc_image_write_u32(writer, batch->slot_count);
    lc_image_write_u32(writer, batch->program);
}

static void lc_image_read_learning_batch(
    lc_image_reader *reader,
    lc_learning_batch *batch
) {
    batch->slot_offset = lc_image_read_u64(reader);
    batch->slot_count = lc_image_read_u32(reader);
    batch->program = lc_image_read_u32(reader);
}

static int lc_image_optional_fields_valid(
    const lc_compiled_graph *graph,
    uint64_t flags
) {
    if (((flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U) !=
            (graph->learning_bindings != NULL ||
             graph->learning_program_count > 0U ||
             graph->learning_observer_program_by_node != NULL ||
             graph->learning_observer_parameter_offset_by_node != NULL) ||
        ((flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U) !=
            (graph->learning_observer_program_by_node != NULL &&
             graph->learning_observer_parameter_offset_by_node != NULL) ||
        ((flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U) !=
            (graph->weight_master_slots != NULL &&
             graph->weight_learning_scales != NULL) ||
        ((flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U) !=
            (graph->incoming_plastic_offsets != NULL &&
             graph->incoming_plastic_slots != NULL) ||
        ((flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U) !=
            (graph->incoming_learning_batch_offsets != NULL &&
             (graph->incoming_learning_batch_count == 0U ||
              graph->incoming_learning_batches != NULL)) ||
        ((flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U) !=
            (graph->modulator_offsets != NULL &&
             graph->modulator_slots != NULL) ||
        ((flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U) !=
            (graph->modulation_learning_batch_offsets != NULL &&
             (graph->modulation_learning_batch_count == 0U ||
              graph->modulation_learning_batches != NULL))) {
        return 0;
    }
    if (graph->parameter_count > 0U && graph->parameters == NULL) return 0;
    if (graph->edge_count > 0U &&
        (graph->edges == NULL || graph->delivery_groups == NULL ||
         graph->delivery_group_edges == NULL)) return 0;
    if (graph->delivery_group_count > graph->edge_count) return 0;
    if (graph->learning_program_count > 0U &&
        graph->learning_programs == NULL) return 0;
    if (graph->learning_parameter_count > 0U &&
        graph->learning_parameters == NULL) return 0;
    if (graph->plastic_edge_count > 0U && graph->plastic_edges == NULL) return 0;
    return 1;
}

static int lc_image_encode_graph(
    lc_image_writer *writer,
    const lc_compiled_graph *graph,
    lc_image_layout *layout,
    uint64_t total_size
) {
    uint32_t index;
    uint64_t position;
    uint64_t learning_expression_offset = 0U;
    if (!lc_image_optional_fields_valid(graph, layout->flags)) return 0;

    lc_image_write_bytes(writer, lc_image_magic, sizeof(lc_image_magic));
    lc_image_write_u32(writer, LC_IMAGE_VERSION);
    lc_image_write_u32(writer, LC_ABI_VERSION);
    lc_image_write_u32(writer, LC_IMAGE_PROFILE);
    lc_image_write_u32(writer, 0U);
    lc_image_write_u64(writer, total_size);
    lc_image_write_u32(writer, 0U);
    lc_image_write_u32(writer, 0U);
#if LC_IMAGE_VERSION == 2U
    lc_image_write_u32(writer, LACUNA_REAL_BITS);
    lc_image_write_u32(writer, LACUNA_TIME_BITS);
    lc_image_write_u32(writer, LC_NUMERIC_ARITHMETIC_REVISION);
    lc_image_write_u32(writer, 0U);
#endif

    lc_image_write_u64(writer, layout->flags);
    lc_image_write_u32(writer, graph->node_count);
    lc_image_write_u32(writer, graph->state_count);
    lc_image_write_u32(writer, graph->parameter_count);
    lc_image_write_u32(writer, graph->edge_count);
    lc_image_write_u32(writer, graph->workspace_count);
    lc_image_write_u32(writer, graph->delivery_group_count);
    lc_image_write_u32(writer, graph->learning_program_count);
    lc_image_write_u32(writer, graph->learning_parameter_count);
    lc_image_write_u32(writer, graph->plastic_edge_count);
    lc_image_write_u32(writer, graph->incoming_learning_batch_count);
    lc_image_write_u32(writer, graph->modulator_count);
    lc_image_write_u32(writer, graph->modulation_learning_batch_count);
    lc_image_write_u64(writer, layout->base.expression_count);
    lc_image_write_u64(writer, layout->specialized.expression_count);
    lc_image_write_u64(writer, layout->learning_expression_count);

    for (index = 0U; index < graph->node_count; ++index) {
        const lc_mixed_node *node = &graph->nodes[index];
        uint64_t program_reference;
        uint64_t deposit_reference;
        if (!lc_image_pool_find(
                &layout->base, node->program_nodes,
                node->program_node_count, &program_reference
            )) return 0;
        if (node->deposit_node_count == 0U) {
            deposit_reference = LC_IMAGE_NULL_REF;
        } else if (!lc_image_pool_find(
                &layout->base, node->deposit_nodes,
                node->deposit_node_count, &deposit_reference
            )) return 0;
        lc_image_write_node(
            writer, node, program_reference, deposit_reference
        );
    }
    for (index = 0U; index < layout->base.count; ++index) {
        const lc_image_expression_block *block = &layout->base.blocks[index];
        uint32_t expression;
        for (expression = 0U; expression < block->count; ++expression)
            lc_image_write_expr(writer, &block->nodes[expression]);
    }
    for (index = 0U; index < layout->specialized.count; ++index) {
        const lc_image_expression_block *block =
            &layout->specialized.blocks[index];
        uint32_t expression;
        for (expression = 0U; expression < block->count; ++expression)
            lc_image_write_expr(writer, &block->nodes[expression]);
    }
    if ((layout->flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U) {
        for (index = 0U; index < graph->node_count; ++index) {
            const lc_mixed_node *node = &graph->nodes[index];
            const lc_node_eval_plan *plan = &graph->node_eval_plans[index];
            uint64_t reference;
            if (!lc_image_frozen_plan_reference(
                    layout, plan->normal_nodes,
                    node->program_node_count, &reference
                )) return 0;
            lc_image_write_u64(writer, reference);
            if (!lc_image_frozen_plan_reference(
                    layout, plan->clamped_nodes,
                    node->program_node_count, &reference
                )) return 0;
            lc_image_write_u64(writer, reference);
            if (!lc_image_frozen_plan_reference(
                    layout, plan->reset_nodes,
                    node->program_node_count, &reference
                )) return 0;
            lc_image_write_u64(writer, reference);
            if (!lc_image_frozen_plan_reference(
                    layout, plan->crossing_nodes,
                    node->program_node_count, &reference
                )) return 0;
            lc_image_write_u64(writer, reference);
            if (!lc_image_frozen_plan_reference(
                    layout, plan->deposit_nodes,
                    node->deposit_node_count, &reference
                )) return 0;
            lc_image_write_u64(writer, reference);
        }
    }
    for (index = 0U; index < graph->parameter_count; ++index)
        lc_image_write_real(writer, graph->parameters[index]);
    for (index = 0U; index < graph->edge_count; ++index)
        lc_image_write_edge(writer, &graph->edges[index]);
    for (index = 0U; index < graph->delivery_group_count; ++index)
        lc_image_write_delivery_group(writer, &graph->delivery_groups[index]);
    for (index = 0U; index < graph->edge_count; ++index)
        lc_image_write_u32(writer, graph->delivery_group_edges[index]);
    for (position = 0U; position <= graph->node_count; ++position)
        lc_image_write_u64(writer, graph->outgoing_offsets[position]);

    if ((layout->flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U) {
        for (index = 0U; index < graph->edge_count; ++index)
            lc_image_write_plasticity_rule(writer, &graph->plasticity[index]);
    }
    for (index = 0U; index < graph->learning_program_count; ++index)
        lc_image_write_learning_program(
            writer, &graph->learning_programs[index],
            &learning_expression_offset
        );
    if (learning_expression_offset != layout->learning_expression_count)
        return 0;
    for (index = 0U; index < graph->learning_program_count; ++index) {
        uint32_t event;
        const lc_learning_program *program = &graph->learning_programs[index];
        for (event = 0U; event < LC_LEARNING_EVENT_COUNT; ++event) {
            uint32_t expression;
            for (expression = 0U;
                 expression < program->events[event].node_count;
                 ++expression) {
                lc_image_write_expr(
                    writer, &program->events[event].nodes[expression]
                );
            }
        }
        {
            uint32_t expression;
            for (expression = 0U;
                 expression < program->observer.node_count;
                 ++expression) {
                lc_image_write_expr(
                    writer, &program->observer.nodes[expression]
                );
            }
        }
    }
    for (index = 0U; index < graph->learning_parameter_count; ++index)
        lc_image_write_real(writer, graph->learning_parameters[index]);
    if ((layout->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U) {
        for (index = 0U; index < graph->edge_count; ++index)
            lc_image_write_learning_binding(
                writer, &graph->learning_bindings[index]
            );
    }
    if ((layout->flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U) {
        for (index = 0U; index < graph->node_count; ++index)
            lc_image_write_u32(
                writer, graph->learning_observer_program_by_node[index]
            );
        for (index = 0U; index < graph->node_count; ++index)
            lc_image_write_u32(
                writer,
                graph->learning_observer_parameter_offset_by_node[index]
            );
    }
    if ((layout->flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) != 0U) {
        for (index = 0U; index < graph->edge_count; ++index)
            lc_image_write_u32(writer, graph->plastic_slot_by_edge[index]);
    }
    for (index = 0U; index < graph->plastic_edge_count; ++index)
        lc_image_write_u32(writer, graph->plastic_edges[index]);
    if ((layout->flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U) {
        for (index = 0U; index < graph->plastic_edge_count; ++index)
            lc_image_write_u32(writer, graph->weight_master_slots[index]);
        for (index = 0U; index < graph->plastic_edge_count; ++index)
            lc_image_write_real(writer, graph->weight_learning_scales[index]);
    }
    if ((layout->flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U) {
        for (position = 0U; position <= graph->node_count; ++position)
            lc_image_write_u64(
                writer, graph->incoming_plastic_offsets[position]
            );
        for (index = 0U; index < graph->plastic_edge_count; ++index)
            lc_image_write_u32(writer, graph->incoming_plastic_slots[index]);
    }
    if ((layout->flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U) {
        for (position = 0U; position <= graph->node_count; ++position)
            lc_image_write_u64(
                writer, graph->incoming_learning_batch_offsets[position]
            );
        for (index = 0U;
             index < graph->incoming_learning_batch_count; ++index)
            lc_image_write_learning_batch(
                writer, &graph->incoming_learning_batches[index]
            );
    }
    if ((layout->flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U) {
        for (position = 0U; position <= graph->modulator_count; ++position)
            lc_image_write_u64(writer, graph->modulator_offsets[position]);
        for (index = 0U; index < graph->plastic_edge_count; ++index)
            lc_image_write_u32(writer, graph->modulator_slots[index]);
    }
    if ((layout->flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U) {
        for (position = 0U; position <= graph->modulator_count; ++position)
            lc_image_write_u64(
                writer, graph->modulation_learning_batch_offsets[position]
            );
        for (index = 0U;
             index < graph->modulation_learning_batch_count; ++index)
            lc_image_write_learning_batch(
                writer, &graph->modulation_learning_batches[index]
            );
    }
    return !writer->failed;
}

lc_status lc_compiled_graph_image_size(
    const lc_compiled_graph *compiled,
    uint64_t *size
) {
    lc_image_layout layout;
    lc_image_writer writer;
    if (size == NULL) return LC_INVALID_ARGUMENT;
    *size = 0U;
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) return LC_NUMERIC_ERROR;
#endif
    if (!lc_image_numeric_supported()) return LC_IMAGE_INCOMPATIBLE;
    if (!lc_compiled_graph_validate_loaded(compiled)) return LC_IMAGE_INVALID;
    if (!lc_image_layout_build(compiled, &layout)) return LC_IMAGE_INVALID;
    memset(&writer, 0, sizeof(writer));
    writer.capacity = UINT64_MAX;
    if (!lc_image_encode_graph(&writer, compiled, &layout, 0U)) {
        lc_image_layout_release(&layout);
        return LC_IMAGE_INVALID;
    }
    *size = writer.position;
    lc_image_layout_release(&layout);
    return LC_OK;
}

lc_status lc_compiled_graph_serialize(
    const lc_compiled_graph *compiled,
    uint8_t *destination,
    uint64_t capacity,
    uint64_t *written
) {
    lc_image_layout layout;
    lc_image_writer measure;
    lc_image_writer writer;
    uint64_t required;
    uint32_t checksum;
    if (written == NULL || (capacity > 0U && destination == NULL))
        return LC_INVALID_ARGUMENT;
    *written = 0U;
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) return LC_NUMERIC_ERROR;
#endif
    if (!lc_image_numeric_supported()) return LC_IMAGE_INCOMPATIBLE;
    if (!lc_compiled_graph_validate_loaded(compiled)) return LC_IMAGE_INVALID;
    if (!lc_image_layout_build(compiled, &layout)) return LC_IMAGE_INVALID;
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    if (!lc_image_encode_graph(&measure, compiled, &layout, 0U)) {
        lc_image_layout_release(&layout);
        return LC_IMAGE_INVALID;
    }
    required = measure.position;
    *written = required;
    if (required > capacity) {
        lc_image_layout_release(&layout);
        return LC_OUTPUT_OVERFLOW;
    }
    memset(&writer, 0, sizeof(writer));
    writer.data = destination;
    writer.capacity = capacity;
    if (!lc_image_encode_graph(&writer, compiled, &layout, required) ||
        writer.position != required || required < LC_IMAGE_HEADER_SIZE) {
        lc_image_layout_release(&layout);
        return LC_IMAGE_INVALID;
    }
    checksum = lc_image_crc32(
        &destination[LC_IMAGE_HEADER_SIZE], required - LC_IMAGE_HEADER_SIZE
    );
    lc_image_patch_u32(destination, 32U, checksum);
    lc_image_layout_release(&layout);
    return LC_OK;
}

typedef struct lc_image_counts {
    uint64_t flags;
    uint32_t node_count;
    uint32_t state_count;
    uint32_t parameter_count;
    uint32_t edge_count;
    uint32_t workspace_count;
    uint32_t delivery_group_count;
    uint32_t learning_program_count;
    uint32_t learning_parameter_count;
    uint32_t plastic_edge_count;
    uint32_t incoming_learning_batch_count;
    uint32_t modulator_count;
    uint32_t modulation_learning_batch_count;
    uint64_t expression_count;
    uint64_t specialized_expression_count;
    uint64_t learning_expression_count;
} lc_image_counts;

static lc_status lc_image_read_header(
    const uint8_t *image,
    uint64_t size,
    lc_image_reader *reader
) {
    uint8_t magic[8];
    uint32_t version;
    uint32_t abi;
    uint32_t numeric;
    uint32_t header_flags;
    uint64_t declared_size;
    uint32_t expected_checksum;
    uint32_t reserved;
    uint32_t actual_checksum;
    if (image == NULL || reader == NULL || size < UINT64_C(40))
        return LC_IMAGE_INVALID;
    if (!lc_image_numeric_supported())
        return LC_IMAGE_INCOMPATIBLE;
    if (size > SIZE_MAX) return LC_IMAGE_INVALID;
    memset(reader, 0, sizeof(*reader));
    reader->data = image;
    reader->size = size;
    lc_image_read_bytes(reader, magic, sizeof(magic));
    version = lc_image_read_u32(reader);
    abi = lc_image_read_u32(reader);
    numeric = lc_image_read_u32(reader);
    header_flags = lc_image_read_u32(reader);
    declared_size = lc_image_read_u64(reader);
    expected_checksum = lc_image_read_u32(reader);
    reserved = lc_image_read_u32(reader);
    if (reader->failed || memcmp(magic, lc_image_magic, sizeof(magic)) != 0)
        return LC_IMAGE_INVALID;
    if (version != LC_IMAGE_VERSION ||
        abi != LC_ABI_VERSION || numeric != LC_IMAGE_PROFILE)
        return LC_IMAGE_INCOMPATIBLE;
    if (header_flags != 0U || reserved != 0U || declared_size != size)
        return LC_IMAGE_INVALID;
#if LC_IMAGE_VERSION == 2U
    {
        uint32_t real_bits;
        uint32_t time_bits;
        uint32_t arithmetic_revision;
        uint32_t profile_reserved;
        if (size < LC_IMAGE_HEADER_SIZE) return LC_IMAGE_INVALID;
        real_bits = lc_image_read_u32(reader);
        time_bits = lc_image_read_u32(reader);
        arithmetic_revision = lc_image_read_u32(reader);
        profile_reserved = lc_image_read_u32(reader);
        if (reader->failed || profile_reserved != 0U) return LC_IMAGE_INVALID;
        if (real_bits != LACUNA_REAL_BITS || time_bits != LACUNA_TIME_BITS ||
            arithmetic_revision != LC_NUMERIC_ARITHMETIC_REVISION)
            return LC_IMAGE_INCOMPATIBLE;
    }
#endif
    actual_checksum = lc_image_crc32(
        &image[LC_IMAGE_HEADER_SIZE], size - LC_IMAGE_HEADER_SIZE
    );
    if (actual_checksum != expected_checksum)
        return LC_IMAGE_CHECKSUM_MISMATCH;
    return LC_OK;
}

static int lc_image_read_counts(
    lc_image_reader *reader,
    lc_image_counts *counts
) {
    counts->flags = lc_image_read_u64(reader);
    counts->node_count = lc_image_read_u32(reader);
    counts->state_count = lc_image_read_u32(reader);
    counts->parameter_count = lc_image_read_u32(reader);
    counts->edge_count = lc_image_read_u32(reader);
    counts->workspace_count = lc_image_read_u32(reader);
    counts->delivery_group_count = lc_image_read_u32(reader);
    counts->learning_program_count = lc_image_read_u32(reader);
    counts->learning_parameter_count = lc_image_read_u32(reader);
    counts->plastic_edge_count = lc_image_read_u32(reader);
    counts->incoming_learning_batch_count = lc_image_read_u32(reader);
    counts->modulator_count = lc_image_read_u32(reader);
    counts->modulation_learning_batch_count = lc_image_read_u32(reader);
    counts->expression_count = lc_image_read_u64(reader);
    counts->specialized_expression_count = lc_image_read_u64(reader);
    counts->learning_expression_count = lc_image_read_u64(reader);
    if (reader->failed || (counts->flags & ~LC_IMAGE_KNOWN_FLAGS) != 0U ||
        counts->node_count == 0U || counts->state_count == 0U ||
        counts->workspace_count == 0U ||
        counts->delivery_group_count > counts->edge_count ||
        counts->plastic_edge_count > counts->edge_count ||
        counts->expression_count == 0U ||
        counts->expression_count >= LC_IMAGE_SPECIALIZED_REF ||
        counts->specialized_expression_count >= LC_IMAGE_SPECIALIZED_REF) {
        return 0;
    }
    if ((counts->flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U &&
        (counts->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U) return 0;
    if (counts->learning_program_count > 0U &&
        (counts->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) == 0U) return 0;
    if (counts->specialized_expression_count > 0U &&
        (counts->flags & LC_IMAGE_HAS_EVAL_PLANS) == 0U) return 0;
    if (counts->plastic_edge_count > 0U &&
        ((counts->flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) == 0U ||
         (counts->flags & LC_IMAGE_HAS_INCOMING_PLASTIC) == 0U)) return 0;
    if (counts->modulator_count > 0U &&
        (counts->flags & LC_IMAGE_HAS_MODULATOR_INDEX) == 0U) return 0;
    return 1;
}

static int lc_image_wire_add(
    uint64_t *total,
    uint64_t count,
    uint64_t element_size
) {
    uint64_t amount;
    if (element_size != 0U && count > UINT64_MAX / element_size) return 0;
    amount = count * element_size;
    if (amount > UINT64_MAX - *total) return 0;
    *total += amount;
    return 1;
}

/* Compute the exact image length before allocating from untrusted counts. */
static int lc_image_wire_size(
    const lc_image_counts *counts,
    uint64_t *result
) {
    lc_image_writer measure;
    lc_mixed_node node;
    lc_mixed_edge edge;
    lc_delivery_group group;
    lc_plasticity_rule rule;
    lc_learning_program program;
    lc_learning_binding binding;
    lc_learning_batch batch;
    uint64_t expression_offset = 0U;
    uint64_t node_size;
    uint64_t edge_size;
    uint64_t group_size;
    uint64_t rule_size;
    uint64_t program_size;
    uint64_t binding_size;
    uint64_t batch_size;
    uint64_t total = LC_IMAGE_HEADER_SIZE + UINT64_C(80);

    memset(&node, 0, sizeof(node));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_node(&measure, &node, 0U, 0U);
    node_size = measure.position;

    memset(&edge, 0, sizeof(edge));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_edge(&measure, &edge);
    edge_size = measure.position;

    memset(&group, 0, sizeof(group));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_delivery_group(&measure, &group);
    group_size = measure.position;

    memset(&rule, 0, sizeof(rule));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_plasticity_rule(&measure, &rule);
    rule_size = measure.position;

    memset(&program, 0, sizeof(program));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_learning_program(
        &measure, &program, &expression_offset
    );
    program_size = measure.position;

    memset(&binding, 0, sizeof(binding));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_learning_binding(&measure, &binding);
    binding_size = measure.position;

    memset(&batch, 0, sizeof(batch));
    memset(&measure, 0, sizeof(measure));
    measure.capacity = UINT64_MAX;
    lc_image_write_learning_batch(&measure, &batch);
    batch_size = measure.position;

    if (!lc_image_wire_add(&total, counts->node_count, node_size) ||
        !lc_image_wire_add(&total, counts->expression_count, LC_IMAGE_EXPR_SIZE) ||
        !lc_image_wire_add(
            &total, counts->specialized_expression_count, LC_IMAGE_EXPR_SIZE
        ) ||
        (((counts->flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U) &&
         !lc_image_wire_add(&total, counts->node_count, 40U)) ||
        !lc_image_wire_add(&total, counts->parameter_count, LC_IMAGE_REAL_SIZE) ||
        !lc_image_wire_add(&total, counts->edge_count, edge_size) ||
        !lc_image_wire_add(
            &total, counts->delivery_group_count, group_size
        ) ||
        !lc_image_wire_add(&total, counts->edge_count, 4U) ||
        !lc_image_wire_add(&total, (uint64_t)counts->node_count + 1U, 8U) ||
        (((counts->flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U) &&
         !lc_image_wire_add(&total, counts->edge_count, rule_size)) ||
        !lc_image_wire_add(
            &total, counts->learning_program_count, program_size
        ) ||
        !lc_image_wire_add(
            &total, counts->learning_expression_count, LC_IMAGE_EXPR_SIZE
        ) ||
        !lc_image_wire_add(
            &total, counts->learning_parameter_count, LC_IMAGE_REAL_SIZE
        ) ||
        (((counts->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U) &&
         !lc_image_wire_add(&total, counts->edge_count, binding_size)) ||
        (((counts->flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U) &&
         !lc_image_wire_add(&total, counts->node_count, 8U)) ||
        (((counts->flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) != 0U) &&
         !lc_image_wire_add(&total, counts->edge_count, 4U)) ||
        !lc_image_wire_add(&total, counts->plastic_edge_count, 4U) ||
        (((counts->flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U) &&
         !lc_image_wire_add(
             &total, counts->plastic_edge_count, UINT64_C(4) + LC_IMAGE_REAL_SIZE
         )) ||
        (((counts->flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U) &&
         (!lc_image_wire_add(
              &total, (uint64_t)counts->node_count + 1U, 8U
          ) || !lc_image_wire_add(
              &total, counts->plastic_edge_count, 4U
          ))) ||
        (((counts->flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U) &&
         (!lc_image_wire_add(
              &total, (uint64_t)counts->node_count + 1U, 8U
          ) || !lc_image_wire_add(
              &total, counts->incoming_learning_batch_count, batch_size
          ))) ||
        (((counts->flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U) &&
         (!lc_image_wire_add(
              &total, (uint64_t)counts->modulator_count + 1U, 8U
          ) || !lc_image_wire_add(
              &total, counts->plastic_edge_count, 4U
          ))) ||
        (((counts->flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U) &&
         (!lc_image_wire_add(
              &total, (uint64_t)counts->modulator_count + 1U, 8U
          ) || !lc_image_wire_add(
              &total, counts->modulation_learning_batch_count, batch_size
          )))) {
        return 0;
    }
    *result = total;
    return 1;
}

static void *lc_image_calloc(uint64_t count, size_t element_size) {
    if (count == 0U || lc_image_allocation_overflows(count, element_size))
        return NULL;
    return calloc((size_t)count, element_size);
}

static int lc_image_resolve_expression_reference(
    uint64_t reference,
    uint32_t count,
    const lc_expr_node *base,
    uint64_t base_count,
    const lc_expr_node *specialized,
    uint64_t specialized_count,
    int allow_specialized,
    const lc_expr_node **result
) {
    uint64_t offset;
    uint64_t available;
    const lc_expr_node *storage;
    if (result == NULL) return 0;
    *result = NULL;
    if (count == 0U) return reference == LC_IMAGE_NULL_REF;
    if (reference == LC_IMAGE_NULL_REF) return 0;
    if ((reference & LC_IMAGE_SPECIALIZED_REF) != 0U) {
        if (!allow_specialized) return 0;
        offset = reference & ~LC_IMAGE_SPECIALIZED_REF;
        available = specialized_count;
        storage = specialized;
    } else {
        offset = reference;
        available = base_count;
        storage = base;
    }
    if (storage == NULL || offset > available || count > available - offset)
        return 0;
    *result = &storage[offset];
    return 1;
}

static int lc_image_allocate_graph(
    const lc_image_counts *counts,
    lc_compiled_graph **graph_out
) {
    lc_compiled_graph *graph = calloc(1U, sizeof(*graph));
    if (graph == NULL) return 0;
    graph->references = 1U;
    graph->node_count = counts->node_count;
    graph->state_count = counts->state_count;
    graph->parameter_count = counts->parameter_count;
    graph->edge_count = counts->edge_count;
    graph->workspace_count = counts->workspace_count;
    graph->delivery_group_count = counts->delivery_group_count;
    graph->learning_program_count = counts->learning_program_count;
    graph->learning_parameter_count = counts->learning_parameter_count;
    graph->plastic_edge_count = counts->plastic_edge_count;
    graph->incoming_learning_batch_count =
        counts->incoming_learning_batch_count;
    graph->modulator_count = counts->modulator_count;
    graph->modulation_learning_batch_count =
        counts->modulation_learning_batch_count;

    graph->nodes = lc_image_calloc(
        counts->node_count, sizeof(lc_mixed_node)
    );
    graph->expression_nodes = lc_image_calloc(
        counts->expression_count, sizeof(lc_expr_node)
    );
    graph->outgoing_offsets = lc_image_calloc(
        (uint64_t)counts->node_count + 1U, sizeof(uint64_t)
    );
    if (counts->specialized_expression_count > 0U)
        graph->specialized_expression_nodes = lc_image_calloc(
            counts->specialized_expression_count, sizeof(lc_expr_node)
        );
    if ((counts->flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U)
        graph->node_eval_plans = lc_image_calloc(
            counts->node_count, sizeof(lc_node_eval_plan)
        );
    if (counts->parameter_count > 0U)
        graph->parameters = lc_image_calloc(
            counts->parameter_count, sizeof(lc_real_t)
        );
    if (counts->edge_count > 0U) {
        graph->edges = lc_image_calloc(
            counts->edge_count, sizeof(lc_mixed_edge)
        );
        graph->delivery_group_edges = lc_image_calloc(
            counts->edge_count, sizeof(uint32_t)
        );
    }
    if (counts->delivery_group_count > 0U)
        graph->delivery_groups = lc_image_calloc(
            counts->delivery_group_count, sizeof(lc_delivery_group)
        );
    if ((counts->flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U)
        graph->plasticity = lc_image_calloc(
            counts->edge_count, sizeof(lc_plasticity_rule)
        );
    if (counts->learning_program_count > 0U)
        graph->learning_programs = lc_image_calloc(
            counts->learning_program_count, sizeof(lc_learning_program)
        );
    if (counts->learning_expression_count > 0U)
        graph->learning_expression_nodes = lc_image_calloc(
            counts->learning_expression_count, sizeof(lc_expr_node)
        );
    if (counts->learning_parameter_count > 0U)
        graph->learning_parameters = lc_image_calloc(
            counts->learning_parameter_count, sizeof(lc_real_t)
        );
    if ((counts->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U)
        graph->learning_bindings = lc_image_calloc(
            counts->edge_count, sizeof(lc_learning_binding)
        );
    if ((counts->flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U) {
        graph->learning_observer_program_by_node = lc_image_calloc(
            counts->node_count, sizeof(uint32_t)
        );
        graph->learning_observer_parameter_offset_by_node = lc_image_calloc(
            counts->node_count, sizeof(uint32_t)
        );
    }
    if ((counts->flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) != 0U)
        graph->plastic_slot_by_edge = lc_image_calloc(
            counts->edge_count, sizeof(uint32_t)
        );
    if (counts->plastic_edge_count > 0U)
        graph->plastic_edges = lc_image_calloc(
            counts->plastic_edge_count, sizeof(uint32_t)
        );
    if ((counts->flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U) {
        graph->weight_master_slots = lc_image_calloc(
            counts->plastic_edge_count, sizeof(uint32_t)
        );
        graph->weight_learning_scales = lc_image_calloc(
            counts->plastic_edge_count, sizeof(lc_real_t)
        );
    }
    if ((counts->flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U) {
        graph->incoming_plastic_offsets = lc_image_calloc(
            (uint64_t)counts->node_count + 1U, sizeof(uint64_t)
        );
        graph->incoming_plastic_slots = lc_image_calloc(
            counts->plastic_edge_count, sizeof(uint32_t)
        );
    }
    if ((counts->flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U) {
        graph->incoming_learning_batch_offsets = lc_image_calloc(
            (uint64_t)counts->node_count + 1U, sizeof(uint64_t)
        );
        if (counts->incoming_learning_batch_count > 0U)
            graph->incoming_learning_batches = lc_image_calloc(
                counts->incoming_learning_batch_count,
                sizeof(lc_learning_batch)
            );
    }
    if ((counts->flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U) {
        graph->modulator_offsets = lc_image_calloc(
            (uint64_t)counts->modulator_count + 1U, sizeof(uint64_t)
        );
        graph->modulator_slots = lc_image_calloc(
            counts->plastic_edge_count, sizeof(uint32_t)
        );
    }
    if ((counts->flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U) {
        graph->modulation_learning_batch_offsets = lc_image_calloc(
            (uint64_t)counts->modulator_count + 1U, sizeof(uint64_t)
        );
        if (counts->modulation_learning_batch_count > 0U)
            graph->modulation_learning_batches = lc_image_calloc(
                counts->modulation_learning_batch_count,
                sizeof(lc_learning_batch)
            );
    }
    if (graph->nodes == NULL || graph->expression_nodes == NULL ||
        graph->outgoing_offsets == NULL ||
        (counts->specialized_expression_count > 0U &&
         graph->specialized_expression_nodes == NULL) ||
        ((counts->flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U &&
         graph->node_eval_plans == NULL) ||
        (counts->parameter_count > 0U && graph->parameters == NULL) ||
        (counts->edge_count > 0U &&
         (graph->edges == NULL || graph->delivery_group_edges == NULL)) ||
        (counts->delivery_group_count > 0U && graph->delivery_groups == NULL) ||
        ((counts->flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U &&
         counts->edge_count > 0U && graph->plasticity == NULL) ||
        (counts->learning_program_count > 0U &&
         graph->learning_programs == NULL) ||
        (counts->learning_expression_count > 0U &&
         graph->learning_expression_nodes == NULL) ||
        (counts->learning_parameter_count > 0U &&
         graph->learning_parameters == NULL) ||
        ((counts->flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U &&
         counts->edge_count > 0U && graph->learning_bindings == NULL) ||
        ((counts->flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U &&
         (graph->learning_observer_program_by_node == NULL ||
          graph->learning_observer_parameter_offset_by_node == NULL)) ||
        ((counts->flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) != 0U &&
         counts->edge_count > 0U && graph->plastic_slot_by_edge == NULL) ||
        (counts->plastic_edge_count > 0U && graph->plastic_edges == NULL) ||
        ((counts->flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U &&
         (graph->weight_master_slots == NULL ||
          graph->weight_learning_scales == NULL)) ||
        ((counts->flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U &&
         (graph->incoming_plastic_offsets == NULL ||
          graph->incoming_plastic_slots == NULL)) ||
        ((counts->flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U &&
         (graph->incoming_learning_batch_offsets == NULL ||
          (counts->incoming_learning_batch_count > 0U &&
           graph->incoming_learning_batches == NULL))) ||
        ((counts->flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U &&
         (graph->modulator_offsets == NULL || graph->modulator_slots == NULL)) ||
        ((counts->flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U &&
         (graph->modulation_learning_batch_offsets == NULL ||
          (counts->modulation_learning_batch_count > 0U &&
           graph->modulation_learning_batches == NULL)))) {
        lc_mixed_graph_destroy(graph);
        return 0;
    }
    *graph_out = graph;
    return 1;
}

lc_status lc_compiled_graph_deserialize(
    const uint8_t *image,
    uint64_t size,
    lc_compiled_graph **compiled
) {
    lc_image_reader reader;
    lc_image_counts counts;
    lc_compiled_graph *graph = NULL;
    uint64_t *node_references = NULL;
    uint64_t *eval_references = NULL;
    uint64_t *learning_references = NULL;
    uint64_t expected_size = 0U;
    lc_status status;
    uint32_t index;
    uint64_t position;
    if (compiled == NULL) return LC_INVALID_ARGUMENT;
    *compiled = NULL;
#if LACUNA_REAL_BITS == 16
    if (!lc_half_environment_valid()) return LC_NUMERIC_ERROR;
#endif
    status = lc_image_read_header(image, size, &reader);
    if (status != LC_OK) return status;
    memset(&counts, 0, sizeof(counts));
    if (!lc_image_read_counts(&reader, &counts)) return LC_IMAGE_INVALID;
    if (!lc_image_wire_size(&counts, &expected_size) ||
        expected_size != size) {
        return LC_IMAGE_INVALID;
    }
    if (counts.node_count > size || counts.state_count > size ||
        counts.workspace_count > size ||
        counts.parameter_count > size || counts.edge_count > size ||
        counts.delivery_group_count > size ||
        counts.learning_program_count > size ||
        counts.learning_parameter_count > size ||
        counts.plastic_edge_count > size || counts.modulator_count > size ||
        counts.expression_count > size / LC_IMAGE_EXPR_SIZE ||
        counts.specialized_expression_count > size / LC_IMAGE_EXPR_SIZE ||
        counts.learning_expression_count > size / LC_IMAGE_EXPR_SIZE) {
        return LC_IMAGE_INVALID;
    }
    if (lc_image_allocation_overflows(
            (uint64_t)counts.node_count * 2U, sizeof(uint64_t)
        ) || lc_image_allocation_overflows(
            (uint64_t)counts.node_count * 5U, sizeof(uint64_t)
        ) || lc_image_allocation_overflows(
            (uint64_t)counts.learning_program_count *
                (LC_LEARNING_EVENT_COUNT + 1U),
            sizeof(uint64_t)
        )) {
        return LC_IMAGE_INVALID;
    }
    node_references = calloc(
        (size_t)counts.node_count * 2U, sizeof(uint64_t)
    );
    if ((counts.flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U)
        eval_references = calloc(
            (size_t)counts.node_count * 5U, sizeof(uint64_t)
        );
    if (counts.learning_program_count > 0U)
        learning_references = calloc(
            (size_t)counts.learning_program_count *
                (LC_LEARNING_EVENT_COUNT + 1U),
            sizeof(uint64_t)
        );
    if (node_references == NULL ||
        ((counts.flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U &&
         eval_references == NULL) ||
        (counts.learning_program_count > 0U && learning_references == NULL)) {
        status = LC_ALLOCATION_FAILED;
        goto cleanup;
    }
    if (!lc_image_allocate_graph(&counts, &graph)) {
        status = LC_ALLOCATION_FAILED;
        goto cleanup;
    }

    for (index = 0U; index < counts.node_count; ++index)
        lc_image_read_node(
            &reader, &graph->nodes[index],
            &node_references[(size_t)index * 2U],
            &node_references[(size_t)index * 2U + 1U]
        );
    position = 0U;
    for (index = 0U; index < counts.node_count; ++index) {
        uint32_t reference_kind;
        for (reference_kind = 0U; reference_kind < 2U; ++reference_kind) {
            uint64_t reference =
                node_references[(size_t)index * 2U + reference_kind];
            uint32_t count = reference_kind == 0U
                ? graph->nodes[index].program_node_count
                : graph->nodes[index].deposit_node_count;
            if (count == 0U) {
                if (reference != LC_IMAGE_NULL_REF) {
                    status = LC_IMAGE_INVALID;
                    goto cleanup;
                }
                continue;
            }
            if (reference == LC_IMAGE_NULL_REF ||
                (reference & LC_IMAGE_SPECIALIZED_REF) != 0U ||
                reference > position || count > position - reference) {
                if (reference != position ||
                    count > counts.expression_count - position) {
                    status = LC_IMAGE_INVALID;
                    goto cleanup;
                }
                position += count;
            }
        }
    }
    if (position != counts.expression_count) {
        status = LC_IMAGE_INVALID;
        goto cleanup;
    }
    for (position = 0U; position < counts.expression_count; ++position)
        lc_image_read_expr(&reader, &graph->expression_nodes[position]);
    for (position = 0U;
         position < counts.specialized_expression_count; ++position)
        lc_image_read_expr(
            &reader, &graph->specialized_expression_nodes[position]
        );
    if (reader.failed) {
        status = LC_IMAGE_INVALID;
        goto cleanup;
    }
    for (index = 0U; index < counts.node_count; ++index) {
        lc_mixed_node *node = &graph->nodes[index];
        if (!lc_image_resolve_expression_reference(
                node_references[(size_t)index * 2U],
                node->program_node_count,
                graph->expression_nodes, counts.expression_count,
                NULL, 0U, 0, &node->program_nodes
            ) || !lc_image_resolve_expression_reference(
                node_references[(size_t)index * 2U + 1U],
                node->deposit_node_count,
                graph->expression_nodes, counts.expression_count,
                NULL, 0U, 0, &node->deposit_nodes
            )) {
            status = LC_IMAGE_INVALID;
            goto cleanup;
        }
    }
    if ((counts.flags & LC_IMAGE_HAS_EVAL_PLANS) != 0U) {
        uint64_t specialized_cursor = 0U;
        for (index = 0U; index < counts.node_count; ++index) {
            uint32_t operation;
            for (operation = 0U; operation < 5U; ++operation)
                eval_references[(size_t)index * 5U + operation] =
                    lc_image_read_u64(&reader);
        }
        for (index = 0U; index < counts.node_count; ++index) {
            uint32_t operation;
            for (operation = 0U; operation < 5U; ++operation) {
                uint64_t reference =
                    eval_references[(size_t)index * 5U + operation];
                uint32_t count = operation == 4U
                    ? graph->nodes[index].deposit_node_count
                    : graph->nodes[index].program_node_count;
                uint64_t offset;
                if (count == 0U) {
                    if (reference != LC_IMAGE_NULL_REF) {
                        status = LC_IMAGE_INVALID;
                        goto cleanup;
                    }
                    continue;
                }
                if (reference == LC_IMAGE_NULL_REF) {
                    status = LC_IMAGE_INVALID;
                    goto cleanup;
                }
                if ((reference & LC_IMAGE_SPECIALIZED_REF) == 0U) {
                    if (reference > counts.expression_count ||
                        count > counts.expression_count - reference) {
                        status = LC_IMAGE_INVALID;
                        goto cleanup;
                    }
                    continue;
                }
                offset = reference & ~LC_IMAGE_SPECIALIZED_REF;
                if (offset > specialized_cursor ||
                    (offset < specialized_cursor &&
                     count > specialized_cursor - offset)) {
                    status = LC_IMAGE_INVALID;
                    goto cleanup;
                }
                if (offset == specialized_cursor) {
                    if (count > counts.specialized_expression_count -
                            specialized_cursor) {
                        status = LC_IMAGE_INVALID;
                        goto cleanup;
                    }
                    specialized_cursor += count;
                }
            }
        }
        if (specialized_cursor != counts.specialized_expression_count) {
            status = LC_IMAGE_INVALID;
            goto cleanup;
        }
        for (index = 0U; index < counts.node_count; ++index) {
            const lc_mixed_node *node = &graph->nodes[index];
            lc_node_eval_plan *plan = &graph->node_eval_plans[index];
            if (!lc_image_resolve_expression_reference(
                    eval_references[(size_t)index * 5U],
                    node->program_node_count,
                    graph->expression_nodes, counts.expression_count,
                    graph->specialized_expression_nodes,
                    counts.specialized_expression_count, 1,
                    &plan->normal_nodes
                ) || !lc_image_resolve_expression_reference(
                    eval_references[(size_t)index * 5U + 1U],
                    node->program_node_count,
                    graph->expression_nodes, counts.expression_count,
                    graph->specialized_expression_nodes,
                    counts.specialized_expression_count, 1,
                    &plan->clamped_nodes
                ) || !lc_image_resolve_expression_reference(
                    eval_references[(size_t)index * 5U + 2U],
                    node->program_node_count,
                    graph->expression_nodes, counts.expression_count,
                    graph->specialized_expression_nodes,
                    counts.specialized_expression_count, 1,
                    &plan->reset_nodes
                ) || !lc_image_resolve_expression_reference(
                    eval_references[(size_t)index * 5U + 3U],
                    node->program_node_count,
                    graph->expression_nodes, counts.expression_count,
                    graph->specialized_expression_nodes,
                    counts.specialized_expression_count, 1,
                    &plan->crossing_nodes
                ) || !lc_image_resolve_expression_reference(
                    eval_references[(size_t)index * 5U + 4U],
                    node->deposit_node_count,
                    graph->expression_nodes, counts.expression_count,
                    graph->specialized_expression_nodes,
                    counts.specialized_expression_count, 1,
                    &plan->deposit_nodes
                )) {
                status = LC_IMAGE_INVALID;
                goto cleanup;
            }
        }
    }
    for (index = 0U; index < counts.parameter_count; ++index)
        graph->parameters[index] = lc_image_read_real(&reader);
    for (index = 0U; index < counts.edge_count; ++index)
        lc_image_read_edge(&reader, &graph->edges[index]);
    for (index = 0U; index < counts.delivery_group_count; ++index)
        lc_image_read_delivery_group(&reader, &graph->delivery_groups[index]);
    for (index = 0U; index < counts.edge_count; ++index)
        graph->delivery_group_edges[index] = lc_image_read_u32(&reader);
    for (position = 0U; position <= counts.node_count; ++position)
        graph->outgoing_offsets[position] = lc_image_read_u64(&reader);

    if ((counts.flags & LC_IMAGE_HAS_LEGACY_PLASTICITY) != 0U) {
        for (index = 0U; index < counts.edge_count; ++index)
            lc_image_read_plasticity_rule(&reader, &graph->plasticity[index]);
    }
    for (index = 0U; index < counts.learning_program_count; ++index)
        lc_image_read_learning_program(
            &reader, &graph->learning_programs[index],
            &learning_references[
                (size_t)index * (LC_LEARNING_EVENT_COUNT + 1U)
            ]
        );
    position = 0U;
    for (index = 0U; index < counts.learning_program_count; ++index) {
        const lc_learning_program *program = &graph->learning_programs[index];
        uint32_t event;
        for (event = 0U; event < LC_LEARNING_EVENT_COUNT; ++event) {
            uint64_t reference = learning_references[
                (size_t)index * (LC_LEARNING_EVENT_COUNT + 1U) + event
            ];
            uint32_t count = program->events[event].node_count;
            if ((count == 0U && reference != LC_IMAGE_NULL_REF) ||
                (count > 0U &&
                 (reference != position ||
                  count > counts.learning_expression_count - position))) {
                status = LC_IMAGE_INVALID;
                goto cleanup;
            }
            position += count;
        }
        {
            uint64_t reference = learning_references[
                (size_t)index * (LC_LEARNING_EVENT_COUNT + 1U) +
                LC_LEARNING_EVENT_COUNT
            ];
            uint32_t count = program->observer.node_count;
            if ((count == 0U && reference != LC_IMAGE_NULL_REF) ||
                (count > 0U &&
                 (reference != position ||
                  count > counts.learning_expression_count - position))) {
                status = LC_IMAGE_INVALID;
                goto cleanup;
            }
            position += count;
        }
    }
    if (position != counts.learning_expression_count) {
        status = LC_IMAGE_INVALID;
        goto cleanup;
    }
    for (position = 0U; position < counts.learning_expression_count; ++position)
        lc_image_read_expr(
            &reader, &graph->learning_expression_nodes[position]
        );
    for (index = 0U; index < counts.learning_program_count; ++index) {
        lc_learning_program *program = &graph->learning_programs[index];
        uint32_t event;
        for (event = 0U; event < LC_LEARNING_EVENT_COUNT; ++event) {
            const lc_expr_node *nodes;
            if (!lc_image_resolve_expression_reference(
                    learning_references[
                        (size_t)index * (LC_LEARNING_EVENT_COUNT + 1U) + event
                    ],
                    program->events[event].node_count,
                    graph->learning_expression_nodes,
                    counts.learning_expression_count,
                    NULL, 0U, 0, &nodes
                )) {
                status = LC_IMAGE_INVALID;
                goto cleanup;
            }
            program->events[event].nodes = nodes;
        }
        {
            const lc_expr_node *nodes;
            if (!lc_image_resolve_expression_reference(
                    learning_references[
                        (size_t)index * (LC_LEARNING_EVENT_COUNT + 1U) +
                        LC_LEARNING_EVENT_COUNT
                    ],
                    program->observer.node_count,
                    graph->learning_expression_nodes,
                    counts.learning_expression_count,
                    NULL, 0U, 0, &nodes
                )) {
                status = LC_IMAGE_INVALID;
                goto cleanup;
            }
            program->observer.nodes = nodes;
        }
    }
    for (index = 0U; index < counts.learning_parameter_count; ++index)
        graph->learning_parameters[index] = lc_image_read_real(&reader);
    if ((counts.flags & LC_IMAGE_HAS_LEARNING_BINDINGS) != 0U) {
        for (index = 0U; index < counts.edge_count; ++index)
            lc_image_read_learning_binding(
                &reader, &graph->learning_bindings[index]
            );
    }
    if ((counts.flags & LC_IMAGE_HAS_LEARNING_OBSERVER_MAPS) != 0U) {
        for (index = 0U; index < counts.node_count; ++index)
            graph->learning_observer_program_by_node[index] =
                lc_image_read_u32(&reader);
        for (index = 0U; index < counts.node_count; ++index)
            graph->learning_observer_parameter_offset_by_node[index] =
                lc_image_read_u32(&reader);
    }
    if ((counts.flags & LC_IMAGE_HAS_PLASTIC_SLOT_MAP) != 0U) {
        for (index = 0U; index < counts.edge_count; ++index)
            graph->plastic_slot_by_edge[index] = lc_image_read_u32(&reader);
    }
    for (index = 0U; index < counts.plastic_edge_count; ++index)
        graph->plastic_edges[index] = lc_image_read_u32(&reader);
    if ((counts.flags & LC_IMAGE_HAS_WEIGHT_GROUPS) != 0U) {
        for (index = 0U; index < counts.plastic_edge_count; ++index)
            graph->weight_master_slots[index] = lc_image_read_u32(&reader);
        for (index = 0U; index < counts.plastic_edge_count; ++index)
            graph->weight_learning_scales[index] =
                lc_image_read_real(&reader);
    }
    if ((counts.flags & LC_IMAGE_HAS_INCOMING_PLASTIC) != 0U) {
        for (position = 0U; position <= counts.node_count; ++position)
            graph->incoming_plastic_offsets[position] =
                lc_image_read_u64(&reader);
        for (index = 0U; index < counts.plastic_edge_count; ++index)
            graph->incoming_plastic_slots[index] = lc_image_read_u32(&reader);
    }
    if ((counts.flags & LC_IMAGE_HAS_INCOMING_BATCHES) != 0U) {
        for (position = 0U; position <= counts.node_count; ++position)
            graph->incoming_learning_batch_offsets[position] =
                lc_image_read_u64(&reader);
        for (index = 0U;
             index < counts.incoming_learning_batch_count; ++index)
            lc_image_read_learning_batch(
                &reader, &graph->incoming_learning_batches[index]
            );
    }
    if ((counts.flags & LC_IMAGE_HAS_MODULATOR_INDEX) != 0U) {
        for (position = 0U; position <= counts.modulator_count; ++position)
            graph->modulator_offsets[position] = lc_image_read_u64(&reader);
        for (index = 0U; index < counts.plastic_edge_count; ++index)
            graph->modulator_slots[index] = lc_image_read_u32(&reader);
    }
    if ((counts.flags & LC_IMAGE_HAS_MODULATION_BATCHES) != 0U) {
        for (position = 0U; position <= counts.modulator_count; ++position)
            graph->modulation_learning_batch_offsets[position] =
                lc_image_read_u64(&reader);
        for (index = 0U;
             index < counts.modulation_learning_batch_count; ++index)
            lc_image_read_learning_batch(
                &reader, &graph->modulation_learning_batches[index]
            );
    }
    if (reader.failed || reader.position != size ||
        !lc_compiled_graph_validate_loaded(graph)) {
        status = LC_IMAGE_INVALID;
        goto cleanup;
    }
    *compiled = graph;
    graph = NULL;
    status = LC_OK;

cleanup:
    free(node_references);
    free(eval_references);
    free(learning_references);
    if (graph != NULL) lc_mixed_graph_destroy(graph);
    return status;
}

lc_status lc_compiled_graph_get_info(
    const lc_compiled_graph *compiled,
    lc_compiled_graph_info *info
) {
    if (compiled == NULL || info == NULL) return LC_INVALID_ARGUMENT;
    info->node_count = compiled->node_count;
    info->state_count = compiled->state_count;
    info->parameter_count = compiled->parameter_count;
    info->edge_count = compiled->edge_count;
    info->plastic_edge_count = compiled->plastic_edge_count;
    return LC_OK;
}

lc_status lc_compiled_graph_copy_node_layouts(
    const lc_compiled_graph *compiled,
    lc_compiled_node_layout *layouts,
    uint32_t layout_count
) {
    uint32_t node;
    if (compiled == NULL || layouts == NULL ||
        layout_count != compiled->node_count) {
        return LC_INVALID_ARGUMENT;
    }
    for (node = 0U; node < compiled->node_count; ++node) {
        layouts[node].state_offset = compiled->nodes[node].state_offset;
        layouts[node].state_count = compiled->nodes[node].state_count;
    }
    return LC_OK;
}
