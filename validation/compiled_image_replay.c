/* Replay independently reset episodes using only a compiled image and C. */
#include "lacuna.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void check(lc_status status)
{
    if (status != LC_OK) {
        fail(lc_status_string(status));
    }
}

static void read_bytes(FILE *file, void *value, size_t size)
{
    if (fread(value, 1, size, file) != size) {
        fail("truncated replay input");
    }
}

static uint64_t read_integer(FILE *file, size_t size)
{
    uint8_t bytes[8];
    uint64_t value = 0;
    read_bytes(file, bytes, size);
    for (size_t i = 0; i < size; ++i) {
        value |= ((uint64_t)bytes[i]) << (8 * i);
    }
    return value;
}

static double read_double(FILE *file)
{
    uint64_t bits = read_integer(file, 8);
    double value;
    memcpy(&value, &bits, sizeof(value));
    if (!isfinite(value)) {
        fail("nonfinite replay value");
    }
    return value;
}

static void write_integer(FILE *file, uint64_t value, size_t size)
{
    uint8_t bytes[8];
    for (size_t i = 0; i < size; ++i) {
        bytes[i] = (uint8_t)(value >> (8 * i));
    }
    if (fwrite(bytes, 1, size, file) != size) {
        fail("cannot write replay output");
    }
}

static void write_double(FILE *file, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    write_integer(file, bits, 8);
}

static void *allocate(uint64_t count, size_t width)
{
    void *value;
    if (count > SIZE_MAX / width) {
        fail("replay allocation overflow");
    }
    value = calloc((size_t)(count ? count : 1), width);
    if (value == NULL) {
        fail("replay allocation failed");
    }
    return value;
}

int main(int argc, char **argv)
{
    FILE *image_file, *input, *output;
    uint8_t *image;
    char magic[8];
    long image_size;
    uint32_t episodes, node_count, state_count;
    lc_compiled_graph *graph = NULL;
    lc_compiled_graph_info info;
    lc_mixed_run *run = NULL;
    lc_run_config config = {0};
    double *initial, *state, *last;
    lc_output_spike *spikes;

    if (argc != 4 || sizeof(double) != 8) {
        fail("usage: compiled_image_replay image inputs outputs");
    }
    image_file = fopen(argv[1], "rb");
    if (image_file == NULL || fseek(image_file, 0, SEEK_END) != 0) {
        fail("cannot open compiled image");
    }
    image_size = ftell(image_file);
    if (image_size <= 0 || fseek(image_file, 0, SEEK_SET) != 0) {
        fail("invalid compiled image size");
    }
    image = allocate((uint64_t)image_size, 1);
    read_bytes(image_file, image, (size_t)image_size);
    fclose(image_file);
    check(lc_compiled_graph_deserialize(image, (uint64_t)image_size, &graph));
    free(image);
    check(lc_compiled_graph_get_info(graph, &info));

    input = fopen(argv[2], "rb");
    if (input == NULL) {
        fail("cannot open replay input");
    }
    read_bytes(input, magic, 8);
    if (memcmp(magic, "LCRPLY01", 8) != 0) {
        fail("invalid replay magic");
    }
    episodes = (uint32_t)read_integer(input, 4);
    node_count = (uint32_t)read_integer(input, 4);
    state_count = (uint32_t)read_integer(input, 4);
    config.t_end = read_double(input);
    config.queue_capacity = read_integer(input, 8);
    config.output_capacity = read_integer(input, 8);
    config.same_time_cascade_limit = (uint32_t)read_integer(input, 4);
    if (episodes == 0 || node_count != info.node_count ||
        state_count != info.state_count || config.t_end < 0 ||
        config.output_capacity == 0 || config.queue_capacity == 0 ||
        config.same_time_cascade_limit == 0) {
        fail("invalid replay header");
    }
    initial = allocate(state_count, sizeof(double));
    state = allocate(state_count, sizeof(double));
    last = allocate(node_count, sizeof(double));
    spikes = allocate(config.output_capacity, sizeof(*spikes));
    for (uint32_t i = 0; i < state_count; ++i) {
        initial[i] = read_double(input);
    }
    check(lc_mixed_run_create(graph, initial, state_count, last, node_count, &run));
    output = fopen(argv[3], "wb");
    if (output == NULL || fwrite("LCOUT001", 1, 8, output) != 8) {
        fail("cannot open replay output");
    }
    write_integer(output, episodes, 4);
    write_integer(output, node_count, 4);
    write_integer(output, state_count, 4);
    for (uint32_t episode = 0; episode < episodes; ++episode) {
        uint32_t count = (uint32_t)read_integer(input, 4);
        lc_mixed_input_spike *events = allocate(count, sizeof(*events));
        uint64_t emitted = 0;
        lc_run_stats stats;
        lc_network_error error;
        for (uint32_t i = 0; i < count; ++i) {
            events[i].t = read_double(input);
            events[i].node = (uint32_t)read_integer(input, 4);
            events[i].deposit_kind = (uint32_t)read_integer(input, 4);
            events[i].target = (uint32_t)read_integer(input, 4);
            events[i].value = read_double(input);
            if (events[i].node >= node_count || events[i].t < 0 ||
                events[i].t > config.t_end ||
                (i > 0 && events[i].t < events[i - 1].t)) {
                fail("invalid replay event");
            }
        }
        memset(last, 0, node_count * sizeof(double));
        check(lc_mixed_run_reset(run, initial, state_count, last, node_count));
        check(lc_mixed_run_execute(run, events, count, NULL, 0, &config,
                                  spikes, &emitted, &stats, &error));
        check(lc_mixed_run_copy_state(run, state, state_count, last, node_count));
        write_integer(output, emitted, 8);
        for (uint64_t i = 0; i < emitted; ++i) {
            write_double(output, spikes[i].t);
            write_integer(output, spikes[i].node, 4);
        }
        for (uint32_t i = 0; i < state_count; ++i) {
            write_double(output, state[i]);
        }
        for (uint32_t i = 0; i < node_count; ++i) {
            write_double(output, last[i]);
        }
        free(events);
    }
    if (fgetc(input) != EOF || ferror(input)) {
        fail("unexpected trailing replay data");
    }
    fclose(input);
    if (fclose(output) != 0) {
        fail("cannot finish replay output");
    }
    free(initial);
    free(state);
    free(last);
    free(spikes);
    lc_mixed_run_destroy(run);
    lc_mixed_graph_destroy(graph);
    return 0;
}
