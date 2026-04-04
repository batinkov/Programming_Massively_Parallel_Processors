#ifndef VEC_ADD_H
#define VEC_ADD_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline void generate_array(float *arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        // RAND_MAX (2^31-1) exceeds float's exact integer range (2^24), so the
        // cast rounds it. This is fine — we just need arbitrary floats in [0,1].
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

static inline void vec_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

static inline int verify_result(const float *expected, const float *actual, int n) {
    for (int i = 0; i < n; i++) {
        if (expected[i] != actual[i]) {
            fprintf(stderr, "Mismatch at index %d: expected %f, got %f\n",
                    i, expected[i], actual[i]);
            return 0;
        }
    }
    return 1;
}

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline void print_stats(const char *label, int n, int num_runs,
                                const double *times) {
    double min_t = times[0];
    double sum = 0.0;
    for (int i = 0; i < num_runs; i++) {
        if (times[i] < min_t) min_t = times[i];
        sum += times[i];
    }
    double avg_t = sum / num_runs;
    double bytes = 3.0 * n * sizeof(float);
    double gb = bytes / (1 << 30);

    printf("[%s] n=%d, runs=%d\n", label, n, num_runs);
    printf("  Best time:  %.4f ms  (%.2f GB/s)\n", min_t * 1000, gb / min_t);
    printf("  Avg time:   %.4f ms  (%.2f GB/s)\n", avg_t * 1000, gb / avg_t);
}

#define DEFAULT_N    (100 * 1000 * 1000)
#define DEFAULT_RUNS 5
#define SEED_A       42
#define SEED_B       137

static inline void parse_args(int argc, char **argv, int *n, int *num_runs) {
    *n = DEFAULT_N;
    *num_runs = DEFAULT_RUNS;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "-n") == 0) {
                *n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0) {
                *num_runs = atoi(argv[++i]);
            }
        }
    }
}

#endif
