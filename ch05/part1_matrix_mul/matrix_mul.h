#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEED_A 42
#define SEED_B 137
#define DEFAULT_RUNS 3

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline void fill_random(float *M, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Naive CPU matrix multiply — used as reference for verification.
static inline void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

static inline void transpose(const float *B, float *Bt, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            Bt[col * N + row] = B[row * N + col];
        }
    }
}

// Compare against reference result. Tolerance accounts for floating-point
// ordering differences (GPU may use FMA, different accumulation order, etc.).
static inline int verify_result(const float *expected, const float *actual,
                                int N) {
    for (int i = 0; i < N * N; i++) {
        float diff = actual[i] - expected[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-2f) {
            int row = i / N;
            int col = i % N;
            fprintf(stderr, "Mismatch at (%d,%d): expected %.4f, got %.4f (diff %.4f)\n",
                    row, col, expected[i], actual[i], diff);
            return 0;
        }
    }
    return 1;
}

// Pass cpu_best <= 0 to skip speedup reporting.
static inline void print_stats(const char *label, int N, int num_runs,
                                const double *times, double cpu_best) {
    double min_t = times[0];
    double sum = 0.0;
    for (int i = 0; i < num_runs; i++) {
        if (times[i] < min_t) min_t = times[i];
        sum += times[i];
    }
    double avg_t = sum / num_runs;
    // 2*N^3 floating-point ops (N^3 multiplies + N^3 adds).
    double flops = 2.0 * (double)N * N * N;
    printf("[%s] N=%d, runs=%d\n", label, N, num_runs);
    printf("  Best: %.4f ms  (%.2f GFLOPS)\n", min_t * 1000, flops / (min_t * 1e9));
    printf("  Avg:  %.4f ms  (%.2f GFLOPS)\n", avg_t * 1000, flops / (avg_t * 1e9));
    if (cpu_best > 0) {
        printf("  Speedup: %.2fx (best vs CPU best)\n", cpu_best / min_t);
    }
}

static inline void parse_args(int argc, char **argv, int *N, int *num_runs) {
    *N = 0;
    *num_runs = DEFAULT_RUNS;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            *N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *num_runs = atoi(argv[++i]);
        }
    }
}

#endif
