// CPU matrix multiplication with B transposed before the multiply.
// Instead of accessing B column-wise (stride N, cache misses), we transpose B
// first so both A and Bt are accessed row-wise (sequential, cache-friendly).

#include "matrix_mul.h"

// C = A * B, but using pre-transposed Bt.
// A[row][k] * Bt[col][k] — both accesses are row-sequential.
static void matmul_transposed(const float *A, const float *Bt, float *C,
                               int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * Bt[col * N + k];
            }
            C[row * N + col] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N, num_runs;
    parse_args(argc, argv, &N, &num_runs);
    if (N <= 0) {
        fprintf(stderr, "Usage: %s -n <matrix_size> [-r runs]\n", argv[0]);
        return 1;
    }

    printf("Matrix size: %dx%d (%d elements, %.1f MB per matrix)\n",
           N, N, N * N, (float)N * N * sizeof(float) / (1 << 20));

    size_t bytes = (size_t)N * N * sizeof(float);

    float *A = malloc(bytes);
    float *B = malloc(bytes);
    float *Bt = malloc(bytes);
    float *C = malloc(bytes);
    float *expected = malloc(bytes);
    if (!A || !B || !Bt || !C || !expected) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return 1;
    }

    fill_random(A, N, SEED_A);
    fill_random(B, N, SEED_B);

    // Transpose B once (O(N^2) — negligible vs O(N^3) multiply).
    transpose(B, Bt, N);

    // CPU baseline for speedup.
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        matmul_cpu(A, B, expected, N);
        double end = get_time_sec();
        double t = end - start;
        if (t < cpu_best) cpu_best = t;
    }

    // Transposed version.
    double *times = calloc((size_t)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        matmul_transposed(A, Bt, C, N);
        double end = get_time_sec();
        times[r] = end - start;
    }
    print_stats("CPU MatMul (transposed B)", N, num_runs, times, cpu_best);

    if (verify_result(expected, C, N)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(times);
        free(expected);
        free(Bt);
        free(A);
        free(B);
        free(C);
        return 1;
    }

    free(times);
    free(expected);
    free(Bt);
    free(A);
    free(B);
    free(C);
    return 0;
}
