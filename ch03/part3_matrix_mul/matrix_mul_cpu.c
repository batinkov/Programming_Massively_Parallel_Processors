// Naive sequential CPU matrix multiplication: C = A * B.
// A and B are filled with random floats. Serves as the baseline for all other
// implementations.

#include "matrix_mul.h"

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
    float *C = malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return 1;
    }

    fill_random(A, N, SEED_A);
    fill_random(B, N, SEED_B);

    double *times = calloc((size_t)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        matmul_cpu(A, B, C, N);
        double end = get_time_sec();
        times[r] = end - start;
    }
    print_stats("CPU MatMul", N, num_runs, times, 0);
    printf("Verification: baseline (reference implementation)\n");

    free(times);
    free(A);
    free(B);
    free(C);
    return 0;
}
