#include <omp.h>
#include <string.h>
#include "vec_add.h"

static void vec_add_openmp(const float *a, const float *b, float *c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    int n, num_runs;
    parse_args(argc, argv, &n, &num_runs);

    printf("OpenMP threads: %d\n", omp_get_max_threads());

    float *a = malloc((size_t)n * sizeof(float));
    float *b = malloc((size_t)n * sizeof(float));
    float *c = malloc((size_t)n * sizeof(float));
    float *expected = malloc((size_t)n * sizeof(float));
    if (!a || !b || !c || !expected) {
        fprintf(stderr, "Failed to allocate memory for %d elements\n", n);
        return 1;
    }

    generate_array(a, n, SEED_A);
    generate_array(b, n, SEED_B);

    double *times = calloc((size_t)num_runs, sizeof(double));

    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        vec_add_openmp(a, b, c, n);
        double end = get_time_sec();
        times[r] = end - start;
    }

    print_stats("OpenMP", n, num_runs, times);

    vec_add_cpu(a, b, expected, n);
    if (verify_result(expected, c, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(times);
        free(expected);
        free(a);
        free(b);
        free(c);
        return 1;
    }

    free(times);
    free(expected);
    free(a);
    free(b);
    free(c);
    return 0;
}
