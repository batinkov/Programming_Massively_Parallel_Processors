#include <string.h>
#include "vec_add.h"

int main(int argc, char **argv) {
    int n, num_runs;
    parse_args(argc, argv, &n, &num_runs);

    float *a = malloc((size_t)n * sizeof(float));
    float *b = malloc((size_t)n * sizeof(float));
    float *c = malloc((size_t)n * sizeof(float));
    if (!a || !b || !c) {
        fprintf(stderr, "Failed to allocate memory for %d elements\n", n);
        return 1;
    }

    generate_array(a, n, SEED_A);
    generate_array(b, n, SEED_B);

    double *times = calloc((size_t)num_runs, sizeof(double));

    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        vec_add_cpu(a, b, c, n);
        double end = get_time_sec();
        times[r] = end - start;
    }

    print_stats("CPU Sequential", n, num_runs, times);

    free(times);
    free(a);
    free(b);
    free(c);
    return 0;
}
