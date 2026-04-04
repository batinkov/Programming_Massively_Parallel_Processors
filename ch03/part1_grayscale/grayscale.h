#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline void grayscale_cpu(const unsigned char *rgb, unsigned char *gray,
                                 int n) {
    for (int i = 0; i < n; i++) {
        int idx = i * 3;
        gray[i] = (unsigned char)(0.299f * rgb[idx]
                                + 0.587f * rgb[idx + 1]
                                + 0.114f * rgb[idx + 2]);
    }
}

static inline int verify_result(const unsigned char *expected,
                                const unsigned char *actual, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(expected[i] - actual[i]) > 1) {
            fprintf(stderr, "Mismatch at pixel %d: expected %d, got %d (diff %d)\n",
                    i, expected[i], actual[i], abs(expected[i] - actual[i]));
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

// Pass cpu_best <= 0 to skip speedup reporting (e.g. for the CPU baseline).
static inline void print_stats(const char *label, int num_runs,
                                const double *times, size_t bytes,
                                double cpu_best) {
    double min_t = times[0];
    double sum = 0.0;
    for (int i = 0; i < num_runs; i++) {
        if (times[i] < min_t) min_t = times[i];
        sum += times[i];
    }
    double avg_t = sum / num_runs;
    double gb = (double)bytes / (1 << 30);
    printf("[%s] runs=%d\n", label, num_runs);
    printf("  Best: %.4f ms  (%.2f GB/s)\n", min_t * 1000, gb / min_t);
    printf("  Avg:  %.4f ms  (%.2f GB/s)\n", avg_t * 1000, gb / avg_t);
    if (cpu_best > 0) {
        printf("  Speedup: %.2fx (best vs CPU best)\n", cpu_best / min_t);
    }
}

#define DEFAULT_RUNS 5

static inline void parse_args(int argc, char **argv, int *num_runs) {
    *num_runs = DEFAULT_RUNS;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *num_runs = atoi(argv[++i]);
        }
    }
}

static inline int write_gray_jpg(const char *input_path, int w, int h,
                                  const unsigned char *gray) {
    char out_name[512];
    const char *dot = strrchr(input_path, '.');
    if (dot) {
        int base_len = (int)(dot - input_path);
        snprintf(out_name, sizeof(out_name), "%.*s_gray.jpg", base_len,
                 input_path);
    } else {
        snprintf(out_name, sizeof(out_name), "%s_gray.jpg", input_path);
    }

    if (!stbi_write_jpg(out_name, w, h, 1, gray, 95)) {
        fprintf(stderr, "Failed to write '%s'\n", out_name);
        return 0;
    }
    printf("Written to %s\n", out_name);
    return 1;
}

#endif
