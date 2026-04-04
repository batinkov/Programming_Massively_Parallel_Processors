#ifndef BLUR_H
#define BLUR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Box blur: each output pixel is the average of a (2*blur_size+1)^2 patch
// centered on it. Boundary pixels use only the valid neighbors (no padding).
// Operates on RGB images — each channel is blurred independently.
static inline void blur_cpu(const unsigned char *in, unsigned char *out,
                            int w, int h, int blur_size) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            int count = 0;

            int row_start = row - blur_size;
            int row_end = row + blur_size;
            int col_start = col - blur_size;
            int col_end = col + blur_size;

            // Clamp to image boundaries.
            if (row_start < 0) row_start = 0;
            if (row_end >= h) row_end = h - 1;
            if (col_start < 0) col_start = 0;
            if (col_end >= w) col_end = w - 1;

            for (int r = row_start; r <= row_end; r++) {
                for (int c = col_start; c <= col_end; c++) {
                    int idx = (r * w + c) * 3;
                    r_sum += in[idx];
                    g_sum += in[idx + 1];
                    b_sum += in[idx + 2];
                    count++;
                }
            }

            int out_idx = (row * w + col) * 3;
            out[out_idx]     = (unsigned char)(r_sum / count);
            out[out_idx + 1] = (unsigned char)(g_sum / count);
            out[out_idx + 2] = (unsigned char)(b_sum / count);
        }
    }
}

static inline int verify_result(const unsigned char *expected,
                                const unsigned char *actual, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(expected[i] - actual[i]) > 1) {
            fprintf(stderr, "Mismatch at byte %d: expected %d, got %d (diff %d)\n",
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
#define DEFAULT_BLUR_SIZE 1

static inline void parse_args(int argc, char **argv, int *num_runs,
                               int *blur_size) {
    *num_runs = DEFAULT_RUNS;
    *blur_size = DEFAULT_BLUR_SIZE;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            *blur_size = atoi(argv[++i]);
        }
    }
}

static inline int write_blurred_jpg(const char *input_path, int w, int h,
                                     const unsigned char *rgb) {
    char out_name[512];
    const char *dot = strrchr(input_path, '.');
    if (dot) {
        int base_len = (int)(dot - input_path);
        snprintf(out_name, sizeof(out_name), "%.*s_blur.jpg", base_len,
                 input_path);
    } else {
        snprintf(out_name, sizeof(out_name), "%s_blur.jpg", input_path);
    }

    if (!stbi_write_jpg(out_name, w, h, 3, rgb, 95)) {
        fprintf(stderr, "Failed to write '%s'\n", out_name);
        return 0;
    }
    printf("Written to %s\n", out_name);
    return 1;
}

#endif
