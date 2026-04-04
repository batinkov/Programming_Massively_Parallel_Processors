// OpenMP parallel box blur. Parallelizes over rows — each thread gets a chunk
// of rows to process. Each output pixel is independent (reads from input,
// writes to separate output), so no synchronization is needed.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <omp.h>
#include "blur.h"

static void blur_omp(const unsigned char *in, unsigned char *out,
                     int w, int h, int blur_size) {
    #pragma omp parallel for
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            int count = 0;

            int row_start = row - blur_size;
            int row_end = row + blur_size;
            int col_start = col - blur_size;
            int col_end = col + blur_size;

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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [-r runs] [-b blur_size]\n",
                argv[0]);
        return 1;
    }

    int num_runs, blur_size;
    parse_args(argc, argv, &num_runs, &blur_size);

    printf("OpenMP threads: %d\n", omp_get_max_threads());

    int w, h, channels;
    unsigned char *rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels), blur_size=%d (%dx%d patch)\n",
           argv[1], w, h, w * h, blur_size,
           2 * blur_size + 1, 2 * blur_size + 1);

    int n = w * h;
    size_t rgb_bytes = (size_t)n * 3;

    unsigned char *out = malloc(rgb_bytes);
    if (!out) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        stbi_image_free(rgb);
        return 1;
    }

    // Run single-threaded CPU baseline for speedup comparison and verification.
    unsigned char *expected = malloc(rgb_bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        free(out);
        stbi_image_free(rgb);
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        blur_cpu(rgb, expected, w, h, blur_size);
        double end = get_time_sec();
        double t = end - start;
        if (t < cpu_best) cpu_best = t;
    }

    double *times = calloc((unsigned)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        blur_omp(rgb, out, w, h, blur_size);
        double end = get_time_sec();
        times[r] = end - start;
    }
    print_stats("OpenMP Blur", num_runs, times, rgb_bytes * 2, cpu_best);

    // Verify against single-threaded CPU (already computed above).
    if (verify_result(expected, out, n * 3)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        free(out);
        stbi_image_free(rgb);
        return 1;
    }

    if (!write_blurred_jpg(argv[1], w, h, out)) {
        free(expected);
        free(times);
        free(out);
        stbi_image_free(rgb);
        return 1;
    }

    free(expected);
    free(times);
    free(out);
    stbi_image_free(rgb);
    return 0;
}
