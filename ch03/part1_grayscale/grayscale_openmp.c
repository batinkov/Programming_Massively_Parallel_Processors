// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION must be defined in
// exactly one .c file. This pulls in the actual function bodies from the
// single-header libraries; other files that include the headers get only
// declarations.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <omp.h>
#include "grayscale.h"

// Same luminance formula as grayscale_cpu in grayscale.h, but with OpenMP
// parallelization. Each pixel is independent — no shared state, no reduction —
// so a simple parallel for is sufficient.
static void grayscale_omp(const unsigned char *rgb, unsigned char *gray,
                          int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int idx = i * 3;
        gray[i] = (unsigned char)(0.299f * rgb[idx]
                                + 0.587f * rgb[idx + 1]
                                + 0.114f * rgb[idx + 2]);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [-r runs]\n", argv[0]);
        return 1;
    }

    int num_runs;
    parse_args(argc, argv, &num_runs);

    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // stbi_load decodes any supported format into a raw RGB pixel buffer.
    // The last argument (3) forces 3-channel output regardless of the source.
    int w, h, channels;
    unsigned char *rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels)\n", argv[1], w, h, w * h);

    int n = w * h;
    unsigned char *gray = malloc((size_t)n);
    if (!gray) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        stbi_image_free(rgb);
        return 1;
    }

    // Run the single-threaded CPU version first. This serves two purposes:
    // 1. Produces the expected output for verification
    // 2. Establishes a baseline time for speedup calculation
    unsigned char *expected = malloc((size_t)n);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        free(gray);
        stbi_image_free(rgb);
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        grayscale_cpu(rgb, expected, n);
        double end = get_time_sec();
        double t = end - start;
        if (t < cpu_best) cpu_best = t;
    }

    // Time only the grayscale conversion, not I/O.
    double *times = calloc((size_t)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        grayscale_omp(rgb, gray, n);
        double end = get_time_sec();
        times[r] = end - start;
    }
    // 3 bytes read (RGB) + 1 byte written (gray) per pixel
    print_stats("OpenMP Grayscale", num_runs, times, (size_t)n * 4, cpu_best);

    // Verify against single-threaded CPU (already computed above).
    // Grayscale is element-wise with no reordering, so results must be
    // bit-identical across all implementations.
    if (verify_result(expected, gray, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        free(gray);
        stbi_image_free(rgb);
        return 1;
    }

    if (!write_gray_jpg(argv[1], w, h, gray)) {
        free(expected);
        free(times);
        free(gray);
        stbi_image_free(rgb);
        return 1;
    }

    free(expected);
    free(times);
    free(gray);
    stbi_image_free(rgb);
    return 0;
}
