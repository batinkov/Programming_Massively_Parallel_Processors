// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION must be defined in
// exactly one .c file. This pulls in the actual function bodies from the
// single-header libraries.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "blur.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [-r runs] [-b blur_size]\n",
                argv[0]);
        return 1;
    }

    int num_runs, blur_size;
    parse_args(argc, argv, &num_runs, &blur_size);

    // stbi_load decodes any supported format into a raw RGB pixel buffer.
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

    // Output is the same size as input — RGB, 3 bytes per pixel.
    unsigned char *out = malloc(rgb_bytes);
    if (!out) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        stbi_image_free(rgb);
        return 1;
    }

    // Time only the blur computation, not I/O.
    double *times = calloc((size_t)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        blur_cpu(rgb, out, w, h, blur_size);
        double end = get_time_sec();
        times[r] = end - start;
    }
    // Each output pixel reads (2*blur_size+1)^2 pixels from the input (3 bytes
    // each) and writes 3 bytes. For throughput we report just the input + output
    // size (same as grayscale: total bytes moved through memory).
    print_stats("CPU Blur", num_runs, times, rgb_bytes * 2, 0);

    if (!write_blurred_jpg(argv[1], w, h, out)) {
        free(times);
        free(out);
        stbi_image_free(rgb);
        return 1;
    }

    free(times);
    free(out);
    stbi_image_free(rgb);
    return 0;
}
