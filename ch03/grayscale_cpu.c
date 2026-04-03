// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION must be defined in
// exactly one .c file. This pulls in the actual function bodies from the
// single-header libraries; other files that include the headers get only
// declarations.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "grayscale.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [-r runs]\n", argv[0]);
        return 1;
    }

    int num_runs;
    parse_args(argc, argv, &num_runs);

    // stbi_load decodes any supported format (PNG, JPEG, BMP, etc.) into a raw
    // pixel buffer. The last argument (3) forces RGB output regardless of the
    // source format — e.g., RGBA images are converted to RGB, grayscale is
    // expanded to 3 channels.
    int w, h, channels;
    unsigned char *rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels)\n", argv[1], w, h, w * h);

    // Output is one byte per pixel (single-channel grayscale).
    int n = w * h;
    unsigned char *gray = malloc((size_t)n);
    if (!gray) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        stbi_image_free(rgb);
        return 1;
    }

    // Time only the grayscale conversion, not I/O. Multiple runs with best
    // time reported to minimize OS scheduling noise.
    double *times = calloc((size_t)num_runs, sizeof(double));
    for (int r = 0; r < num_runs; r++) {
        double start = get_time_sec();
        grayscale_cpu(rgb, gray, n);
        double end = get_time_sec();
        times[r] = end - start;
    }
    // 3 bytes read (RGB) + 1 byte written (gray) per pixel
    print_stats("CPU Grayscale", num_runs, times, (size_t)n * 4, 0);

    if (!write_gray_jpg(argv[1], w, h, gray)) {
        free(times);
        free(gray);
        stbi_image_free(rgb);
        return 1;
    }

    free(times);
    free(gray);
    stbi_image_free(rgb);
    return 0;
}
