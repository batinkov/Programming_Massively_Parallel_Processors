// Naive GPU box blur — each thread reads its entire patch directly from global
// memory. No shared memory optimization. This serves as a baseline to measure
// the impact of shared memory tiling later.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <hip/hip_runtime.h>
#include <cmath>

extern "C" {
#include "blur.h"
}

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Each thread computes one output pixel by averaging its (2*blur_size+1)^2
// neighborhood. All reads come from global memory — neighboring threads read
// overlapping pixels redundantly.
__global__ void blur_kernel(const unsigned char *in, unsigned char *out,
                            int w, int h, int blur_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= w || row >= h) return;

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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [-r runs] [-b blur_size]\n",
                argv[0]);
        return 1;
    }

    int num_runs, blur_size;
    parse_args(argc, argv, &num_runs, &blur_size);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);

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

    HIP_CHECK(hipHostRegister(rgb, rgb_bytes, hipHostRegisterDefault));

    // Output buffer: hipMallocManaged so both GPU and CPU can access it.
    unsigned char *out;
    HIP_CHECK(hipMallocManaged(&out, rgb_bytes));

    // CPU baseline for speedup comparison and verification.
    unsigned char *expected = (unsigned char *)malloc(rgb_bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(out));
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        blur_cpu(rgb, expected, w, h, blur_size);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // 2D block/grid configuration.
    int flat_block_size, min_grid_size;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                 blur_kernel, 0, n));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        HIP_CHECK(hipEventRecord(start));
        blur_kernel<<<grid, block>>>(rgb, out, w, h, blur_size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_stats("HIP GPU Blur (naive)", num_runs, times, rgb_bytes * 2, cpu_best);

    if (verify_result(expected, out, n * 3)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(out));
        return 1;
    }

    if (!write_blurred_jpg(argv[1], w, h, out)) {
        free(expected);
        free(times);
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(out));
        return 1;
    }

    free(expected);
    free(times);
    HIP_CHECK(hipHostUnregister(rgb));
    stbi_image_free(rgb);
    HIP_CHECK(hipFree(out));
    return 0;
}
