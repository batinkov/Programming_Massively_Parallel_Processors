// CUDA box blur with explicit memory copies. Naive version — each thread reads
// its entire patch from global memory. No shared memory optimization.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cmath>

extern "C" {
#include "blur.h"
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Each thread computes one output pixel by averaging its (2*blur_size+1)^2
// neighborhood. All reads come from global memory (VRAM).
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

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);

    int w, h, channels;
    unsigned char *h_rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!h_rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels), blur_size=%d (%dx%d patch)\n",
           argv[1], w, h, w * h, blur_size,
           2 * blur_size + 1, 2 * blur_size + 1);

    int n = w * h;
    size_t rgb_bytes = (size_t)n * 3;

    // Host output buffer.
    unsigned char *h_out = (unsigned char *)malloc(rgb_bytes);
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host output buffer\n");
        stbi_image_free(h_rgb);
        return 1;
    }

    // Device (GPU) memory.
    unsigned char *d_rgb, *d_out;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, rgb_bytes));

    // Copy input from host to device over PCIe.
    double t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));
    double t1 = get_time_sec();
    printf("Host -> Device: %.4f ms\n", (t1 - t0) * 1000);

    // CPU baseline for speedup comparison and verification.
    unsigned char *expected = (unsigned char *)malloc(rgb_bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
        stbi_image_free(h_rgb);
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        blur_cpu(h_rgb, expected, w, h, blur_size);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // 2D block/grid configuration.
    int flat_block_size, min_grid_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                   blur_kernel, 0, n));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        blur_kernel<<<grid, block>>>(d_rgb, d_out, w, h, blur_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    print_stats("CUDA GPU Blur (naive)", num_runs, times, rgb_bytes * 2, cpu_best);

    // Copy result back from device to host.
    t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, rgb_bytes, cudaMemcpyDeviceToHost));
    t1 = get_time_sec();
    printf("Device -> Host: %.4f ms\n", (t1 - t0) * 1000);

    if (verify_result(expected, h_out, n * 3)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
        stbi_image_free(h_rgb);
        return 1;
    }

    if (!write_blurred_jpg(argv[1], w, h, h_out)) {
        free(expected);
        free(times);
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
        stbi_image_free(h_rgb);
        return 1;
    }

    free(expected);
    free(times);
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_out));
    free(h_out);
    stbi_image_free(h_rgb);
    return 0;
}
