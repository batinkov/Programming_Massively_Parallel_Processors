// CUDA grayscale conversion with explicit memory copies. For discrete GPUs
// (like the RTX 3060) with dedicated VRAM, data must travel over PCIe between
// host (CPU) and device (GPU) memory.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cmath>

extern "C" {
#include "grayscale.h"
}

// Every CUDA call returns an error code. This macro checks it and aborts with
// a descriptive message if something went wrong.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// __global__ marks this as a GPU kernel. Each thread converts one pixel from
// RGB to grayscale using the ITU-R BT.601 luminance weights.
//
// The thread/block structure is 2D to match the image's spatial layout:
//   threadIdx.x / blockIdx.x -> columns
//   threadIdx.y / blockIdx.y -> rows
// Threads in the same warp (adjacent threadIdx.x) access adjacent columns,
// which are contiguous in row-major memory — this gives coalesced reads.
__global__ void grayscale_kernel(const unsigned char *rgb, unsigned char *gray,
                                 int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int i = row * w + col;
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

    // Query GPU properties and print the device name.
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);

    // stbi_load decodes any supported format into a raw RGB pixel buffer.
    int w, h, channels;
    unsigned char *h_rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!h_rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels)\n", argv[1], w, h, w * h);

    int n = w * h;
    size_t rgb_bytes = (size_t)n * 3;
    size_t gray_bytes = (size_t)n;

    // Host output buffer for copying results back from the GPU.
    unsigned char *h_gray = (unsigned char *)malloc(gray_bytes);
    if (!h_gray) {
        fprintf(stderr, "Failed to allocate host output buffer\n");
        stbi_image_free(h_rgb);
        return 1;
    }

    // Device (GPU) memory — allocated in VRAM with cudaMalloc. These pointers
    // are only valid on the GPU and cannot be dereferenced on the CPU.
    unsigned char *d_rgb, *d_gray;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));

    // Copy input image from host (CPU RAM) to device (GPU VRAM) over PCIe.
    // cudaMemcpy is synchronous — the CPU blocks until the transfer completes.
    double t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));
    double t1 = get_time_sec();
    printf("Host -> Device: %.4f ms\n", (t1 - t0) * 1000);

    // CPU baseline for speedup comparison and verification.
    unsigned char *expected = (unsigned char *)malloc(gray_bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        free(h_gray);
        stbi_image_free(h_rgb);
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        grayscale_cpu(h_rgb, expected, n);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // Query the runtime for the optimal 1D block size, then split it into a 2D
    // block that matches the image's spatial layout.
    int flat_block_size, min_grid_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                   grayscale_kernel, 0, n));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    // Ceiling division in each dimension to cover all pixels.
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    // Create GPU event markers for accurate kernel timing.
    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, w, h);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 3 bytes read (RGB) + 1 byte written (gray) per pixel
    print_stats("CUDA GPU Grayscale", num_runs, times, (size_t)n * 4, cpu_best);

    // Copy result back from device (GPU VRAM) to host (CPU RAM) over PCIe.
    t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
    t1 = get_time_sec();
    printf("Device -> Host: %.4f ms\n", (t1 - t0) * 1000);

    // Verify against CPU baseline. Allow ±1 tolerance for GPU/CPU floating-point
    // rounding differences (FMA vs separate multiply-add).
    if (verify_result(expected, h_gray, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        free(h_gray);
        stbi_image_free(h_rgb);
        return 1;
    }

    if (!write_gray_jpg(argv[1], w, h, h_gray)) {
        free(expected);
        free(times);
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        free(h_gray);
        stbi_image_free(h_rgb);
        return 1;
    }

    free(expected);
    free(times);
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    free(h_gray);
    stbi_image_free(h_rgb);
    return 0;
}
