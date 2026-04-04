// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION must be defined in
// exactly one .c file. This pulls in the actual function bodies from the
// single-header libraries.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <hip/hip_runtime.h>
#include <cmath>

extern "C" {
#include "grayscale.h"
}

// Every HIP call returns an error code. This macro checks it and aborts with
// a descriptive message if something went wrong.
#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
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
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);

    // stbi_load decodes any supported format into a raw RGB pixel buffer.
    // This allocates with malloc — we'll register it with hipHostRegister
    // so the GPU can read it directly without a copy.
    int w, h, channels;
    unsigned char *rgb = stbi_load(argv[1], &w, &h, &channels, 3);
    if (!rgb) {
        fprintf(stderr, "Failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return 1;
    }
    printf("Loaded %s: %dx%d (%d pixels)\n", argv[1], w, h, w * h);

    int n = w * h;
    size_t rgb_bytes = (size_t)n * 3;

    // Register the stbi-allocated buffer with the HIP runtime. On integrated
    // GPUs (like the Radeon 860M) this is essentially free — same physical RAM,
    // just a page table update so the GPU can see the address. No data is
    // copied.
    HIP_CHECK(hipHostRegister(rgb, rgb_bytes, hipHostRegisterDefault));

    // Output buffer: hipMallocManaged so both the GPU (kernel writes) and the
    // CPU (verification + JPEG writing) can access it without explicit copies.
    unsigned char *gray;
    HIP_CHECK(hipMallocManaged(&gray, (size_t)n));

    // Run single-threaded CPU baseline for speedup comparison and verification.
    unsigned char *expected = (unsigned char *)malloc((size_t)n);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(gray));
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        grayscale_cpu(rgb, expected, n);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // Query the runtime for the optimal 1D block size, then split it into a 2D
    // block that matches the image's spatial layout. We take the square root to
    // get roughly equal dimensions in both axes.
    int flat_block_size, min_grid_size;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                 grayscale_kernel, 0, n));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    // Ceiling division in each dimension to cover all pixels.
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    // Create GPU event markers for accurate kernel timing. Events are recorded
    // on the GPU timeline, so they measure actual kernel execution — not
    // CPU-side launch overhead.
    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        HIP_CHECK(hipEventRecord(start));

        // Launch the kernel with 2D grid and block dimensions. Each thread
        // processes one pixel at (row, col).
        grayscale_kernel<<<grid, block>>>(rgb, gray, w, h);

        HIP_CHECK(hipEventRecord(stop));
        // Block the CPU until the GPU finishes the kernel.
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // 3 bytes read (RGB) + 1 byte written (gray) per pixel
    print_stats("HIP GPU Grayscale", num_runs, times, (size_t)n * 4, cpu_best);

    // Verify against single-threaded CPU (already computed above).
    // After hipEventSynchronize, the GPU is done writing to gray. With managed
    // memory the CPU can read it directly.
    if (verify_result(expected, gray, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(gray));
        return 1;
    }

    if (!write_gray_jpg(argv[1], w, h, gray)) {
        free(expected);
        free(times);
        HIP_CHECK(hipHostUnregister(rgb));
        stbi_image_free(rgb);
        HIP_CHECK(hipFree(gray));
        return 1;
    }

    free(expected);
    free(times);
    // Unregister before stbi_image_free — must unregister while the pointer is
    // still valid.
    HIP_CHECK(hipHostUnregister(rgb));
    stbi_image_free(rgb);
    HIP_CHECK(hipFree(gray));
    return 0;
}
