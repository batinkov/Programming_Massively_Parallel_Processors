// Naive CUDA matrix multiplication with explicit memory copies.
// Each thread computes one element of C. Data travels over PCIe between
// host (CPU RAM) and device (GPU VRAM).

#include <cuda_runtime.h>
#include <cmath>

extern "C" {
#include "matrix_mul.h"
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void matmul_kernel(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

int main(int argc, char **argv) {
    int N, num_runs;
    parse_args(argc, argv, &N, &num_runs);
    if (N <= 0) {
        fprintf(stderr, "Usage: %s -n <matrix_size> [-r runs]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);
    printf("Matrix size: %dx%d (%d elements, %.1f MB per matrix)\n",
           N, N, N * N, (float)N * N * sizeof(float) / (1 << 20));

    size_t bytes = (size_t)N * N * sizeof(float);

    // Host matrices.
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices\n");
        return 1;
    }
    fill_random(h_A, N, SEED_A);
    fill_random(h_B, N, SEED_B);

    // Device matrices in VRAM.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy inputs from host to device over PCIe.
    double t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    double t1 = get_time_sec();
    printf("Host -> Device: %.4f ms\n", (t1 - t0) * 1000);

    // CPU baseline for speedup comparison and verification.
    float *expected = (float *)malloc(bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        matmul_cpu(h_A, h_B, expected, N);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // 2D block/grid configuration.
    int flat_block_size, min_grid_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                   matmul_kernel, 0, N * N));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    print_stats("CUDA GPU MatMul (naive)", N, num_runs, times, cpu_best);

    // Copy result back from device to host.
    t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    t1 = get_time_sec();
    printf("Device -> Host: %.4f ms\n", (t1 - t0) * 1000);

    if (verify_result(expected, h_C, N)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }

    free(expected);
    free(times);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
