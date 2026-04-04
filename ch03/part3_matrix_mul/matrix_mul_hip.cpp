// Naive GPU matrix multiplication: C = A * B.
// Each thread computes one element of C by taking the dot product of a row of A
// and a column of B. No shared memory, no tiling.
//
// Unlike on the CPU, the column access of B is less problematic here — adjacent
// threads in a warp access adjacent columns (B[k*N+col] and B[k*N+col+1]),
// which the memory controller can coalesce into a single transaction.

#include <hip/hip_runtime.h>
#include <cmath>

extern "C" {
#include "matrix_mul.h"
}

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
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

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);
    printf("Matrix size: %dx%d (%d elements, %.1f MB per matrix)\n",
           N, N, N * N, (float)N * N * sizeof(float) / (1 << 20));

    size_t bytes = (size_t)N * N * sizeof(float);

    // Host matrices.
    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    if (!A || !B) {
        fprintf(stderr, "Failed to allocate host matrices\n");
        return 1;
    }
    fill_random(A, N, SEED_A);
    fill_random(B, N, SEED_B);

    // Register host memory so the GPU can read it directly.
    HIP_CHECK(hipHostRegister(A, bytes, hipHostRegisterDefault));
    HIP_CHECK(hipHostRegister(B, bytes, hipHostRegisterDefault));

    // Output: managed memory accessible by both GPU and CPU.
    float *C;
    HIP_CHECK(hipMallocManaged(&C, bytes));

    // CPU baseline for speedup comparison and verification.
    float *expected = (float *)malloc(bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate verification buffer\n");
        HIP_CHECK(hipHostUnregister(A));
        HIP_CHECK(hipHostUnregister(B));
        free(A);
        free(B);
        HIP_CHECK(hipFree(C));
        return 1;
    }
    double cpu_best = 1e30;
    for (int r = 0; r < num_runs; r++) {
        double start_t = get_time_sec();
        matmul_cpu(A, B, expected, N);
        double end_t = get_time_sec();
        double t = end_t - start_t;
        if (t < cpu_best) cpu_best = t;
    }

    // 2D block/grid configuration.
    int flat_block_size, min_grid_size;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size, &flat_block_size,
                                                 matmul_kernel, 0, N * N));
    int block_side = (int)sqrt((double)flat_block_size);
    dim3 block(block_side, block_side);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        HIP_CHECK(hipEventRecord(start));
        matmul_kernel<<<grid, block>>>(A, B, C, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_stats("HIP GPU MatMul (naive)", N, num_runs, times, cpu_best);

    if (verify_result(expected, C, N)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        free(times);
        HIP_CHECK(hipHostUnregister(A));
        HIP_CHECK(hipHostUnregister(B));
        free(A);
        free(B);
        HIP_CHECK(hipFree(C));
        return 1;
    }

    free(expected);
    free(times);
    HIP_CHECK(hipHostUnregister(A));
    HIP_CHECK(hipHostUnregister(B));
    free(A);
    free(B);
    HIP_CHECK(hipFree(C));
    return 0;
}
