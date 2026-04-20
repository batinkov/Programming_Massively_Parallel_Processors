// Tiled GPU matrix multiplication using shared memory.
// Each block computes one TILE_WIDTH x TILE_WIDTH tile of the output.
// Threads in a block cooperatively load tiles of A and B into shared memory,
// then reuse them for the partial dot products — reducing global memory
// traffic by a factor of TILE_WIDTH compared to the naive kernel.

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

#define TILE_WIDTH 32

__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                    int N) {
    // Shared memory tiles of A and B — one copy per block.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Output element this thread is responsible for.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Accumulator — lives in a register, private to this thread.
    float sum = 0.0f;

    // Number of phases along the K (shared) dimension.
    int num_phases = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ph++) {
        // Cooperative load: each thread loads one element of A into Mds.
        // Pad with 0.0f for out-of-bounds positions — won't affect the sum.
        int a_col = ph * TILE_WIDTH + tx;
        if (row < N && a_col < N)
            Mds[ty][tx] = A[row * N + a_col];
        else
            Mds[ty][tx] = 0.0f;

        // Cooperative load: each thread loads one element of B into Nds.
        int b_row = ph * TILE_WIDTH + ty;
        if (b_row < N && col < N)
            Nds[ty][tx] = B[b_row * N + col];
        else
            Nds[ty][tx] = 0.0f;

        // Wait for all threads to finish loading before anyone reads.
        __syncthreads();

        // Compute the partial dot product for this phase from shared memory.
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += Mds[ty][k] * Nds[k][tx];
        }

        // Wait for all threads to finish reading before we overwrite the tiles
        // in the next iteration.
        __syncthreads();
    }

    // Write the final result, guarded by bounds check for non-multiple N.
    if (row < N && col < N)
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
    printf("Tile size: %dx%d\n", TILE_WIDTH, TILE_WIDTH);

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

    // Register host memory so the GPU can read it directly (integrated GPU).
    HIP_CHECK(hipHostRegister(A, bytes, hipHostRegisterDefault));
    HIP_CHECK(hipHostRegister(B, bytes, hipHostRegisterDefault));

    // Output in managed memory — accessible by both GPU (kernel writes) and
    // CPU (verification reads).
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

    // Launch configuration: each block computes a TILE_WIDTH x TILE_WIDTH
    // tile of the output. Grid is sized to cover all output tiles.
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (N + TILE_WIDTH - 1) / TILE_WIDTH);
    printf("Block: %dx%d (%d threads), Grid: %dx%d (%d blocks)\n",
           block.x, block.y, block.x * block.y,
           grid.x, grid.y, grid.x * grid.y);

    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        HIP_CHECK(hipEventRecord(start));
        matmul_tiled_kernel<<<grid, block>>>(A, B, C, N);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_stats("HIP GPU MatMul (tiled)", N, num_runs, times, cpu_best);

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
