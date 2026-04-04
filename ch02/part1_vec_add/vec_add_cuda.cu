#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "vec_add.h"
}

// Every CUDA call returns an error code. This macro checks it and aborts with
// a descriptive message if something went wrong (e.g., out of GPU memory).
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// __global__ marks this as a GPU kernel — it runs on the GPU but is launched
// from the CPU. Each GPU thread executes this function independently.
__global__ void vec_add_kernel(const float *a, const float *b, float *c, int n) {
    // Compute a unique global index for this thread:
    //   blockDim.x  = number of threads per block
    //   blockIdx.x  = which block this thread belongs to
    //   threadIdx.x = this thread's position within its block
    // This maps each thread to exactly one array element.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // The total number of threads may exceed n (due to rounding up the grid
    // size), so we must check bounds to avoid out-of-bounds memory access.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    int n, num_runs;
    parse_args(argc, argv, &n, &num_runs);

    // Query GPU properties and print the device name.
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s\n", props.name);

    size_t bytes = (size_t)n * sizeof(float);

    // Host (CPU) memory — allocated with regular malloc.
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_expected = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c || !h_expected) {
        fprintf(stderr, "Failed to allocate host memory for %d elements\n", n);
        return 1;
    }

    generate_array(h_a, n, SEED_A);
    generate_array(h_b, n, SEED_B);

    // Device (GPU) memory — allocated with cudaMalloc. These pointers are only
    // valid on the GPU and cannot be dereferenced on the CPU.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy input arrays from host (CPU) to device (GPU) memory.
    // cudaMemcpy blocks until the transfer completes, so wall-clock timing is
    // accurate here.
    double t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    double t1 = get_time_sec();
    printf("Host -> Device: %.4f ms\n", (t1 - t0) * 1000);

    // Let the runtime choose the block size that maximizes occupancy for this
    // kernel on this GPU, based on register usage and shared memory requirements.
    int block_size, grid_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                   vec_add_kernel, 0, n));
    // Integer ceiling division: ensures enough blocks to cover all n elements.
    // Plain n/block_size truncates, leaving the remainder elements unprocessed.
    grid_size = (n + block_size - 1) / block_size;
    printf("Block size: %d, Grid size: %d\n", block_size, grid_size);

    // Create GPU event markers for accurate kernel timing. Events are recorded
    // on the GPU timeline, so they measure actual kernel execution time — not
    // CPU-side launch overhead.
    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        // Record a timestamp on the GPU command stream before the kernel.
        CUDA_CHECK(cudaEventRecord(start));

        // Launch the kernel. This is asynchronous — the CPU does not wait for
        // the GPU to finish. <<<grid_size, block_size>>> specifies the number
        // of blocks and threads per block.
        vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

        // Record a timestamp after the kernel and wait for the GPU to reach it.
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Get the elapsed time between the two GPU timestamps.
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    // Clean up event objects.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    print_stats("CUDA GPU", n, num_runs, times);

    // Copy the result back from device (GPU) to host (CPU) memory.
    t0 = get_time_sec();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    t1 = get_time_sec();
    printf("Device -> Host: %.4f ms\n", (t1 - t0) * 1000);

    // Verify against CPU
    vec_add_cpu(h_a, h_b, h_expected, n);
    if (verify_result(h_expected, h_c, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        free(times);
        free(h_expected);
        free(h_a);
        free(h_b);
        free(h_c);
        return 1;
    }

    // Free GPU memory, then CPU memory.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(times);
    free(h_expected);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
