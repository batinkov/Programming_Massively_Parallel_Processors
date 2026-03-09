#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "vec_add.h"
}

// Every HIP call returns an error code. This macro checks it and aborts with
// a descriptive message if something went wrong (e.g., out of GPU memory).
#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
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
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("GPU: %s (unified memory)\n", props.name);

    size_t bytes = (size_t)n * sizeof(float);

    // hipMallocManaged allocates memory accessible by both CPU and GPU through
    // the same pointer. No explicit copies needed. This is ideal for integrated
    // GPUs (like APUs) that share physical memory with the CPU.
    float *a, *b, *c;
    HIP_CHECK(hipMallocManaged(&a, bytes));
    HIP_CHECK(hipMallocManaged(&b, bytes));
    HIP_CHECK(hipMallocManaged(&c, bytes));

    // The CPU can write directly to managed memory — these pointers work on
    // both the CPU and GPU side.
    generate_array(a, n, SEED_A);
    generate_array(b, n, SEED_B);

    // Let the runtime choose the block size that maximizes occupancy for this
    // kernel on this GPU, based on register usage and shared memory requirements.
    int block_size, grid_size;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                 vec_add_kernel, 0, n));
    // Integer ceiling division: ensures enough blocks to cover all n elements.
    // Plain n/block_size truncates, leaving the remainder elements unprocessed.
    grid_size = (n + block_size - 1) / block_size;
    printf("Block size: %d, Grid size: %d\n", block_size, grid_size);

    // Create GPU event markers for accurate kernel timing.
    double *times = (double *)calloc((size_t)num_runs, sizeof(double));
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int r = 0; r < num_runs; r++) {
        HIP_CHECK(hipEventRecord(start));

        // Launch the kernel. With unified memory, we pass the same pointers
        // that the CPU used — no hipMemcpy needed.
        vec_add_kernel<<<grid_size, block_size>>>(a, b, c, n);

        HIP_CHECK(hipEventRecord(stop));
        // Block the CPU until the GPU finishes the kernel.
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times[r] = ms / 1000.0;
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    print_stats("HIP GPU (unified)", n, num_runs, times);

    // After hipEventSynchronize, the GPU is done writing to c. With unified
    // memory the CPU can read the results directly — no copy back needed.
    float *expected = (float *)malloc(bytes);
    if (!expected) {
        fprintf(stderr, "Failed to allocate memory for verification\n");
        return 1;
    }

    vec_add_cpu(a, b, expected, n);
    if (verify_result(expected, c, n)) {
        printf("Verification: PASS\n");
    } else {
        fprintf(stderr, "Verification: FAIL\n");
        free(expected);
        HIP_CHECK(hipFree(a));
        HIP_CHECK(hipFree(b));
        HIP_CHECK(hipFree(c));
        free(times);
        return 1;
    }

    // hipFree for managed memory, regular free for CPU-only allocations.
    free(expected);
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));
    free(times);
    return 0;
}
