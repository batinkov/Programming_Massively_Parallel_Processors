# Chapter 2 — Notes

## Data parallelism

Vector addition is the canonical example of **data parallelism**: each output element `c[i] = a[i] + b[i]` depends only on the corresponding input elements. No thread needs to communicate with any other thread. This is why the kernels in this chapter use no shared memory, no synchronization, and no reduction. Every thread is completely independent.

This is the simplest class of parallel problem. Later chapters introduce patterns where threads must cooperate (reductions, stencils, matrix multiplication), which require shared memory and synchronization barriers.

## Heterogeneous computing model

A CUDA/HIP program is a **heterogeneous** program — it runs on two processors simultaneously:

- **Host (CPU)**: Orchestrates execution — allocates memory, transfers data, launches kernels, verifies results.
- **Device (GPU)**: Executes the parallel computation.

The CPU is the control plane; the GPU is the data plane. In the ch02 code this is visible in `main()`: all the setup, timing, verification, and cleanup is CPU code. The only thing the GPU does is execute `vec_add_kernel`.

The host and device have **separate memory spaces** (even on integrated GPUs at the programming model level). Data must be explicitly moved between them unless unified memory is used.

## Function type qualifiers

CUDA/HIP provide three qualifiers that control where a function runs and where it can be called from:

| Qualifier | Runs on | Called from | Notes |
|-----------|---------|-------------|-------|
| `__global__` | GPU | CPU (or GPU in dynamic parallelism) | Kernel — the entry point for GPU execution. Returns `void`. |
| `__device__` | GPU | GPU | Helper function called from kernels or other device functions. |
| `__host__` | CPU | CPU | Regular CPU function (this is the default — the qualifier is optional). |

`__host__` and `__device__` can be combined: `__host__ __device__` makes the compiler generate both CPU and GPU versions of the same function. Useful for utility functions needed on both sides.

The call direction is strictly **CPU → GPU**, never the reverse. The GPU cannot call host functions. Dynamic parallelism lets a kernel launch other kernels, but that's GPU → GPU — still no callback to the CPU.

In ch02, only `__global__` is used. The shared header (`vec_add.h`) contains CPU-only functions — `generate_array`, `verify_result`, etc. — which are implicitly `__host__`.

## Kernel launch syntax

```c
vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
```

The `<<<grid_size, block_size>>>` syntax is not a function call — it's a **launch configuration** that specifies how many threads to create:

- `grid_size` — number of thread blocks
- `block_size` — number of threads per block
- Total threads = `grid_size × block_size`

The launch is **asynchronous**: the CPU queues the kernel on the GPU's command stream and continues immediately. The GPU executes it independently. To wait for completion, you must explicitly synchronize (`cudaEventSynchronize`, `hipEventSynchronize`, or `cudaDeviceSynchronize`).

### Multi-process GPU sharing

The GPU is a shared resource. When multiple processes launch kernels, the GPU **time-slices compute** (SMs, warps) between them. However, **memory is not swapped** — `cudaMalloc` allocations persist in VRAM for their entire lifetime, not just during kernel execution. If two processes' allocations together exceed VRAM, the second `cudaMalloc` fails with out-of-memory. It's the overlap of allocation lifetimes that matters, not whether kernels run simultaneously. Running the same programs sequentially works fine because one frees before the other allocates. This is why GPU job schedulers (SLURM, Kubernetes) typically assign whole GPUs to processes. NVIDIA MIG solves this on data center GPUs by hard-partitioning SMs and memory.

The `<<<>>>` syntax is CUDA/HIP-specific — the compiler (`nvcc`/`hipcc`) transforms it into runtime API calls before passing to the underlying C++ compiler.

## Memory model and allocation lifecycle

### The explicit copy pattern

This is the standard GPU programming workflow, used in `vec_add_cuda.cu` and `vec_add_hip.cpp`:

```
1. malloc()          — allocate host memory
2. cudaMalloc()      — allocate device memory
3. cudaMemcpy(H→D)   — copy inputs to GPU
4. kernel<<<>>>()    — execute on GPU
5. cudaMemcpy(D→H)   — copy results back
6. cudaFree()        — free device memory
7. free()            — free host memory
```

Host pointers and device pointers are **not interchangeable**. Dereferencing a device pointer on the CPU (or vice versa) is undefined behavior — it will crash or silently corrupt memory.

### Three allocation strategies

| Function | Accessible by | Transfer needed | Best for |
|----------|--------------|-----------------|----------|
| `malloc` | CPU only | N/A | CPU-only data |
| `cudaMalloc` / `hipMalloc` | GPU only | Yes — explicit `cudaMemcpy` / `hipMemcpy` | Discrete GPUs, maximum control |
| `cudaMallocManaged` / `hipMallocManaged` | Both CPU and GPU | No — runtime handles it | Integrated GPUs, prototyping |

On **integrated GPUs** (like the Radeon 860M), managed memory avoids pointless copies — the data is already in the same physical RAM. On **discrete GPUs** (like the RTX 3060), managed memory migrates pages over PCIe on demand, which can be slower than explicit copies for unpredictable access patterns.

### cudaMemcpy direction constants

The fourth argument to `cudaMemcpy` / `hipMemcpy` specifies the transfer direction:

- `cudaMemcpyHostToDevice` — CPU → GPU
- `cudaMemcpyDeviceToHost` — GPU → CPU
- `cudaMemcpyDeviceToDevice` — GPU → GPU (for copies within device memory)

These calls are **synchronous** by default — the CPU blocks until the transfer completes.

## The ceiling division pattern

```c
grid_size = (n + block_size - 1) / block_size;
```

This is integer ceiling division. If `n = 1000` and `block_size = 256`, plain `n / block_size = 3` (truncated), which only covers 768 elements. Ceiling division gives 4 blocks (1024 threads), covering all elements.

This means the last block has **excess threads** (1024 - 1000 = 24 threads that have no work). The bounds check in the kernel handles this:

```c
if (i < n) {
    c[i] = a[i] + b[i];
}
```

Without this check, excess threads would read/write out-of-bounds memory. This ceiling-division + bounds-check pair appears in virtually every GPU kernel.

## Error handling

Every CUDA/HIP API call returns an error code. The ch02 code wraps this in a macro:

```c
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)
```

Common failure modes caught by this: out of GPU memory, invalid device pointer, kernel launch with invalid configuration, driver not loaded.

Kernel launches don't return errors directly (they're async). Errors surface on the next synchronization point — `cudaEventSynchronize`, `cudaDeviceSynchronize`, or even `cudaMemcpy`.

## CUDA vs HIP API mapping

The HIP and CUDA implementations in ch02 are nearly line-for-line identical. HIP was designed as a thin portability layer over CUDA. The mapping is mechanical:

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaFree` | `hipFree` |
| `cudaMallocManaged` | `hipMallocManaged` |
| `cudaEventCreate` | `hipEventCreate` |
| `cudaEventRecord` | `hipEventRecord` |
| `cudaEventSynchronize` | `hipEventSynchronize` |
| `cudaEventElapsedTime` | `hipEventElapsedTime` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaOccupancyMaxPotentialBlockSize` | `hipOccupancyMaxPotentialBlockSize` |
| `cudaError_t` | `hipError_t` |
| `cudaSuccess` | `hipSuccess` |
| `nvcc` | `hipcc` |
| `cuda_runtime.h` | `hip/hip_runtime.h` |

The kernel code itself (`__global__`, `threadIdx`, `blockIdx`, `blockDim`, `<<<>>>`) is identical between CUDA and HIP. AMD's `hipify` tool can automate the conversion for larger codebases.

## Occupancy-based launch configuration

Rather than hardcoding block size (a common pattern in tutorials), the ch02 code uses the runtime to choose it:

```c
cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, vec_add_kernel, 0, n);
```

This queries the GPU's hardware limits (registers per SM, max threads per block, shared memory) and the kernel's resource usage to find the block size that maximizes **occupancy** — the ratio of active warps to the maximum the hardware supports. Higher occupancy generally means better latency hiding for memory-bound kernels like vector addition.
