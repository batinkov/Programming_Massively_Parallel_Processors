# Chapter 5 — Tiled Matrix Multiplication

Exercises from "Programming Massively Parallel Processors". Square matrix multiplication C = A × B using the tiling technique introduced in chapter 5. Builds directly on the naive GPU matmul from ch03/part3 by adding shared memory tiling.

The goal is to reduce global memory traffic by loading tiles of A and B into on-chip shared memory once per block, then reusing them for many partial products. This transforms the kernel from memory-bound to compute-bound.

## Hardware

### Machine A — Integrated GPU (shared memory)

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152, 16 CUs)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **L3 Cache**: 16 MB
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4

### Machine B — Discrete GPU (dedicated VRAM, cloud)

- **CPU**: AMD Ryzen 5 5600X, 6 cores / 12 threads
- **GPU**: NVIDIA GeForce RTX 3060, 12 GB GDDR6
- **L3 Cache**: 32 MB
- **Toolchain**: CUDA 12.1, NVIDIA driver 535.154.05

## Implementations

| File | Description |
|------|-------------|
| `matrix_mul.h` | Shared header: CPU matmul reference, transpose, fill_random, verify, timing, stats (copied from ch03) |
| `matrix_mul_hip_tiled.cpp` | AMD GPU — tiled kernel with shared memory (TILE_WIDTH = 32) |
| `matrix_mul_cuda_tiled.cu` | NVIDIA GPU — tiled kernel with shared memory and explicit copies |

### GPU kernel design

The kernel uses a 2D block of `TILE_WIDTH × TILE_WIDTH` threads. Each block computes one tile of the output matrix C by iterating through phases along the K dimension:

```c
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

float sum = 0.0f;
for (int ph = 0; ph < num_phases; ph++) {
    // 1. Cooperative load: each thread loads one element of A and B
    Mds[ty][tx] = A[row * N + ph * TILE_WIDTH + tx];  // with boundary check
    Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col]; // with boundary check
    __syncthreads();

    // 2. Partial dot product from shared memory
    for (int k = 0; k < TILE_WIDTH; k++)
        sum += Mds[ty][k] * Nds[k][tx];
    __syncthreads();
}
C[row * N + col] = sum;  // with boundary check
```

Key details:
- **Accumulator in a register**: `sum` is a private automatic variable, fastest access.
- **Tiles in shared memory**: Mds and Nds are `__shared__`, visible to all threads in the block.
- **Zero padding for boundaries**: out-of-bounds loads write 0.0f so the inner loop runs unconditionally.
- **Two barriers per phase**: one after loading (before anyone reads), one after reading (before the next phase overwrites the tiles).
- **TILE_WIDTH = 32**: matches the warp size and gives 1024 threads per block (block dimension cap).

### Memory strategy

Same patterns as the ch03 exercises:
- **HIP (integrated GPU)**: `hipHostRegister` for inputs, `hipMallocManaged` for output — no copies needed since CPU and GPU share DDR5
- **CUDA (discrete GPU)**: `cudaMalloc` + `cudaMemcpy` for both inputs and output — required because data must travel over PCIe between CPU RAM and GPU VRAM

## Building and Running

```bash
make all

# Run with different matrix sizes
make run-hip-tiled ARGS="-n 1024 -r 3"
make run-cuda-tiled ARGS="-n 1024 -r 3"

# Or export ARGS once
export ARGS='-n 1024 -r 3'
make run-hip-tiled
make run-cuda-tiled
```

## Results (Machine A)

### Tiled vs naive HIP GPU (TILE_WIDTH = 32)

| Size | Naive (GFLOPS) | Tiled (GFLOPS) | Speedup from tiling |
|------|---------------|----------------|---------------------|
| 512×512   | 36.7  | 95.6  | 2.6x |
| 1024×1024 | 58.2  | 115.2 | 2.0x |
| 2048×2048 | 52.8  | 169.2 | 3.2x |

Naive numbers are taken from the ch03/part3_matrix_mul README (same Machine A, earlier runs).

### Tiled scaling

| Size | Best (ms) | GFLOPS | Speedup vs CPU |
|------|-----------|--------|----------------|
| 512×512   | 2.8    | 95.6  | 118x |
| 1024×1024 | 18.6   | 115.2 | 320x |
| 2048×2048 | 101.5  | 169.2 | 411x |

The tiled kernel scales up with matrix size — from 95 GFLOPS at N=512 to 169 GFLOPS at N=2048. More work per kernel better amortizes launch overhead and fills the GPU.

## Results (Machine B)

### Tiled CUDA scaling

| Size | Best (ms) | GFLOPS | Speedup vs CPU | Copy overhead (H→D + D→H) |
|------|-----------|--------|----------------|---------------------------|
| 512×512   | 0.30   | 883 | 538x   | 0.5 + 0.6 ms   |
| 1024×1024 | 2.37   | 906 | 1,296x | 2.4 + 1.5 ms   |
| 2048×2048 | 18.72  | 918 | 2,069x | 11.2 + 5.3 ms  |
| 4096×4096 | 139.80 | 983 | 2,367x | 46.7 + 20.9 ms |

The tiled CUDA kernel reaches **~980 GFLOPS** on the RTX 3060 at N=4096 — about 7.7% of the card's ~12.7 TFLOPS FP32 peak. Performance keeps climbing with N as larger grids amortize launch and synchronization overhead. Closing this gap would require the optimizations covered in later chapters and production libraries: register-level tiling (each thread computes a 4×4 or 8×8 output block), double-buffered loads, vectorized memory access, and tensor cores for mixed precision.

The speedup vs CPU is much higher here (up to 2,069x) than on Machine A (411x) mainly because Machine B's naive CPU matmul suffers badly from cache thrashing — not because the GPU is fundamentally faster than on Machine A. What's more telling is the absolute GFLOPS: ~920 on the RTX 3060 vs ~170 on the integrated Radeon 860M, a ~5.4x advantage from dedicated GDDR6 and more compute units.

A direct tiled-vs-naive comparison for CUDA isn't included here because the earlier ch03 naive CUDA numbers were measured on a different cloud machine (RTX A2000). To do a proper comparison we'd need to re-run the naive CUDA kernel on this exact machine (RTX 3060).

## Key Findings

### Tiling transforms the kernel from memory-bound to compute-bound

The naive version plateaus and even regresses at large N (58 GFLOPS at N=1024, dropping to 53 at N=2048). This is the bandwidth wall — more threads compete for the same DDR5 bandwidth, so extra parallelism doesn't help.

The tiled version scales up with N (95 → 115 → 169 GFLOPS). With global memory traffic reduced by ~32x (one load per tile instead of per thread), memory is no longer the bottleneck. More work means more opportunity for the compute units to stay busy.

### The speedup from tiling grows with matrix size

- N=512: 2.6x (tiled/naive)
- N=1024: 2.0x
- N=2048: 3.2x

The ratio is non-monotonic because of two effects pulling in opposite directions at smaller sizes:
- At N=512, the naive version benefits from L1/L2 caching since the matrices partially fit in cache
- At N=2048, the matrices exceed cache capacity, so the naive version loses its cache advantage while the tiled version is unaffected (it manages its own cache via shared memory)

### Why the speedup is less than 32x

The book's math says TILE_WIDTH=32 should reduce global memory traffic by 32x. In practice the speedup is much less (2-3x) for three reasons:

1. **The naive kernel was not fully memory-bound** — GPU L1/L2 caches were already catching some of the redundant reads that tiling eliminates.
2. **Tiling has overhead** — shared memory loads, `__syncthreads()` barriers, and register pressure that the naive kernel doesn't pay.
3. **Amdahl's law** — even with perfect memory reuse, the kernel still has to do the compute. The tiled version hits the GPU's compute ceiling, not the memory ceiling, so further reducing memory traffic wouldn't help.

### Integrated GPU sharing DDR5 still benefits significantly

Even though the Radeon 860M shares the same DDR5 as the CPU, tiling brings a clear win. 169 GFLOPS on an integrated GPU is a strong result — about 3x the best OpenMP transposed CPU version on the same machine (22 GFLOPS at N=1024).

### Timing methodology

- **CPU**: `clock_gettime(CLOCK_MONOTONIC)` — wall-clock time, multiple runs, best reported
- **GPU kernel**: `hipEvent` timestamps on the GPU command stream
- **Only compute is timed**: host → device registration and verification are excluded

## File Structure

```
part1_matrix_mul/
  matrix_mul.h                  # Shared header (copied from ch03)
  matrix_mul_hip_tiled.cpp      # AMD GPU — tiled kernel with shared memory
  matrix_mul_cuda_tiled.cu      # NVIDIA GPU — tiled kernel with explicit copies
  Makefile                      # Build system
  README.md                     # This file
```
