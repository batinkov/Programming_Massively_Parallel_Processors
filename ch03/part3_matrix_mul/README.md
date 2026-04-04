# Chapter 3 — Matrix Multiplication

Exercises from "Programming Massively Parallel Processors". Square matrix multiplication C = A × B using random floats.

Matrix multiplication is **compute-bound** with O(N³) arithmetic on O(N²) data. This makes it the ideal GPU workload — high arithmetic intensity with massive parallelism.

## Hardware

### Machine A — Integrated GPU (shared memory)

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152, 16 CUs)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **L3 Cache**: 16 MB
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4


## Implementations

| File | Description |
|------|-------------|
| `matrix_mul.h` | Shared header: CPU matmul, transpose, fill_random, verify, timing, stats |
| `matrix_mul_cpu.c` | Naive sequential CPU — row × column dot product |
| `matrix_mul_cpu_transposed.c` | Transpose B first, then row × row dot product (cache-friendly) |
| `matrix_mul_openmp.c` | OpenMP parallel — naive algorithm, parallelized over rows |
| `matrix_mul_openmp_transposed.c` | OpenMP parallel + transposed B — best CPU version |
| `matrix_mul_hip.cpp` | AMD GPU — naive, one thread per output element, no shared memory |

### GPU kernel design

2D thread/block structure — each thread computes one element of C:

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
float sum = 0.0f;
for (int k = 0; k < N; k++) {
    sum += A[row * N + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

Unlike on the CPU, the column access of B is less problematic on the GPU — adjacent threads in a warp access adjacent columns (`B[k*N+col]` and `B[k*N+col+1]`), which the memory controller coalesces into a single transaction.

### Verification

A is filled with random floats (seed 42), B with random floats (seed 137). All versions verify against the naive CPU result with a tolerance of 1e-2 to account for floating-point ordering differences.

## Building and Running

```bash
# Build everything (GPU targets auto-detected based on available compilers)
make all

# Build and run individual versions
make run-cpu ARGS="-n 1024 -r 3"
make run-cpu-transposed ARGS="-n 1024 -r 3"
make run-openmp ARGS="-n 1024 -r 3"
make run-openmp-transposed ARGS="-n 1024 -r 3"
make run-hip ARGS="-n 1024 -r 3"

# -n is required (matrix size), -r is optional (runs, default 3)

# Or export ARGS once
export ARGS='-n 1024 -r 3'
make run-cpu
make run-hip

# Control OpenMP thread count
OMP_NUM_THREADS=4 make run-openmp
```

## Results (Machine A)

### All versions compared (N=1024)

| Version | Best (ms) | GFLOPS | Speedup vs naive CPU |
|---------|-----------|--------|----------------------|
| CPU naive | 3,793 | 0.57 | — |
| CPU transposed | 753 | 2.85 | 5.0x |
| OpenMP 16t naive | 510 | 4.21 | 7.4x |
| OpenMP 16t transposed | 98 | 21.98 | 39.1x |
| **HIP GPU naive** | **37** | **58.22** | **109.7x** |

### GPU scaling with matrix size

| Size | GPU (ms) | GFLOPS | Speedup vs CPU |
|------|----------|--------|----------------|
| 512×512 | 7.3 | 36.7 | 34x |
| 1024×1024 | 36.9 | 58.2 | 110x |
| 2048×2048 | 325.6 | 52.8 | 165x |

### Effect of transposing B (single-threaded CPU, N=1024)

| Version | Best (ms) | GFLOPS |
|---------|-----------|--------|
| Naive (column access of B) | 3,793 | 0.57 |
| Transposed (row access of B) | 753 | 2.85 |

5x speedup from a simple O(N²) transpose before the O(N³) multiply. Sequential memory access vs strided access — the cache miss penalty dominates at this size.

### OpenMP thread scaling — naive (N=1024)

| Threads | Best (ms) | GFLOPS | Speedup |
|---------|-----------|--------|---------|
| 1 | 4,407 | 0.49 | 0.9x |
| 2 | 2,037 | 1.05 | 1.9x |
| 4 | 1,113 | 1.93 | 4.1x |
| 8 | 830 | 2.59 | 4.2x |
| 16 | 510 | 4.21 | 7.2x |

Scaling stalls at 4→8 threads (4.1x → 4.2x) due to cache contention on column-strided B access. Hyperthreading at 16 threads helps hide the latency.

### OpenMP thread scaling — transposed (N=1024)

| Threads | Best (ms) | GFLOPS | Speedup |
|---------|-----------|--------|---------|
| 1 | 751 | 2.86 | 4.9x |
| 2 | 376 | 5.71 | 9.4x |
| 4 | 196 | 10.98 | 17.8x |
| 8 | 148 | 14.53 | 30.3x |
| 16 | 98 | 21.98 | 39.1x |

Much better scaling — nearly linear up to 8 threads because the transpose eliminates cache contention. The transposed version at 16 threads (22 GFLOPS) is over 5x faster than the naive version at 16 threads (4.2 GFLOPS).

## Key Findings

### Memory access pattern dominates CPU performance

The naive CPU matmul achieves only 0.57 GFLOPS at N=1024 — less than 1% of the CPU's theoretical peak. The column-strided access of B causes constant cache misses. Transposing B gives 5x improvement (2.85 GFLOPS) simply by making both accesses sequential.

### The GPU doesn't need the transpose trick

The naive GPU kernel achieves 58 GFLOPS without transposing B. Adjacent threads in a warp access adjacent columns, so the memory controller coalesces the column reads naturally. What's a disaster for CPU caches is fine for GPU memory coalescing.

### GPU speedup grows with matrix size

The CPU gets slower with size (more cache misses), while the GPU stays relatively stable. At 512×512 the speedup is 34x; at 2048×2048 it reaches 165x. Larger matrices provide more parallelism and better amortize launch overhead.

### Combining optimizations on CPU

| Technique | Alone | Combined |
|-----------|-------|----------|
| Transpose only | 5x | — |
| OpenMP only | 7.2x | — |
| Transpose + OpenMP | — | 39x |

The techniques are multiplicative, not additive — transpose removes cache misses, OpenMP adds parallelism. Together they achieve 39x over naive CPU, approaching the GPU's 110x.

### Reporting GFLOPS, not GB/s

Matrix multiplication is compute-bound, not memory-bound. GFLOPS is the correct metric: 2N³ floating-point operations (N³ multiplies + N³ adds) per matmul. At N=1024, that's 2.1 billion FLOPs.

## File Structure

```
part3_matrix_mul/
  matrix_mul.h                    # Shared: matmul_cpu, transpose, fill_random,
                                  #         verify_result, print_stats, parse_args
  matrix_mul_cpu.c                # Naive sequential CPU
  matrix_mul_cpu_transposed.c     # Sequential CPU with B transposed
  matrix_mul_openmp.c             # OpenMP parallel (naive)
  matrix_mul_openmp_transposed.c  # OpenMP parallel + transposed B
  matrix_mul_hip.cpp              # AMD GPU — naive (no shared memory)
  Makefile                        # Build system
  README.md                       # This file
```
