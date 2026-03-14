# Chapter 2 — Vector Addition

Exercises from "Programming Massively Parallel Processors". Four implementations of vector addition (`c[i] = a[i] + b[i]`) that demonstrate the progression from sequential CPU to GPU parallel execution.

## Hardware

### Machine A — Integrated GPU (shared memory)

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4

### Machine B — Discrete GPU (dedicated VRAM)

- **CPU**: AMD Ryzen 5 5600X, 6 cores / 12 threads
- **GPU**: NVIDIA GeForce RTX 3060, 12 GB GDDR6
- **Toolchain**: CUDA 12.x, NVIDIA driver 550.107.02

## Implementations

| File | Description |
|------|-------------|
| `vec_add.h` | Shared header: sequential CPU add, array generation, verification, timing, stats |
| `vec_add_cpu.c` | Sequential single-threaded CPU baseline |
| `vec_add_openmp.c` | OpenMP parallel CPU version (all cores) |
| `vec_add_hip.cpp` | AMD GPU with explicit memory copies (hipMalloc + hipMemcpy) |
| `vec_add_hip_unified.cpp` | AMD GPU with unified/managed memory (hipMallocManaged, no copies) |
| `vec_add_cuda.cu` | NVIDIA GPU with explicit memory copies (cudaMalloc + cudaMemcpy) |
| `test_vec_add.c` | Unit tests for the shared header functions |

## Building and Running

```bash
# Build everything (GPU targets auto-detected based on available compilers)
make all

# Build and run individual versions
make run-cpu
make run-openmp
make run-hip            # AMD GPU
make run-hip-unified    # AMD GPU (unified memory)
make run-cuda           # NVIDIA GPU

# Pass arguments (array size, number of runs)
make run-cpu ARGS="-n 50000000 -r 10"

# Control OpenMP thread count
OMP_NUM_THREADS=4 make run-openmp ARGS="-n 100000000 -r 5"

# Run unit tests
make test

# Run under valgrind
make valgrind-test
make valgrind-cpu
make valgrind-openmp
```

## Results (n=100,000,000 floats)

### Machine A — Integrated GPU (Radeon 860M, shared DDR5)

| Version | Best time | Throughput | Notes |
|---------|----------|------------|-------|
| CPU sequential | ~28 ms | ~40 GB/s | Single-threaded baseline |
| OpenMP (2 threads) | ~23 ms | ~49 GB/s | Peak — bandwidth saturated at 2 threads |
| OpenMP (16 threads) | ~24 ms | ~47 GB/s | No degradation, just no further gain |
| HIP GPU (explicit copy) | ~15 ms kernel | ~73 GB/s | +185 ms upload, +49 ms download |
| HIP GPU (unified memory) | ~15 ms | ~72 GB/s | No copy overhead |

#### OpenMP thread scaling on Machine A

| Threads | Best time | Throughput |
|---------|----------|------------|
| 1 | 30.5 ms | 36.7 GB/s |
| 2 | 22.9 ms | 48.7 GB/s |
| 4 | 22.7 ms | 49.3 GB/s |
| 8 | 23.3 ms | 48.1 GB/s |
| 16 | 23.7 ms | 47.2 GB/s |

DDR5 bandwidth saturates at 2 threads (~49 GB/s). Adding more threads doesn't help but doesn't hurt — the Ryzen AI 7 350 handles extra threads gracefully.

### Machine B — Discrete GPU (RTX 3060, dedicated GDDR6)

| Version | Best time | Throughput | Notes |
|---------|----------|------------|-------|
| CPU sequential | ~44 ms | ~25 GB/s | Single-threaded baseline |
| OpenMP (2 threads) | ~41 ms | ~28 GB/s | Peak — bandwidth saturated at 2 threads |
| OpenMP (12 threads) | ~52 ms | ~21 GB/s | Worse — thread overhead with no extra bandwidth |
| CUDA GPU (explicit copy) | ~3.6 ms kernel | ~313 GB/s | +53 ms upload, +174 ms download |

The RTX 3060's dedicated GDDR6 delivers ~313 GB/s — over 4x the integrated GPU's ~73 GB/s. However, data must travel over PCIe, adding ~227 ms of transfer overhead for a kernel that takes only 3.6 ms. The kernel-only speedup is dramatic, but the total wall-clock time (including transfers) is similar across both machines.

#### OpenMP thread scaling on Machine B

| Threads | Best time | Throughput |
|---------|----------|------------|
| 1 | 44.5 ms | 25.1 GB/s |
| 2 | 40.6 ms | 27.5 GB/s |
| 4 | 46.4 ms | 24.1 GB/s |
| 6 | 50.5 ms | 22.1 GB/s |
| 12 | 52.2 ms | 21.4 GB/s |

Performance peaks at 2 threads and degrades from there. The Ryzen 5600X with DDR4 dual-channel saturates around 25-27 GB/s. Adding more threads just adds OpenMP synchronization overhead without any extra bandwidth to exploit. The 12-thread average (9.9 GB/s) was also much worse than its best (21.4 GB/s), suggesting the shared cloud machine had other processes competing for resources.

#### Thread scaling comparison: DDR5 vs DDR4

Both machines saturate bandwidth at 2 threads, but they differ in how they handle additional threads:

- **DDR5 (Machine A)**: Plateaus at ~49 GB/s. Extra threads cause no degradation — throughput stays flat from 2 to 16 threads.
- **DDR4 (Machine B)**: Peaks at ~28 GB/s. Performance degrades with more threads — 12 threads is 22% slower than 2 threads.

DDR5's bandwidth ceiling (~49 GB/s) is nearly 2x DDR4's (~28 GB/s), reflecting the generational improvement in memory technology.

## Key Findings

### Vector addition is memory-bandwidth bound

Vector addition does one floating-point operation per three memory accesses (read a, read b, write c). The bottleneck is memory bandwidth, not compute. This explains why:

- **OpenMP barely helps**: 16 threads share the same memory bus. Even 2 threads are enough to nearly saturate it. Adding more threads doesn't increase available bandwidth.
- **Throughput matters more than raw time**: Reporting GB/s lets you compare against the hardware's theoretical memory bandwidth ceiling, giving context to the numbers.

### GPU is faster despite sharing memory with the CPU

The Radeon 860M is an integrated GPU — it uses the same physical DDR5 as the CPU. Yet it achieves ~73 GB/s vs the CPU's ~40 GB/s from the same RAM. The GPU's memory subsystem can issue far more outstanding memory requests simultaneously, and its thousands of threads hide memory latency through zero-cost wavefront switching. Where a CPU core stalls waiting for memory, the GPU just switches to another ready wavefront.

### Explicit copies are wasteful on integrated GPUs

The explicit copy version (`vec_add_hip.cpp`) spends ~234 ms copying data that's already in the same physical RAM, for a kernel that takes only ~15 ms. The unified memory version (`vec_add_hip_unified.cpp`) eliminates this entirely by using `hipMallocManaged`, which allocates memory accessible by both CPU and GPU through the same pointer.

On discrete GPUs with dedicated VRAM (like the RTX 3060), explicit copies over PCIe are unavoidable. The RTX 3060 results confirm this: 53 ms upload + 174 ms download for a 3.6 ms kernel. The data transfer cost is ~63x the compute cost.

### malloc vs hipMalloc vs hipMallocManaged

- **`malloc`** — CPU-only memory. The GPU cannot access it — a kernel would crash or read garbage.
- **`hipMalloc`** — GPU-only memory. The CPU cannot dereference it. Requires `hipMemcpy` to move data between host and device.
- **`hipMallocManaged`** — shared memory accessible by both CPU and GPU through the same pointer. On integrated GPUs this is essentially free (same physical RAM, just shared page tables). On discrete GPUs the runtime migrates pages over PCIe on demand, which can be slower than explicit copies if the access pattern isn't predictable.

The same distinction applies to CUDA: `cudaMalloc` vs `cudaMallocManaged`.

### Correctness verification

All parallel versions verify their results against the sequential CPU implementation. Since vector addition is element-wise (`c[i] = a[i] + b[i]`) with no reordering or reduction, identical inputs produce bit-identical outputs across all implementations. Verification uses exact float equality (`==`), which works here because there are no floating-point ordering differences. This assumption may need revisiting for operations where the GPU's floating-point rounding could differ from the CPU's.

### Timing methodology

- **CPU / OpenMP**: `clock_gettime(CLOCK_MONOTONIC)` — wall-clock time, not affected by system clock adjustments. Multiple runs with the best time reported to minimize OS scheduling noise.
- **GPU kernel**: `hipEvent`/`cudaEvent` timestamps recorded on the GPU command stream. These measure actual kernel execution, not CPU-side launch overhead. Kernel launches are asynchronous — `hipEventSynchronize`/`cudaEventSynchronize` blocks the CPU until the GPU reaches the stop marker.
- **GPU transfers**: `clock_gettime` around `hipMemcpy`/`cudaMemcpy`, which are blocking calls.

## File Structure

```
ch02/
  vec_add.h              # Shared: generate_array, vec_add_cpu, verify_result,
                         #         get_time_sec, print_stats, parse_args
  vec_add_cpu.c          # Sequential CPU version
  vec_add_openmp.c       # OpenMP parallel version
  vec_add_hip.cpp        # AMD GPU with explicit copies
  vec_add_hip_unified.cpp # AMD GPU with unified memory
  vec_add_cuda.cu        # NVIDIA GPU with explicit copies
  test_vec_add.c         # Unit tests (13 tests)
  Makefile               # Build system (auto-detects available GPU compilers)
  openmp.supp            # Valgrind suppression for OpenMP false positives
  README.md              # This file
```
