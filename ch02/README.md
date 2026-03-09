# Chapter 2 — Vector Addition

Exercises from "Programming Massively Parallel Processors". Four implementations of vector addition (`c[i] = a[i] + b[i]`) that demonstrate the progression from sequential CPU to GPU parallel execution.

## Hardware

All benchmarks were run on:

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4

## Implementations

| File | Description |
|------|-------------|
| `vec_add.h` | Shared header: sequential CPU add, array generation, verification, timing, stats |
| `vec_add_cpu.c` | Sequential single-threaded CPU baseline |
| `vec_add_openmp.c` | OpenMP parallel CPU version (all cores) |
| `vec_add_hip.cpp` | GPU with explicit memory copies (hipMalloc + hipMemcpy) |
| `vec_add_hip_unified.cpp` | GPU with unified/managed memory (hipMallocManaged, no copies) |
| `test_vec_add.c` | Unit tests for the shared header functions |

## Building and Running

```bash
# Build everything
make all

# Build and run individual versions
make run-cpu
make run-openmp
make run-hip
make run-hip-unified

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

| Version | Best time | Throughput | Notes |
|---------|----------|------------|-------|
| CPU sequential | ~28 ms | ~40 GB/s | Single-threaded baseline |
| OpenMP (16 threads) | ~25 ms | ~44 GB/s | Modest speedup over sequential |
| HIP GPU (explicit copy) | ~15 ms kernel | ~73 GB/s | +185 ms upload, +49 ms download |
| HIP GPU (unified memory) | ~15 ms | ~72 GB/s | No copy overhead |

## Key Findings

### Vector addition is memory-bandwidth bound

Vector addition does one floating-point operation per three memory accesses (read a, read b, write c). The bottleneck is memory bandwidth, not compute. This explains why:

- **OpenMP barely helps**: 16 threads share the same memory bus. Even 2 threads are enough to nearly saturate it. Adding more threads doesn't increase available bandwidth.
- **Throughput matters more than raw time**: Reporting GB/s lets you compare against the hardware's theoretical memory bandwidth ceiling, giving context to the numbers.

### GPU is faster despite sharing memory with the CPU

The Radeon 860M is an integrated GPU — it uses the same physical DDR5 as the CPU. Yet it achieves ~73 GB/s vs the CPU's ~40 GB/s from the same RAM. The GPU's memory subsystem can issue far more outstanding memory requests simultaneously, and its thousands of threads hide memory latency through zero-cost wavefront switching. Where a CPU core stalls waiting for memory, the GPU just switches to another ready wavefront.

### Explicit copies are wasteful on integrated GPUs

The explicit copy version (`vec_add_hip.cpp`) spends ~234 ms copying data that's already in the same physical RAM, for a kernel that takes only ~15 ms. The unified memory version (`vec_add_hip_unified.cpp`) eliminates this entirely by using `hipMallocManaged`, which allocates memory accessible by both CPU and GPU through the same pointer.

On discrete GPUs with dedicated VRAM, explicit copies are unavoidable and the tradeoff is different.

### malloc vs hipMallocManaged

Regular `malloc` allocates memory that only the CPU's page tables know about. The GPU cannot access it — a kernel would crash or read garbage. `hipMallocManaged` sets up page table mappings for both the CPU and GPU, allowing both to access the same physical pages through the same pointer. Even though integrated GPUs share the same physical RAM, the CPU and GPU have separate virtual address spaces that must be explicitly bridged.

### Correctness verification

All parallel versions verify their results against the sequential CPU implementation. Since vector addition is element-wise (`c[i] = a[i] + b[i]`) with no reordering or reduction, identical inputs produce bit-identical outputs across all implementations. Verification uses exact float equality (`==`), which works here because there are no floating-point ordering differences. This assumption may need revisiting for operations where the GPU's floating-point rounding could differ from the CPU's.

### Timing methodology

- **CPU / OpenMP**: `clock_gettime(CLOCK_MONOTONIC)` — wall-clock time, not affected by system clock adjustments. Multiple runs with the best time reported to minimize OS scheduling noise.
- **GPU kernel**: `hipEvent` timestamps recorded on the GPU command stream. These measure actual kernel execution, not CPU-side launch overhead. Kernel launches are asynchronous — `hipEventSynchronize` blocks the CPU until the GPU reaches the stop marker.
- **GPU transfers**: `clock_gettime` around `hipMemcpy`, which is a blocking call.

## File Structure

```
ch02/
  vec_add.h              # Shared: generate_array, vec_add_cpu, verify_result,
                         #         get_time_sec, print_stats, parse_args
  vec_add_cpu.c          # Sequential CPU version
  vec_add_openmp.c       # OpenMP parallel version
  vec_add_hip.cpp        # GPU with explicit copies
  vec_add_hip_unified.cpp # GPU with unified memory
  test_vec_add.c         # Unit tests (13 tests)
  Makefile               # Build system
  README.md              # This file
```
