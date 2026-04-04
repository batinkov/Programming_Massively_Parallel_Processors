# Chapter 3 — Box Blur

Exercises from "Programming Massively Parallel Processors". RGB box blur (average filter) — each output pixel is the average of its `(2*blur_size+1)²` neighbors.

Unlike grayscale conversion (part1), blur is **compute-bound**: each output pixel reads many input pixels, giving the GPU's parallel ALUs real work to do.

## Hardware

### Machine A — Integrated GPU (shared memory)

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152, 16 CUs)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **L3 Cache**: 16 MB
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4

Machine B (RTX 3060) results pending — CUDA version not yet benchmarked.

## Implementations

| File | Description |
|------|-------------|
| `blur.h` | Shared header: CPU blur, verification, timing, stats, output |
| `blur_cpu.c` | Sequential single-threaded CPU baseline |
| `blur_openmp.c` | OpenMP parallel CPU version (parallelized over rows) |
| `blur_hip.cpp` | AMD GPU — naive version, one thread per pixel, no shared memory |

### GPU kernel design

Same 2D thread/block structure as part1 (grayscale):

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Each thread computes one output pixel by looping over the `(2*blur_size+1)²` neighborhood. This is the naive version — neighboring threads redundantly read overlapping input pixels from global memory. A shared memory version would load the tile once cooperatively.

### Memory strategy

- **HIP (integrated GPU)**: `hipHostRegister` for input, `hipMallocManaged` for output — same as part1.

## Building and Running

```bash
# Build everything (GPU targets auto-detected based on available compilers)
make all

# Build and run individual versions
make run-cpu ARGS="<PATH_TO_IMAGE> -r 10 -b 3"
make run-openmp ARGS="<PATH_TO_IMAGE> -r 10 -b 3"
make run-hip ARGS="<PATH_TO_IMAGE> -r 10 -b 3"

# -b sets blur_size (default 1). Patch is (2*b+1)x(2*b+1):
#   -b 1  ->  3x3   (9 pixels)
#   -b 3  ->  7x7   (49 pixels)
#   -b 15 -> 31x31  (961 pixels)

# Or export ARGS once to avoid repeating it
export ARGS='<PATH_TO_IMAGE> -r 10 -b 3'
make run-cpu
make run-openmp
make run-hip

# Control OpenMP thread count
OMP_NUM_THREADS=4 make run-openmp
```

## Results (5600×3200, 17.9M pixels, Machine A)

### Performance scaling with patch size

| Patch | CPU (ms) | OpenMP 16t (ms) | OMP speedup | GPU (ms) | GPU speedup |
|-------|----------|-----------------|-------------|----------|-------------|
| 3×3   | ~106     | ~21.3           | 5.0x        | ~9.2     | 11.6x       |
| 5×5   | ~203     | ~44.6           | 4.6x        | ~12.6    | 17.6x       |
| 7×7   | ~336     | ~65.1           | 5.2x        | ~18.5    | 21.7x       |
| 11×11 | ~736     | ~124            | 5.9x        | ~34.8    | 27.0x       |
| 15×15 | ~1,371   | ~222            | 6.2x        | ~68.9    | 26.9x       |
| 21×21 | ~2,464   | ~392            | 6.3x        | ~144     | 24.0x       |
| 31×31 | ~4,960   | ~803            | 6.2x        | ~267     | 27.6x       |
| 41×41 | ~8,238   | ~1,361          | 6.1x        | ~448     | 28.5x       |
| 63×63 | ~18,800  | ~3,313          | 5.7x        | ~1,053   | 28.4x       |

### OpenMP thread scaling (7×7 patch)

| Threads | Best (ms) | Speedup |
|---------|-----------|---------|
| 1       | 391.8     | 1.0x    |
| 2       | 197.2     | 2.0x    |
| 4       | 99.3      | 4.0x    |
| 8       | 67.3      | 5.9x    |
| 16      | 60.5      | 6.6x    |

### OpenMP thread scaling (31×31 patch)

| Threads | Best (ms) | Speedup |
|---------|-----------|---------|
| 1       | 5,710     | 1.0x    |
| 2       | 2,872     | 2.0x    |
| 4       | 1,446     | 4.0x    |
| 8       | 926       | 6.3x    |
| 16      | 797       | 7.4x    |

Larger patches scale better with threads — 7.4x at 31×31 vs 6.6x at 7×7. The workload becomes more compute-bound, so additional threads have real work to do rather than waiting on memory.

## Key Findings

### Blur is compute-bound — GPU dominates

Unlike grayscale (part1) where the GPU lost to OpenMP on the integrated GPU, blur shows **24-28x speedup** over single-threaded CPU. The GPU beats 16-thread OpenMP by 3-5x across all patch sizes. This is because blur has high arithmetic intensity — each output pixel reads `(2*blur_size+1)²` input pixels, giving the GPU's thousands of ALUs real work to do.

### GPU speedup grows with patch size

At 3×3 (9 reads per pixel), the GPU achieves 11.6x speedup. At 63×63 (3969 reads per pixel), it reaches 28.4x. More computation per pixel means a higher compute-to-memory ratio, which better exploits the GPU's parallel architecture.

### OpenMP scaling is limited by core count

OpenMP plateaus at ~5-7x speedup regardless of patch size — constrained by the 8 physical cores. The GPU has no such limit: its 16 CUs with 64 stream processors each provide far more parallel compute capacity.

### Neighboring threads read overlapping data

This is the naive version — each thread independently reads its full patch from global memory. For a 7×7 blur, adjacent threads share 42 of their 49 input pixels. This redundant reading is the main optimization opportunity. A shared memory version would load each tile once cooperatively and serve the overlapping reads from fast on-chip memory (~1-2 cycles vs ~hundreds for global memory).

### Contrast with grayscale (part1)

| Property | Grayscale | Blur (7×7) |
|----------|-----------|------------|
| Reads per output pixel | 3 bytes | 147 bytes (49 × 3) |
| Bottleneck | Memory bandwidth | Compute |
| GPU vs CPU speedup | 2.9x | 21.7x |
| GPU vs OpenMP 16t | 0.7x (GPU loses) | 3.6x (GPU wins) |

The same hardware, same image, same integrated GPU — but the workload characteristics completely change the outcome.

### Timing methodology

- **CPU / OpenMP**: `clock_gettime(CLOCK_MONOTONIC)` — wall-clock time. Multiple runs, best time reported.
- **GPU kernel**: `hipEvent` timestamps on the GPU command stream. Measures actual kernel execution, not CPU-side launch overhead.
- **Only compute is timed**: Image loading and JPEG writing are excluded from all benchmarks.

## File Structure

```
part2_blur/
  blur.h              # Shared: blur_cpu, verify_result, get_time_sec,
                      #         print_stats, parse_args, write_blurred_jpg
  blur_cpu.c          # Sequential CPU version
  blur_openmp.c       # OpenMP parallel version
  blur_hip.cpp        # AMD GPU — naive (no shared memory)
  Makefile            # Build system (auto-detects available GPU compilers)
  README.md           # This file
```
