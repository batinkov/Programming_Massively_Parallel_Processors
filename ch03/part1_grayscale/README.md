 # Chapter 3 — Grayscale Conversion

Exercises from "Programming Massively Parallel Processors". RGB to grayscale conversion using the ITU-R BT.601 luminance formula:

```
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

## Hardware

### Machine A — Integrated GPU (shared memory)

- **CPU**: AMD Ryzen AI 7 350, 8 cores / 16 threads
- **GPU**: AMD Radeon 860M (integrated, RDNA3, gfx1152, 16 CUs)
- **Memory**: 94 GB DDR5, 128-bit bus (dual channel), shared between CPU and GPU
- **L3 Cache**: 16 MB
- **Toolchain**: GCC 15.2 (C23), ROCm 6.4.2 / HIP 6.4

### Machine B — Discrete GPU (dedicated VRAM, cloud)

- **CPU**: Intel Xeon E5-2697A v4 @ 2.60 GHz, 16 cores / 32 threads
- **GPU**: NVIDIA GeForce RTX 3060, 12 GB GDDR6
- **L3 Cache**: 40 MB
- **Toolchain**: CUDA 13.0, NVIDIA driver 580.126.09

## Implementations

| File | Description |
|------|-------------|
| `grayscale.h` | Shared header: CPU grayscale, verification, timing, stats, output |
| `grayscale_cpu.c` | Sequential single-threaded CPU baseline |
| `grayscale_openmp.c` | OpenMP parallel CPU version (all cores) |
| `grayscale_hip.cpp` | AMD GPU with `hipHostRegister` (no copies on integrated GPU) |
| `grayscale_cuda.cu` | NVIDIA GPU with explicit copies (`cudaMalloc` + `cudaMemcpy`) |

### GPU kernel design

The kernel uses a **2D thread/block structure** matching the image's spatial layout — this is the main focus of PMPP chapter 3:

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Block size is determined by the runtime via `hipOccupancyMaxPotentialBlockSize` / `cudaOccupancyMaxPotentialBlockSize`, then split into a square 2D block (e.g., 1024 → 32×32).

### Memory strategy

- **HIP (integrated GPU)**: `hipHostRegister` for input (avoids copying data that's already in shared RAM), `hipMallocManaged` for output (both GPU and CPU can access directly).
- **CUDA (discrete GPU)**: `cudaMalloc` + `cudaMemcpy` for both input and output (data must travel over PCIe between CPU RAM and GPU VRAM).

## Building and Running

```bash
# Build everything (GPU targets auto-detected based on available compilers)
make all

# Build and run individual versions
make run-cpu ARGS="<PATH_TO_IMAGE> -r 20"
make run-openmp ARGS="<PATH_TO_IMAGE> -r 20"
make run-hip ARGS="<PATH_TO_IMAGE> -r 20"
make run-cuda ARGS="<PATH_TO_IMAGE> -r 20"

# Control OpenMP thread count
OMP_NUM_THREADS=4 make run-openmp ARGS="<PATH_TO_IMAGE> -r 10"

# Or export ARGS once to avoid repeating it
export ARGS='<PATH_TO_IMAGE> -r 20'
make run-cpu
make run-openmp
make run-hip
```

## Results (5600×3200, 17.9M pixels)

### Machine A — Integrated GPU (Radeon 860M, shared DDR5)

| Version | Best time | Throughput | Speedup |
|---------|----------|------------|---------|
| CPU sequential | ~12.0 ms | ~5.6 GB/s | — |
| OpenMP (1 thread) | ~14.9 ms | ~4.5 GB/s | 1.0x |
| OpenMP (2 threads) | ~7.4 ms | ~9.0 GB/s | 2.0x |
| OpenMP (4 threads) | ~3.7 ms | ~18.0 GB/s | 4.0x |
| OpenMP (8 threads) | ~3.7 ms | ~18.0 GB/s | 4.0x |
| OpenMP (16 threads) | ~3.1 ms | ~21.3 GB/s | 4.9x |
| HIP GPU | ~4.3 ms | ~15.6 GB/s | 2.9x |

#### OpenMP thread scaling on Machine A

| Threads | Best time | Throughput |
|---------|----------|------------|
| 1 | 14.9 ms | 4.5 GB/s |
| 2 | 7.4 ms | 9.0 GB/s |
| 4 | 3.7 ms | 18.0 GB/s |
| 8 | 3.7 ms | 18.0 GB/s |
| 16 | 3.1 ms | 21.3 GB/s |

Bandwidth saturates around 4 threads (~18 GB/s). Going from 4 to 16 threads provides only modest improvement.

### Machine B — Discrete GPU (RTX 3060, dedicated GDDR6)

| Version | Best time | Throughput | Speedup |
|---------|----------|------------|---------|
| CPU sequential | ~30.1 ms | ~2.2 GB/s | — |
| OpenMP (1 thread) | ~30.1 ms | ~2.2 GB/s | 1.0x |
| OpenMP (2 threads) | ~16.6 ms | ~4.0 GB/s | 1.9x |
| OpenMP (4 threads) | ~9.0 ms | ~7.4 GB/s | 3.3x |
| OpenMP (8 threads) | ~6.2 ms | ~10.8 GB/s | 4.9x |
| OpenMP (16 threads) | ~5.7 ms | ~11.8 GB/s | 5.3x |
| OpenMP (32 threads) | ~6.0 ms | ~11.2 GB/s | 5.4x |
| CUDA GPU (kernel only) | ~0.34 ms | ~196 GB/s | 88.1x |

The RTX 3060's dedicated GDDR6 delivers ~196 GB/s — over 12x the integrated GPU's ~15.6 GB/s. However, data must travel over PCIe, adding ~24 ms of transfer overhead (13.6 ms upload + 10.4 ms download) for a kernel that takes only 0.34 ms.

#### OpenMP thread scaling on Machine B

| Threads | Best time | Throughput |
|---------|----------|------------|
| 1 | 30.1 ms | 2.2 GB/s |
| 2 | 16.6 ms | 4.0 GB/s |
| 4 | 9.0 ms | 7.4 GB/s |
| 8 | 6.2 ms | 10.8 GB/s |
| 16 | 5.7 ms | 11.8 GB/s |
| 32 | 6.0 ms | 11.2 GB/s |

Machine B's memory bandwidth peaks at ~12 GB/s with 16 threads. 32 threads (hyperthreading) actually degrades performance — the average time jumps to 10.9 ms, suggesting contention on the shared cloud machine. Machine A's DDR5 reaches ~21 GB/s, nearly 2x Machine B's bandwidth ceiling.

## Key Findings

### Grayscale conversion is memory-bandwidth bound

Like vector addition in ch02, grayscale does minimal computation per memory access — one weighted sum per 4 bytes transferred (3 bytes read, 1 byte written). The bottleneck is memory bandwidth, not compute. We confirmed this by implementing an integer-only version (multiply + bit-shift instead of floating-point) — it ran at the same speed as the float version, proving the arithmetic is not the limiting factor.

### The GPU does not beat OpenMP on the integrated GPU

On Machine A, the best GPU kernel time (~4.3 ms, 15.6 GB/s) is slower than 16-thread OpenMP (~3.1 ms, 21.3 GB/s). **We do not fully understand why.** In ch02's vector addition, the same integrated GPU achieved ~73 GB/s — well above the CPU's ~49 GB/s on the same DDR5. Several hypotheses were tested:

#### Hypotheses tested

1. **`hipHostRegister` coherency overhead**: We tested an explicit-copy version using `hipMalloc`. The kernel was faster (3.26 ms, 20.5 GB/s) but required 190 ms of pointless copies within the same RAM. This suggests `hipHostRegister` does add some access overhead, but even the `hipMalloc` kernel didn't clearly beat OpenMP.

2. **3-byte RGB stride hurting coalescing**: We tested RGBA (4 bytes per pixel, aligned) — it was actually slightly slower due to moving more data.

3. **Byte-sized loads vs 32-bit loads**: We tested loading RGBA pixels as a single `unsigned int` and extracting channels with bit shifts. This improved throughput from 15.5 to 23.8 GB/s — a significant gain, confirming the GPU prefers wider loads. But still below ch02's 73 GB/s.

4. **Too many threads / scheduling overhead**: We tested a grid-stride loop launching only enough threads to fill the GPU (8 blocks × 1024 threads, ~2188 pixels per thread). Combined with 32-bit loads, this achieved the best GPU result: 3.09 ms, 27.0 GB/s — finally matching OpenMP.

5. **Floating-point vs integer compute**: An integer-only kernel (multiply + shift) ran at the same speed as the float version, ruling out compute as the bottleneck.

#### What we know

- The operation is memory-bandwidth bound on both CPU and GPU.
- The integrated GPU and CPU share the same DDR5, so there is no bandwidth advantage for the GPU (unlike discrete GPUs with dedicated GDDR6/HBM).
- The GPU's advantage in ch02 (~73 GB/s vs ~49 GB/s) came from better latency hiding on contiguous 4-byte float accesses. Grayscale's byte-sized, 3-byte-stride access pattern does not benefit as much from the GPU's memory subsystem.
- The best GPU result required combining multiple optimizations (32-bit packed loads + grid-stride loop) to match what OpenMP achieved with no special tuning.

### Explicit copies are wasteful on integrated GPUs

Same finding as ch02. The explicit-copy version spent ~190 ms copying data that's already in the same physical RAM, for a kernel that takes ~3.3 ms. `hipHostRegister` avoids this entirely by letting the GPU read CPU-allocated memory directly, at the cost of somewhat slower kernel execution due to coherency overhead.

### GPU floating-point rounding differs from CPU

The GPU's FMA (fused multiply-add) instruction produces slightly different rounding than the CPU's separate multiply and add operations. For grayscale this manifests as ±1 differences in pixel values. Verification uses a tolerance of ±1 to account for this. An integer-only formula (`(77*R + 150*G + 29*B) >> 8`) would eliminate this entirely.

### Timing methodology

- **CPU / OpenMP**: `clock_gettime(CLOCK_MONOTONIC)` — wall-clock time. Multiple runs, best time reported.
- **GPU kernel**: `hipEvent` / `cudaEvent` timestamps on the GPU command stream. Measures actual kernel execution, not CPU-side launch overhead.
- **GPU transfers**: `clock_gettime` around `hipMemcpy` / `cudaMemcpy` (blocking calls).
- **Only compute is timed**: Image loading and JPEG writing are excluded from all benchmarks.

## File Structure

```
ch03/
  grayscale.h          # Shared: grayscale_cpu, verify_result, get_time_sec,
                       #         print_stats, parse_args, write_gray_jpg
  grayscale_cpu.c      # Sequential CPU version
  grayscale_openmp.c   # OpenMP parallel version
  grayscale_hip.cpp    # AMD GPU (hipHostRegister + hipMallocManaged)
  grayscale_cuda.cu    # NVIDIA GPU (cudaMalloc + cudaMemcpy)
  Makefile             # Build system (auto-detects available GPU compilers)
  ch03-notes.md        # Study notes on threads, warps, blocks, scheduling
  README.md            # This file
```
