# Chapter 5 — Memory Architecture and Data Locality

## Compute to global memory access ratio

The **compute to global memory access ratio**, also called **arithmetic intensity**, is the number of floating-point operations (FLOPs) performed for each byte accessed from global memory. It is the fundamental metric for determining whether a kernel is **compute-bound** or **memory-bound**.

Every GPU has a specific balance point determined by its hardware:

```
balance point = peak FLOPS / peak memory bandwidth
```

For example, if a GPU has 10 TFLOPS of compute and 500 GB/s of bandwidth:

```
10,000 GFLOPS / 500 GB/s = 20 FLOPs per byte
```

- **Below 20 FLOPs/byte** — memory-bound. The ALUs are starved, waiting for data. Adding more compute would not help; the bottleneck is how fast data can be fetched from global memory.
- **Above 20 FLOPs/byte** — compute-bound. The memory system can keep up; the bottleneck is arithmetic throughput.

This ratio determines the correct performance metric to report: **GB/s** for memory-bound kernels (how efficiently you use bandwidth) and **GFLOPS** for compute-bound kernels (how efficiently you use ALUs).

### Why it matters

The ratio explains why the same GPU can dominate on one workload and barely match the CPU on another. A kernel with a low ratio (e.g., vector addition: 1 FLOP per 12 bytes = 0.08 FLOPs/byte) will be entirely limited by memory bandwidth. No amount of parallel ALUs will help — the hardware is idle waiting for data. A kernel with a high ratio (e.g., matrix multiplication with tiling) can keep thousands of ALUs busy because each byte fetched from global memory feeds many operations.

### Improving the ratio

The ratio is not a fixed property of the algorithm — it can be improved by reducing the number of global memory accesses. This is the central optimization strategy for GPU programming:

- **Tiling with shared memory** — load a block of data from global memory into fast on-chip shared memory once, then reuse it many times. This reduces global memory traffic without reducing computation, increasing the effective ratio.
- **Register reuse** — keep frequently accessed values in registers instead of re-reading from global memory.
- **Memory coalescing** — while this doesn't change the ratio directly, it ensures that the bytes you do access are transferred efficiently (fewer transactions for the same data).

The goal is to move a kernel's arithmetic intensity above the hardware's balance point, transforming it from memory-bound to compute-bound. Chapter 5 introduces tiling as the primary technique for achieving this.

## CUDA memory types

A CUDA device has several types of memory, each with different speed, scope, and lifetime:

| Declaration | Memory | Scope | Lifetime | Location |
|---|---|---|---|---|
| Automatic scalar variables | **Register** | Thread | Grid (kernel) | On-chip (SM register file) |
| Automatic array variables | **Local** | Thread | Grid (kernel) | Off-chip (DRAM) |
| `__shared__` | **Shared** | Block | Grid (kernel) | On-chip (SM SRAM) |
| `__device__` | **Global** | Grid | Application | Off-chip (DRAM) |
| `__constant__` | **Constant** | Grid | Application | Off-chip (DRAM), cached on-chip |

### Registers

The fastest memory. Every automatic scalar variable declared in a kernel (`float sum`, `int row`, etc.) goes into registers. Each thread gets its own private copy. Access takes ~1 cycle.

The SM has a fixed register file (e.g., 65,536 registers on A100). These are dynamically partitioned among all resident threads. More registers per thread means fewer threads can be resident:

- 32 registers/thread → 2,048 threads → full occupancy
- 64 registers/thread → 1,024 threads → half occupancy
- 128 registers/thread → 512 threads → quarter occupancy

When a kernel needs more registers than available, the compiler **spills** excess registers to local memory (off-chip DRAM). This is the worst outcome — slow DRAM latency for what should be an instant register access, and occupancy doesn't improve since the thread still claims the maximum register allocation.

### Local memory

Despite the name, local memory lives in **off-chip DRAM** — the same physical memory as global memory. "Local" only means it's private to a thread, not that it's physically local to the SM. It has the same hundreds-of-cycles latency as global memory. Local memory is used for automatic arrays that can't fit in registers and for spilled register values.

### Shared memory

On-chip SRAM, explicitly managed by the programmer. All threads in a block see the same shared memory. This is the key tool for tiling — threads cooperatively load data from global memory into shared memory, synchronize with `__syncthreads()`, then reuse the data many times from fast on-chip storage.

On modern NVIDIA GPUs (since Volta), shared memory and the L1 cache share the same physical SRAM. The split is configurable — more shared memory means less L1 cache and vice versa.

### Global memory

Off-chip DRAM (VRAM on discrete GPUs, system RAM on integrated GPUs). Large, slow (~hundreds of cycles), accessible by all threads and the host. This is what `cudaMalloc` / `hipMalloc` allocates. The L1 and L2 caches help with global memory accesses but are hardware-managed — you can't guarantee data stays cached.

### Constant memory

Stored in global memory but aggressively cached on-chip. Read-only from the device, written by the host. Limited to 64 KB. When all threads in a warp read the same address, the cached value is broadcast to all threads in a single cycle. Good for values like convolution filter coefficients that every thread reads. Covered in detail in chapter 7.

## GPU caches

In addition to the programmer-managed memories above, GPUs have **hardware-managed caches** similar to CPUs:

- **L1 cache** — per SM, fast, caches global and local memory accesses
- **L2 cache** — shared across all SMs, medium speed

These caches work automatically — the programmer doesn't control what gets cached. When a thread reads from global memory, data passes through L2 then L1 on the way to registers. If another thread on the same SM reads a nearby address, the cache may still have it.

### GPU caches vs CPU caches

CPU caches exploit **temporal locality within a single thread** — the same thread accessing the same data again. GPU caches primarily exploit **spatial locality across threads in a warp** — 32 threads accessing adjacent addresses, which the hardware coalesces into a few cache line fetches (128 bytes on NVIDIA, 64 bytes on AMD).

GPU caches are much smaller per thread than CPU caches because thousands of threads would thrash a large cache. GPUs rely on **massive parallelism to hide latency** (warp switching) rather than caching to avoid it.

### Shared memory vs caches

- **Shared memory** — you control it. You decide what gets loaded, when, and it stays until you're done. Guaranteed fast access for predictable reuse patterns.
- **L1/L2 cache** — the hardware controls it. Data might be there, might not. Best-effort, not guaranteed.

This is why the book focuses on shared memory for optimization — it gives guaranteed performance, while caches are transparent helpers.

## Registers and occupancy

Using too many registers per thread directly reduces occupancy. This is a fundamental tradeoff:

- **More registers** → faster per-thread execution (no spilling), but fewer resident warps → less latency hiding
- **Fewer registers** → more resident warps → better latency hiding, but potential spilling to slow local memory

You can inspect register usage with `nvcc --ptxas-options=-v` and cap it with `--maxrregcount=N` or `__launch_bounds__` to force the compiler to use fewer registers at the cost of more spilling.

The "performance cliff" occurs when a small increase in register usage causes a large drop in occupancy. For example, if going from 31 to 33 registers per thread causes the SM to fit 3 blocks instead of 4, the occupancy drops by 25% from a change of just 2 registers.

## Impact of memory usage on occupancy

Shared memory, like registers, is a limited per-SM resource. Using too much shared memory per block means fewer blocks can be resident on an SM, reducing occupancy and hurting latency hiding.

### The three resources that compete for occupancy

1. **Registers** per thread — more registers per thread means fewer resident threads
2. **Shared memory** per block — more shared memory per block means fewer resident blocks
3. **Block slots** per SM — hard limit on number of concurrent blocks regardless of their size (e.g., 32 on A100)

The most restrictive resource wins. You can have plenty of registers and shared memory available but still hit the block slot limit if your blocks are too small.

### Dynamic shared memory sizing

Shared memory arrays declared with a compile-time size work well when you know the hardware in advance:

```c
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];   // size fixed at compile time
```

But different GPUs have different shared memory capacities (48 KB on older cards, up to 164 KB on A100). A kernel hardcoded to a smaller tile size won't benefit from the extra memory on newer hardware. The solution is `extern __shared__`, which sizes the array at kernel launch time:

```c
extern __shared__ float shared_data[];          // size unknown at compile time

// Launch — third <<<>>> argument specifies shared memory size in bytes:
kernel<<<grid, block, shared_bytes>>>(args);
```

The host code queries `cudaGetDeviceProperties` for `sharedMemPerBlock`, computes an appropriate tile size, and passes the size at launch.

**Tradeoffs:**
- You lose the compile-time known size, which enables some compiler optimizations
- `extern __shared__` gives you a single flat 1D array — if you need multiple shared arrays (like `Mds` and `Nds`), you have to manually compute offsets into the flat array
- Kernel code has to use linearized indexing like `shared_data[ty * TILE_WIDTH + tx]` instead of `Mds[ty][tx]`
