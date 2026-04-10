# GPU Architecture Discussion Notes

## NVIDIA vs AMD Terminology

### Software Hierarchy (top → bottom)

| NVIDIA | AMD | Description |
|--------|-----|-------------|
| Grid | NDRange | Full dispatch of all blocks/workgroups |
| Thread Block | Workgroup | Group of threads assigned together to one SM/CU |
| Warp (32 threads) | Wavefront (64 threads) | Group of threads executing in SIMD lockstep |
| Thread | Work-item | Single unit of execution |

### Hardware Hierarchy (top → bottom)

| NVIDIA | AMD | What it is | Transistor cost |
|--------|-----|------------|-----------------|
| GPU | GPU | Array of SMs/CUs + global memory | — |
| Streaming Multiprocessor (SM) | Compute Unit (CU) | Multiple processing blocks + register file + shared memory + schedulers | Bulk of die area: register file, shared memory, and control logic dominate |
| Processing Block / Sub-partition | SIMD Unit | One control unit + multiple ALUs | Control unit is expensive but shared across many ALUs |
| CUDA Core | Stream Processor (SP) | Individual ALU, no control logic of its own | Small and cheap to replicate |

- This hierarchy implements the SIMD paradigm: one instruction fetch drives many ALUs in parallel. The tradeoff is control divergence causing ALUs to sit idle.
- Typically **16-32 ALUs per control unit** across architectures (e.g., A100: 4 processing blocks x 16 cores = 64 cores per SM).
- The ALUs are a small fraction (~10-15%) of the SM's transistor budget. The register file and shared memory alone likely exceed all ALUs combined.

### Memory Hierarchy

| NVIDIA | AMD | Description |
|--------|-----|-------------|
| Global Memory (DRAM) | Global Memory (VRAM) | Off-chip, high latency, accessible by all threads |
| Shared Memory | Local Data Share (LDS) | On-chip, fast, shared within a block/workgroup |
| Register File | VGPRs / SGPRs | On-chip, fastest, private per thread |

### Software → Hardware Mapping

| Software | Assigned to Hardware |
|----------|---------------------|
| Grid | Distributed across all SMs/CUs |
| Block / Workgroup | One SM/CU (cannot span two) |
| Warp / Wavefront | One Processing Block / SIMD Unit |
| Thread / Work-item | One CUDA Core / SP (for one cycle) |

## SM and CUDA Cores (Streaming Processors)

- A GPU is organized as an array of **Streaming Multiprocessors (SMs)**.
- Each SM contains multiple **CUDA cores** (originally called Streaming Processors/SPs).
- The number of cores per SM varies by architecture (e.g., Ampere A100 has 64 cores per SM, 108 SMs total = 6912 cores).
- Cores within an SM share control logic and memory resources.

## Why ALUs Per Control Unit Matters: The Kepler Lesson

- When ALUs per control unit **<= 32** (warp size): one warp instruction per cycle keeps all ALUs busy. Simple and robust.
- When ALUs per control unit **> 32** (e.g., Kepler's 96): the scheduler must **multi-issue** — dispatch multiple independent instructions from the same warp in one cycle (3 x 32 = 96).
- This only works when the compiler finds enough **instruction-level parallelism (ILP)** — independent instructions with no data dependencies.
- **Dependent** instructions (e.g., `b = a * z` where `a` is computed in the previous instruction) force serialization, leaving ALUs idle.
- In practice, real workloads couldn't consistently provide enough ILP, so Kepler's 96 cores per control unit were often underutilized.
- NVIDIA corrected this from Maxwell onwards (16-32 cores per control unit), relying on **warp-level parallelism** (many warps scheduled across processing blocks) instead of ILP within a single warp.

## Processing Blocks within an SM

- Cores inside an SM are grouped into **processing blocks**.
- Each processing block has its **own instruction fetch/dispatch unit** (control unit).
- Example: A100 has 4 processing blocks with 16 cores each.
- This means an SM has multiple independent control units, allowing different warps to execute different instructions simultaneously.
- SIMD applies **within a warp**, but **across warps** on the same SM, different code can run in parallel.

## Scheduling Hierarchy

| Level | Unit | Role |
|-------|------|------|
| **Block** | Unit of **assignment** | The runtime assigns blocks to SMs. A block cannot span two SMs, but one SM can host multiple blocks. |
| **Warp (32 threads)** | Unit of **scheduling** | Within an SM, warp schedulers independently pick which ready warp executes each cycle. The scheduler doesn't care which block a warp belongs to. |
| **Thread** | Unit of **execution** | All 32 threads in a warp execute the same instruction via SIMD, operating on different data. |

## Warp-to-Hardware Mapping

- A warp of 32 threads is assigned to a processing block.
- A processing block with 16 cores executes a warp's 32 threads over 2 cycles (16 per cycle).
- The number of cores per SM is about **throughput**, not a 1:1 mapping to warps.
- More cores = more warps can make progress per cycle.

## Multiple Blocks on One SM

- Multiple blocks can reside on the same SM simultaneously.
- Warps from all resident blocks share the same pool — warp schedulers draw from all of them.
- Blocks on the same SM **cannot** share data via shared memory or use `__syncthreads()` with each other. Each block gets its own partition of shared memory.
- This restriction enables **transparent scalability** — blocks can be distributed across SMs in any order.

## Synchronization and Transparent Scalability

- `__syncthreads()` works only **within a block**, never across blocks.
- This restriction is what enables **transparent scalability** — since blocks can't depend on each other, the runtime can execute them in any order on any number of SMs. The same code runs on cheap and high-end GPUs without changes.
- Cross-block communication requires separate kernel launches, atomic operations on global memory, or the Cooperative Groups API.
- The runtime guarantees all threads in a block are assigned to the same SM **simultaneously** with all resources reserved upfront. This prevents deadlock at barriers.

### `__syncthreads()` Rules

- Every `__syncthreads()` call must be reached by **all threads in the block** — not just some.
- Placing `__syncthreads()` in an `if` branch and another in the `else` branch creates **two different barriers** — neither will ever have all threads arrive. This causes undefined behavior (deadlock or incorrect results).
- Valid usage: place `__syncthreads()` where all threads unconditionally reach it, or inside a condition that evaluates the same way for all threads in the block.
- These rules apply equally to CUDA and HIP (AMD) — HIP uses the same `__syncthreads()` function name and semantics by design.
- Variants available in both CUDA and HIP: `__syncthreads_count(pred)`, `__syncthreads_and(pred)`, `__syncthreads_or(pred)` — barrier + reduction across the block.

## Warp Execution When Cores < Warp Size

- When a processing block has fewer cores than the warp size (e.g., 16 cores, 32-thread warp), the instruction executes over **multiple cycles** (2 cycles for 16 cores / 32 threads).
- The two halves are atomic from the scheduler's perspective — no other warp is interleaved on the same cores mid-instruction.
- Other processing blocks in the SM execute independently in parallel, so the SM still does useful work every cycle.
- The warp/wavefront size is a **logical** grouping; the physical execution width can be smaller. This is transparent to the programmer.
- **AMD GCN:** 4 SIMD units x 16 SPs each → wave64 takes 4 cycles. **AMD RDNA:** supports both wave32 (native) and wave64 (2 cycles on 32-wide SIMD units).

## Control Divergence

- When threads in the same warp take different control paths, the hardware serializes the paths — one path executes while non-participating threads are masked off (inactive but consuming resources).
- **If/else:** warp executes both paths in separate passes. Pre-Volta: sequential. Volta+: passes may be interleaved (independent thread scheduling).
- **Loops with varying iterations:** all threads active for the minimum iteration count, then threads progressively drop off as they complete.
- **How to spot it:** if the branch condition depends on `threadIdx` values → likely divergence. If the condition is uniform across the warp → no divergence.
- **Boundary conditions** (e.g., `if(i < n)`) are the most common source. Only the last warp of the last block is affected. For large datasets the impact is negligible (<1% for 10,000+ elements).
- For 2D data, only blocks at image edges have divergent warps. Interior blocks have none.

## Resource Partitioning and Occupancy

- **Occupancy** = resident warps / max warps supported by the SM. Higher occupancy → more warps available for latency hiding.
- Three resources constrain occupancy (the most restrictive one wins):
  - **Registers:** Dynamically partitioned among all resident threads. More registers per thread → fewer threads can be resident.
  - **Shared memory:** Partitioned among resident blocks. More shared memory per block → fewer blocks fit.
  - **Block slots:** Hard limit on simultaneous blocks per SM (e.g., 32 on A100), regardless of block size.
- **Dynamic partitioning** means the runtime allocates registers and shared memory based on actual kernel needs — programmers trade per-thread resources for occupancy.
- **Block size affects occupancy.** If block size doesn't divide evenly into the SM's warp capacity, some slots go unused.
- **Block size tradeoff:** More threads per block → more warps for latency hiding, but larger blocks are harder to fit on an SM → fewer blocks per SM → potentially worse occupancy. E.g., a block of 1025 threads on an SM with 2048 max threads → only 1 block fits → ~50% occupancy wasted. A block of 1024 → 2 blocks fit → 100%.
- Sweet spot is typically **128-256 threads per block** — divides evenly into SM limits, leaves room for multiple blocks, balances all three constraints.
- Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` or the CUDA Occupancy Calculator to analyze these tradeoffs.

## Querying Device Properties

- **Purpose:** `cudaGetDeviceProperties(&devProp, i)` fills a `cudaDeviceProp` struct with SM count, max threads per block, registers per SM, warp size, grid/block dimension limits, etc.
- **Practical takeaway:** Portable CUDA code should query these properties and adapt block sizes and grid dimensions accordingly. Useful for autotuning systems that sweep configurations to find the best performance on the target hardware.

## Warp Scheduling and Latency Tolerance

- An SM has many more **resident warps** than it can execute simultaneously. Only a few run each cycle; the rest are ready or waiting.
- When a warp stalls (memory access, instruction dependency), the scheduler picks another ready warp **instantly at zero cost** — all resident warps' registers are already in the on-chip register file. No save/restore needed.
- This is fundamentally different from **CPU context switching**, which is expensive because registers must be saved to and restored from memory.
- This is the primary mechanism for tolerating long-latency operations (especially global memory access) on GPUs.
- **Sufficient resident warps are critical.** If too few warps are on the SM, the scheduler has nothing to switch to when a warp stalls → cores go idle. This is the direct link to **occupancy** (Section 4.7).
