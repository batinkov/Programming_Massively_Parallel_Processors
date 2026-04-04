# Chapter 3 — Notes

## threadIdx and blockIdx

The built-in variables are `threadIdx` and `blockIdx` (with `Idx`, not `Id`).

- **`threadIdx`** ranges from `0` to `blockDim - 1`. `blockDim` is the number of threads per block. So if you launch 256 threads per block, `threadIdx` goes 0..255.
- **`blockIdx`** ranges from `0` to `gridDim - 1`. `gridDim` is the number of blocks in the grid.
- **Grid = entire kernel launch**. One kernel launch creates one grid, which contains all the blocks, which contain all the threads.

### These are multi-dimensional

`threadIdx` and `blockIdx` are `dim3` structs with `.x`, `.y`, `.z` components. For 1D problems (like vector addition) you only use `.x`. For 2D problems (like matrix operations) you'd use `.x` and `.y`. The same applies to `blockDim` and `gridDim`.

### 1D launch structure

```
Grid (gridDim.x blocks)
├── Block 0:  threadIdx.x = 0, 1, 2, ..., blockDim.x - 1
├── Block 1:  threadIdx.x = 0, 1, 2, ..., blockDim.x - 1
├── ...
└── Block gridDim.x - 1:  threadIdx.x = 0, 1, 2, ..., blockDim.x - 1
```

### Global index pattern

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This gives each thread a unique index across the entire grid. Same idea as converting (row, column) to a flat array index.

## Warps and Wavefronts

### Execution model

Threads are not executed individually. They are grouped into **warps** (NVIDIA, 32 threads) or **wavefronts** (AMD, 64 threads — though RDNA architectures like the Radeon 860M use 32). The warp is the fundamental, indivisible scheduling unit. The hardware doesn't even have a concept of "one thread" at the execution level — it only sees warps.

### Warp assignment is fixed

A thread's warp assignment is determined at launch time and never changes. Threads 0-31 in a block form warp 0, threads 32-63 form warp 1, and so on. No migration, no reassignment for the lifetime of the kernel.

### SIMT execution

All threads in a warp share the same program counter — they execute the same instruction at the same time (Single Instruction, Multiple Threads).

### Branch divergence

If threads in a warp take different paths at an `if`, both paths execute sequentially — threads not on the active path are **masked** (they don't write results or read from memory effectively). Once paths reconverge, full utilization resumes. If *all* threads in a warp take the same branch, there's no penalty — the other path is skipped entirely.

### Latency hiding via zero-cost warp switching

All warps resident on an SM/CU have their registers allocated simultaneously — nothing gets saved or restored on a switch. The warp scheduler simply picks a different warp that has its operands ready. This is **zero-cost context switching**, fundamentally different from CPU context switches. This is why GPUs need thousands of threads in flight — the more warps resident, the more likely one is ready to execute while others wait on memory.

### Occupancy

The number of warps that can be resident on an SM/CU is limited by register usage and shared memory usage per block. If each thread uses many registers, fewer warps fit, reducing the GPU's ability to hide latency. This tradeoff is called **occupancy**.

## Row-major vs Column-major

### Memory layout

A 2D array is stored in a flat 1D memory. The layout determines which elements are adjacent.

**Row-major (C, C++, NumPy default)** — stores row by row. `matrix[i][j] = *(base + i * NUM_COLS + j)`

**Column-major (Fortran, MATLAB, Julia)** — stores column by column. `matrix[i][j] = *(base + j * NUM_ROWS + i)`

### Why it matters — cache lines

Cache fetches entire lines (typically 64 bytes), so accessing contiguous memory is fast. The inner loop should always iterate over the contiguous dimension:

```c
// Row-major (C): inner loop over columns — GOOD
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i][j];  // sequential in memory

// Row-major (C): inner loop over rows — BAD
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i][j];  // jumps by cols elements each step
```

### GPU relevance — memory coalescing

Same principle applies: when 32 threads in a warp access global memory simultaneously, adjacent threads should access adjacent addresses. This allows the hardware to combine accesses into fewer memory transactions.

## Block scheduling

The grid is a logical description — the GPU does not create all threads at once. Blocks are assigned to SMs as resources become available. When a block completes and frees its resources (registers, shared memory), the scheduler assigns a new block to that SM. This continues until all blocks have completed.

- **Blocks are the scheduling unit**, not threads. An entire block is assigned to one SM and stays there until all its threads finish.
- **Blocks never migrate** between SMs.
- **Scheduling order is undefined** — you cannot rely on any particular block running before another. This is what makes the model scalable: the same kernel runs on GPUs with different SM counts without code changes.

## Shared memory

Each block has access to **shared memory** (CUDA) / **LDS — Local Data Store** (AMD). Threads in different blocks cannot share memory this way — they can only communicate through global memory.
