# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

There is no top-level build system. Each day's programs are compiled and run individually with `nvcc`.

**Standard pattern for `.cu` / `.cpp` files:**
```bash
# Compile
nvcc -O3 -o <output> <source>.cu
# or for Driver API files that use libcuda directly
nvcc -O3 -lcuda -o <output> <source>.cpp

# Run
./<output>
```

**Day 1 – CUBIN loader (Driver API, C++):**
```bash
nvcc -O3 -lcuda -o run_cubin run_cubin.cpp
./run_cubin
```

**Day 2/3 – PTX manual loader (Driver API):**
```bash
# First generate the PTX for the kernel it loads:
nvcc -ptx vector_add.cu -o vector_add.ptx
nvcc -O3 -lcuda -o run_ptx_manual run_ptx_manual.cu
./run_ptx_manual
```

**Day 6 – compile all attention variants at once:**
```bash
cd day6
bash compile_all.sh
./flash_attention
```

**Day 7 – performance tuning:**
```bash
cd day7
nvcc -O3 -arch=sm_89 -o performance_tuning performance_tuning.cu
./performance_tuning
```

**Profile a kernel (requires Nsight Compute):**
```bash
ncu --set full ./<binary>
```

**Generate PTX or CUBIN from a `.cu` file:**
```bash
nvcc -ptx -o output.ptx source.cu       # PTX intermediate
nvcc -cubin -arch=sm_89 -o output.cubin source.cu   # CUBIN for sm_89
```

## Environment

- CUDA 12.4 at `/usr/local/cuda-12.4`; `nvcc` should be on `$PATH`
- Target GPU: RTX 4090, compute capability **sm_89** (Ada Lovelace)
- For Hopper (H100) use `sm_90`; for Blackwell use `sm_100`

## Architecture

Each `dayN/` directory is a self-contained lesson — standalone executables, no shared library between days.

**Two API layers appear throughout:**
- **Runtime API** (`cuda_runtime.h`) — used for most `.cu` kernels (vector add, matrix mul, CNN, attention). Simpler; `cudaMalloc`/`cudaMemcpy`/kernel-launch syntax (`<<<grid, block>>>`).
- **Driver API** (`cuda.h`) — used in `run_cubin.cpp` and `run_ptx_manual.cu`. Requires explicit `cuInit`/`cuCtxCreate`/`cuModuleLoad`/`cuLaunchKernel`. Day 1's `run_cubin.cpp` wraps it in RAII classes (`CUDADevice`, `CUDAModule`, `CUDAMemory`).

**Progression of optimization techniques across days:**

| Day | Key technique |
|-----|--------------|
| 1 | Basic kernel launch, thread indexing, CUBIN dynamic loading |
| 2 | PTX emission/loading, Driver API, perf analysis tools |
| 3 | Shared memory tiling (TILE_SIZE), coalesced access, CUDA streams |
| 4 | 2D convolution, separable convolution, shared memory halo |
| 5 | Self-attention, multi-head attention (theory + structure) |
| 6 | Flash Attention (online softmax), GQA, mixed-precision (FP16/BF16), sparse attention |
| 7 | Occupancy tuning, register/shared-memory trade-offs, Blackwell `tcgen05.mma` |

**Day 6 attention kernels** are the most complex: `flash_attention.cu` implements the online-softmax tiled algorithm (no O(N²) attention matrix); `mixed_precision_attention.cu` demonstrates FP16 Tensor Core paths; `sparse_attention.cu` skips zero blocks.

**Day 1 `run_cubin.cpp`** is the sole cross-module dependency: `gpu_info.py` calls into it to demonstrate dynamic kernel loading via the Driver API. All other files are fully independent.
