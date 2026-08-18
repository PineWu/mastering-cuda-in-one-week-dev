# CUDA Kernel Template Framework

Production starting point for writing a new custom CUDA kernel. Drop
`cuda_kernel_template.cuh` into your project; copy and adapt
`cuda_kernel_template.cu` for each new kernel.

---

## File Layout

```
template/
├── cuda_kernel_template.cuh   ← reusable infrastructure (include this)
├── cuda_kernel_template.cu    ← example kernels + main (copy and adapt)
└── CUDA_KERNEL_TEMPLATE.md    ← this document
```

The `.cuh` / `.cu` split is intentional: the header is a stable
infrastructure layer you `#include` from every new kernel file; the `.cu`
is a concrete starting point you fork.

---

## Compile Commands

```bash
# Release — optimise, target Ada Lovelace (RTX 40xx / sm_89)
nvcc -O3 -arch=sm_89 -o kernel_template cuda_kernel_template.cu

# Debug — device printf + verbose logging + CUDA-memcheck-friendly
nvcc -O0 -g -G -DCUDA_DEBUG -DCUDA_LOG_LEVEL=3 \
     -o kernel_template cuda_kernel_template.cu

# Profile — NVTX ranges visible in Nsight Systems / Nsight Compute
nvcc -O3 -arch=sm_89 -DCUDA_PROFILE \
     -o kernel_template cuda_kernel_template.cu -lnvtx3

# Run
./kernel_template
```

For other architectures replace `-arch=sm_89`:

| GPU family         | Flag       |
|--------------------|------------|
| Ampere (A100/30xx) | `sm_80/86` |
| Hopper (H100)      | `sm_90`    |
| Ada (RTX 40xx)     | `sm_89`    |
| Blackwell (B200)   | `sm_100`   |

---

## Compile Flags Reference

All flags are **additive and orthogonal** — combine freely.

| Flag | Default | Effect |
|------|---------|--------|
| `-DCUDA_LOG_LEVEL=0` | — | Errors only |
| `-DCUDA_LOG_LEVEL=1` | — | + WARN |
| `-DCUDA_LOG_LEVEL=2` | **on** | + INFO |
| `-DCUDA_LOG_LEVEL=3` | — | + DEBUG (includes buffer alloc/free) |
| `-DCUDA_DEBUG` | off | Enables `KERNEL_LOG` / `KERNEL_LOG_EVERY` device printf and `verifyResult` calls |
| `-DCUDA_PROFILE` | off | Enables `NVTX_PUSH/POP/MARK`; link `-lnvtx3` |

---

## Infrastructure Reference

### `CUDA_CHECK(expr)` — Error Handling

Wraps any CUDA Runtime call. Throws `std::runtime_error` on failure,
including the error string, file name, and line number.

```cpp
CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
```

Use `CUDA_CHECK_LAUNCH()` **immediately after** a kernel launch to catch
invalid-grid or too-much-shared-memory errors:

```cpp
myKernel<<<grid, block>>>(args...);
CUDA_CHECK_LAUNCH();
```

Or use the single-expression convenience macro:

```cpp
CUDA_LAUNCH(myKernel, cfg, arg1, arg2, arg3);
// expands to: myKernel<<<cfg.grid, cfg.block, cfg.shared_mem, cfg.stream>>>(...)
//             CUDA_CHECK_LAUNCH()
```

---

### `LOG_INFO / LOG_WARN / LOG_ERROR / LOG_DEBUG` — Logging

Zero-dependency macros writing to `stderr` with timestamp, file, and line.
Controlled at compile time by `-DCUDA_LOG_LEVEL=N` (no runtime overhead
for suppressed levels).

```cpp
LOG_INFO("Allocated %zu MB", bytes / (1024*1024));
LOG_DEBUG("tile t=%d  row=%d  col=%d", t, row, col);
LOG_WARN("Tolerance relaxed to %.1e", tol);
LOG_ERROR("Unexpected shape: M=%d N=%d", M, N);
```

Sample output:
```
[14:23:01 INFO  cuda_kernel_template.cu:91] Allocated 128 MB
[14:23:01 ERROR cuda_kernel_template.cu:42] [CUDA Error] out of memory | ...
```

---

### `##__VA_ARGS__` — 变参宏中的逗号消除

所有 `LOG_*` 宏和 `KERNEL_LOG` 都使用了 `##__VA_ARGS__` 而不是普通的 `__VA_ARGS__`，原因如下。

**问题根源**：C99/C++11 的标准变参宏在可变参数为空时会残留一个多余的逗号：

```c
// 标准写法 (__VA_ARGS__ without ##)
#define LOG_INFO(fmt, ...) fprintf(stderr, fmt "\n", __VA_ARGS__)

LOG_INFO("hello")
// 展开为: fprintf(stderr, "hello" "\n", )
//                                      ↑ 多余的逗号 → 编译报错
```

**`##` 的作用**：GCC/Clang 扩展，在可变参数为空时自动删除前面的 `,`：

```c
// 本文件的写法 (##__VA_ARGS__)
#define LOG_INFO(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

LOG_INFO("hello")          // → fprintf(stderr, "hello" "\n")          ✓
LOG_INFO("n=%d", 42)       // → fprintf(stderr, "n=%d" "\n", 42)       ✓
LOG_INFO("a=%d b=%d", a, b)// → fprintf(stderr, "a=%d b=%d" "\n", a, b)✓
```

**可移植性**：

| 写法 | 支持环境 |
|---|---|
| `##__VA_ARGS__` | GCC / Clang / **NVCC**（事实标准，CUDA 完全支持） |
| `__VA_OPT__(,) __VA_ARGS__` | C++20 标准写法，需 `-std=c++20` |

CUDA 代码使用 `##__VA_ARGS__` 是正确且惯用的写法——NVCC 底层走 Clang/GCC host 编译器，完全支持此扩展。

---

### `KERNEL_LOG / KERNEL_LOG_EVERY` — Device-side Debug Prints

Active only when compiled with `-DCUDA_DEBUG`; expand to `((void)0)` otherwise.

```cpp
// In kernel body:
KERNEL_LOG("n=%d scale=%.3f", n, scale);          // first thread only
KERNEL_LOG_EVERY("idx=%d val=%.4f", idx, val);    // every thread — very verbose
```

`KERNEL_LOG` is safe to leave in production kernels because it compiles
away completely. `KERNEL_LOG_EVERY` should always be guarded:

```cpp
if (blockIdx.x == 0 && threadIdx.x < 8)
    KERNEL_LOG_EVERY("tile[%d]=%.4f", threadIdx.x, sA[threadIdx.y][threadIdx.x]);
```

---

### `NVTX_PUSH / NVTX_POP / NVTX_MARK` — Profiling Markers

Expand to `nvtxRangePushA/nvtxRangePop/nvtxMarkA` when compiled with
`-DCUDA_PROFILE -lnvtx3`. Expand to `((void)0)` otherwise.

```cpp
NVTX_PUSH("myKernel");
CUDA_LAUNCH(myKernel, cfg, args...);
NVTX_POP();
```

Named ranges appear as coloured bars in the Nsight Systems timeline.

---

### `CudaBuffer<T>` — RAII Device Memory

Owns a `cudaMalloc`'d buffer; frees it on destruction. Exception-safe by
construction — no manual cleanup even if a `CUDA_CHECK` throws.

```cpp
CudaBuffer<float> d_x(N);          // allocate N floats on device
d_x.copyFromHost(h_x.data());      // sync H→D copy
d_x.zero();                        // memset to 0
d_x.copyToHost(h_x.data());        // sync D→H copy

// Async variants
d_x.copyFromHost(h_x.data(), 0, stream);
d_x.copyToHost(h_x.data(),   0, stream);

// Access raw pointer for kernel arguments
myKernel<<<...>>>(d_x.get(), d_x.count());
```

`CudaBuffer` is non-copyable and movable:

```cpp
CudaBuffer<float> moved = std::move(d_x);  // ownership transferred
```

---

### `LaunchConfig` + `make1DConfig` / `make2DConfig` — Grid/Block Sizing

Carry `grid`, `block`, `shared_mem`, and `stream` in one struct to keep
launch sites readable and to enable logging of the launch configuration.

```cpp
// 1-D element-wise kernel
LaunchConfig cfg = make1DConfig(/*n=*/N, /*block_size=*/256);
cfg.print("myKernel");   // logs grid/block/smem/stream at INFO level

// 2-D kernel (e.g. matmul): nx=columns, ny=rows
LaunchConfig cfg = make2DConfig(/*nx=*/N, /*ny=*/M, /*bx=*/16, /*by=*/16);

// Custom: add dynamic shared memory or a non-default stream
LaunchConfig cfg = make1DConfig(N, 256, /*shared_mem=*/smem_bytes, stream);

// Launch
CUDA_LAUNCH(myKernel, cfg, d_in, d_out, N);
```

> **Static vs dynamic shared memory.** If your kernel uses statically
> declared `__shared__` arrays (the common case), set `shared_mem = 0`.
> Only set it to a non-zero value when the kernel declares
> `extern __shared__ float buf[];` and reads the size from the third
> launch argument.

---

### `CudaTimer` — Ad-hoc Timing

RAII wrapper around a pair of `cudaEvent_t`. Use during development when
you want to time one region interactively.

```cpp
CudaTimer timer;
timer.start();
CUDA_LAUNCH(myKernel, cfg, args...);
float ms = timer.stop();
LOG_INFO("%.3f ms", ms);
```

For repeatable, multi-iteration results use `benchmarkKernel` instead.

---

### `benchmarkKernel<F>` — Structured Benchmarking

```cpp
auto result = benchmarkKernel(
    "myKernel (label)",           // printed label
    [&]() {                        // zero-argument lambda wrapping the launch
        CUDA_LAUNCH(myKernel, cfg, d_in, d_out, N);
    },
    2LL * N,                       // FLOPs per call (0 → skip GFLOPS)
    /*warmup=*/10,
    /*iters=*/200);

// result.ms_avg / ms_min / ms_max / gflops / iters
```

`benchmarkKernel` automatically:
1. Runs `warmup` iterations discarded from timing.
2. Synchronises the device (or stream) before the timed loop.
3. Times each iteration individually using `CudaTimer`.
4. Computes min / avg / max latency and peak GFLOPS.
5. Logs the result at `INFO` level.

---

### `verifyResult<T>` — Correctness Checking

```cpp
verifyResult(h_ref.data(), h_out.data(), N, /*tol=*/1e-4f);
```

Compares every element; logs `PASSED` or `FAILED` with the maximum
absolute error. Call after `cudaDeviceSynchronize` + `copyToHost`.

Use a relaxed tolerance for kernels with long FP32 accumulation chains
(e.g. `1e-3f` for matmul with K ≥ 256).

---

### `printDeviceInfo()` — GPU Properties

Logs name, compute capability, SM count, threads-per-SM, shared memory,
global memory, and estimated peak memory bandwidth. Call once in `main`.

---

## How to Fork a New Kernel

### Step 1 — Copy the source file

```bash
cp template/cuda_kernel_template.cu myproject/my_kernel.cu
```

### Step 2 — Include the header

```cpp
#include "cuda_kernel_template.cuh"
using namespace cuda_template;
```

### Step 3 — Replace the example kernels

Delete `scaleBiasKernel` and `tiledMatMulKernel` (and their wrappers).
Write your kernel following this structure:

```cpp
__global__ void myKernel(const float* __restrict__ in,
                          float* __restrict__       out,
                          int n) {
    // 1. Log kernel config once (compiled away in release)
    KERNEL_LOG("myKernel  n=%d", n);

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 2. Kernel logic
    out[idx] = in[idx] * 2.0f;
}
```

### Step 4 — Write a launch wrapper

```cpp
void launchMyKernel(const float* d_in, float* d_out, int n,
                    const LaunchConfig& cfg) {
    NVTX_PUSH("myKernel");
    CUDA_LAUNCH(myKernel, cfg, d_in, d_out, n);
    NVTX_POP();
}
```

### Step 5 — Wire up in main

```cpp
CudaBuffer<float> d_in(N), d_out(N);
d_in.copyFromHost(h_in.data());

LaunchConfig cfg = make1DConfig(N);
cfg.print("myKernel");

launchMyKernel(d_in.get(), d_out.get(), N, cfg);
CUDA_CHECK(cudaDeviceSynchronize());

d_out.copyToHost(h_out.data());
verifyResult(h_ref.data(), h_out.data(), N);

benchmarkKernel("myKernel", [&]() {
    launchMyKernel(d_in.get(), d_out.get(), N, cfg);
}, /*flops=*/N, 10, 200);
```

### Step 6 — Iterate

| Phase | Command |
|-------|---------|
| Correctness | `nvcc -O0 -g -G -DCUDA_DEBUG -DCUDA_LOG_LEVEL=3 -o k my_kernel.cu && ./k` |
| Profiling | `nvcc -O3 -arch=sm_89 -DCUDA_PROFILE -o k my_kernel.cu -lnvtx3 && ncu --set full ./k` |
| Release | `nvcc -O3 -arch=sm_89 -o k my_kernel.cu && ./k` |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Exceptions over `exit(1)` | Template is intended as a PyTorch/pybind11 extension starting point; `exit(1)` would crash the Python process. |
| Macro logging, not spdlog | Zero external dependencies; the template drops into any repo without CMake changes. |
| `CudaBuffer<T>` RAII | Makes cleanup exception-safe. Mirrors the pattern already in `day1/run_cubin.cpp`. |
| `LaunchConfig` struct | Prevents silent swapped-argument bugs (e.g. `grid` and `block` swapped), makes launch sites self-documenting, and enables `cfg.print()` for quick sanity checks. |
| `benchmarkKernel` lambda | Decouples the benchmark harness from kernel call syntax; a single harness handles any kernel signature. |
| Three orthogonal flags | `CUDA_LOG_LEVEL`, `CUDA_DEBUG`, `CUDA_PROFILE` can be combined freely. All are zero-cost when off — no `#ifdef` clutter needed in kernel bodies. |
| `shared_mem = 0` for static smem | Static `__shared__` arrays do not consume the dynamic shared-memory argument. Setting it non-zero is harmless but misleading; the comment in the `.cu` clarifies this. |
