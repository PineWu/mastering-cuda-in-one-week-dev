// =============================================================================
// cuda_kernel_template.cuh
// Reusable CUDA kernel infrastructure.
//
// Drop this header into any project and #include it in your .cu file.
// All components live in namespace cuda_template; macros are at file scope.
//
// Compile flags (all orthogonal, all zero-cost when off):
//   -DCUDA_LOG_LEVEL=N   0=ERROR 1=+WARN 2=+INFO(default) 3=+DEBUG
//   -DCUDA_DEBUG         Device-side printf guards + host verifyResult
//   -DCUDA_PROFILE       NVTX range markers (link -lnvtx3)
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>

// =============================================================================
// Section 1: Error Handling
// =============================================================================

// Wrap any CUDA Runtime API call; throws std::runtime_error on failure.
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("[CUDA Error] ") + cudaGetErrorString(_e) +        \
                " | " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)

// Call immediately after a kernel launch to catch configuration errors
// (invalid grid, too much shared mem, etc.).
#define CUDA_CHECK_LAUNCH()                                                     \
    do {                                                                        \
        cudaError_t _e = cudaGetLastError();                                   \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("[Kernel Launch Error] ") + cudaGetErrorString(_e) + \
                " | " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)

// Convenience: launch a kernel via LaunchConfig and immediately check for errors.
// Usage: CUDA_LAUNCH(myKernel, cfg, arg1, arg2, ...);
#define CUDA_LAUNCH(kernel, cfg, ...)                                           \
    do {                                                                        \
        (kernel)<<<(cfg).grid, (cfg).block,                                    \
                   (cfg).shared_mem, (cfg).stream>>>(__VA_ARGS__);             \
        CUDA_CHECK_LAUNCH();                                                    \
    } while (0)

// =============================================================================
// Section 2: Logging  (host-side, stderr, zero-dep)
// =============================================================================

#ifndef CUDA_LOG_LEVEL
#define CUDA_LOG_LEVEL 2
#endif

namespace cuda_template {
namespace detail {
inline const char* log_ts() {
    static char buf[16];
    time_t t = time(nullptr);
    strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&t));
    return buf;
}
} // namespace detail
} // namespace cuda_template

// ERROR always prints (cannot be silenced by CUDA_LOG_LEVEL).
#define LOG_ERROR(fmt, ...)                                                     \
    fprintf(stderr, "[%s ERROR %s:%d] " fmt "\n",                             \
            cuda_template::detail::log_ts(), __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_WARN(fmt, ...)                                                      \
    do { if (CUDA_LOG_LEVEL >= 1)                                               \
        fprintf(stderr, "[%s WARN  %s:%d] " fmt "\n",                         \
                cuda_template::detail::log_ts(), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

#define LOG_INFO(fmt, ...)                                                      \
    do { if (CUDA_LOG_LEVEL >= 2)                                               \
        fprintf(stderr, "[%s INFO  %s:%d] " fmt "\n",                         \
                cuda_template::detail::log_ts(), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

#define LOG_DEBUG(fmt, ...)                                                     \
    do { if (CUDA_LOG_LEVEL >= 3)                                               \
        fprintf(stderr, "[%s DEBUG %s:%d] " fmt "\n",                         \
                cuda_template::detail::log_ts(), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

// =============================================================================
// Section 3: NVTX Profiling Markers
// Compile with -DCUDA_PROFILE and link -lnvtx3 to enable.
// All macros expand to ((void)0) in release builds — zero overhead.
// =============================================================================

#ifdef CUDA_PROFILE
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP()      ((void)0)
#define NVTX_MARK(name) ((void)0)
#endif

// =============================================================================
// Section 4: Device-side Debug Prints
// Compile with -DCUDA_DEBUG to enable.
//
// KERNEL_LOG        — prints once from block(0,0,0) thread(0,0,0) only.
//                     Use for kernel-wide state (sizes, strides, config).
// KERNEL_LOG_EVERY  — prints from every thread.
//                     Extremely verbose; add your own if-guard.
// =============================================================================

#ifdef CUDA_DEBUG
#define KERNEL_LOG(fmt, ...)                                                    \
    do {                                                                        \
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&          \
            threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {       \
            printf("[DEVICE B(0,0,0) T(0,0,0)] " fmt "\n", ##__VA_ARGS__);    \
        }                                                                       \
    } while (0)

#define KERNEL_LOG_EVERY(fmt, ...)                                              \
    printf("[DEVICE B(%u,%u,%u) T(%u,%u,%u)] " fmt "\n",                       \
           blockIdx.x, blockIdx.y, blockIdx.z,                                 \
           threadIdx.x, threadIdx.y, threadIdx.z, ##__VA_ARGS__)
#else
#define KERNEL_LOG(fmt, ...)       ((void)0)
#define KERNEL_LOG_EVERY(fmt, ...) ((void)0)
#endif

// =============================================================================
// Remaining components are in namespace cuda_template
// =============================================================================

namespace cuda_template {

// =============================================================================
// Section 5: RAII Device Memory Buffer
//
// Owns a device allocation; frees it on destruction.
// Non-copyable, movable.  Use std::move to transfer ownership.
// =============================================================================

template <typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count) : count_(count), ptr_(nullptr) {
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        LOG_DEBUG("CudaBuffer alloc: %zu elems (%zu bytes) @ %p",
                  count_, count_ * sizeof(T), static_cast<void*>(ptr_));
    }

    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            LOG_DEBUG("CudaBuffer free: %p", static_cast<void*>(ptr_));
        }
    }

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& o) noexcept : count_(o.count_), ptr_(o.ptr_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }

    // Host → Device copy.  Async when stream != nullptr.
    void copyFromHost(const T* src, size_t count = 0, cudaStream_t s = nullptr) {
        size_t n = count ? count : count_;
        if (s)
            CUDA_CHECK(cudaMemcpyAsync(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice, s));
        else
            CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Device → Host copy.  Async when stream != nullptr.
    void copyToHost(T* dst, size_t count = 0, cudaStream_t s = nullptr) const {
        size_t n = count ? count : count_;
        if (s)
            CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, s));
        else
            CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero(cudaStream_t s = nullptr) {
        if (s)
            CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), s));
        else
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    T*       get()         { return ptr_; }
    const T* get()   const { return ptr_; }
    size_t   count() const { return count_; }
    size_t   bytes() const { return count_ * sizeof(T); }

private:
    size_t count_;
    T*     ptr_;
};

// =============================================================================
// Section 6: Launch Configuration
//
// Carry grid/block/smem/stream together so launch sites are self-documenting.
// =============================================================================

struct LaunchConfig {
    dim3         grid       = {1, 1, 1};
    dim3         block      = {256, 1, 1};
    size_t       shared_mem = 0;      // dynamic shared memory bytes
    cudaStream_t stream     = nullptr;

    void print(const char* label = "") const {
        LOG_INFO("LaunchConfig %-20s  grid=(%u,%u,%u)  block=(%u,%u,%u)  smem=%zu  stream=%p",
                 label,
                 grid.x, grid.y, grid.z,
                 block.x, block.y, block.z,
                 shared_mem, static_cast<void*>(stream));
    }
};

// 1-D element-wise kernel launch configuration.
inline LaunchConfig make1DConfig(int          n,
                                  int          block_size  = 256,
                                  size_t       shared_mem  = 0,
                                  cudaStream_t stream      = nullptr) {
    LaunchConfig cfg;
    cfg.block      = dim3(static_cast<unsigned>(block_size));
    cfg.grid       = dim3(static_cast<unsigned>((n + block_size - 1) / block_size));
    cfg.shared_mem = shared_mem;
    cfg.stream     = stream;
    return cfg;
}

// 2-D (nx × ny) kernel launch configuration.
// nx = width (maps to blockIdx.x), ny = height (maps to blockIdx.y).
inline LaunchConfig make2DConfig(int          nx,
                                  int          ny,
                                  int          bx         = 16,
                                  int          by         = 16,
                                  size_t       shared_mem = 0,
                                  cudaStream_t stream     = nullptr) {
    LaunchConfig cfg;
    cfg.block      = dim3(static_cast<unsigned>(bx), static_cast<unsigned>(by));
    cfg.grid       = dim3(static_cast<unsigned>((nx + bx - 1) / bx),
                          static_cast<unsigned>((ny + by - 1) / by));
    cfg.shared_mem = shared_mem;
    cfg.stream     = stream;
    return cfg;
}

// =============================================================================
// Section 7: RAII CUDA Timer (cudaEvent-based)
//
// Ad-hoc timing during kernel development.
// For structured regression benchmarking use benchmarkKernel() below.
// =============================================================================

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaTimer(const CudaTimer&)            = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // Returns elapsed milliseconds since the last start() call.
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// =============================================================================
// Section 8: Structured Benchmarking
//
// benchmarkKernel<F>(label, fn, flop_count, warmup, iters, stream)
//   fn        — zero-argument callable wrapping one kernel launch.
//   flop_count — total FP operations per single launch (0 = skip GFLOPs).
// =============================================================================

struct BenchmarkResult {
    float ms_avg = 0.0f;
    float ms_min = 0.0f;
    float ms_max = 0.0f;
    float gflops = 0.0f;
    int   iters  = 0;

    void print(const char* label) const {
        if (gflops > 0.0f) {
            LOG_INFO("Benchmark [%-28s]  avg=%7.3f ms  min=%7.3f ms  max=%7.3f ms  %7.2f GFLOPS  (n=%d)",
                     label, ms_avg, ms_min, ms_max, gflops, iters);
        } else {
            LOG_INFO("Benchmark [%-28s]  avg=%7.3f ms  min=%7.3f ms  max=%7.3f ms  (n=%d)",
                     label, ms_avg, ms_min, ms_max, iters);
        }
    }
};

template <typename F>
BenchmarkResult benchmarkKernel(const char*  label,
                                 F            kernel_fn,
                                 long long    flop_count = 0,
                                 int          warmup     = 5,
                                 int          iters      = 100,
                                 cudaStream_t stream     = nullptr) {
    LOG_DEBUG("Benchmarking '%s': warmup=%d iters=%d flops=%lld",
              label, warmup, iters, flop_count);

    for (int i = 0; i < warmup; ++i) kernel_fn();
    if (stream)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    else
        CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    float ms_min = std::numeric_limits<float>::max();
    float ms_max = 0.0f;
    float ms_sum = 0.0f;

    for (int i = 0; i < iters; ++i) {
        timer.start(stream);
        kernel_fn();
        float ms = timer.stop(stream);
        ms_sum += ms;
        ms_min  = std::min(ms_min, ms);
        ms_max  = std::max(ms_max, ms);
    }

    BenchmarkResult r;
    r.ms_avg = ms_sum / static_cast<float>(iters);
    r.ms_min = ms_min;
    r.ms_max = ms_max;
    r.iters  = iters;
    r.gflops = (flop_count > 0)
               ? static_cast<float>(flop_count) / 1e9f / (r.ms_avg / 1e3f)
               : 0.0f;
    r.print(label);
    return r;
}

// =============================================================================
// Section 9: Result Verification  (host-side, element-wise absolute error)
// =============================================================================

template <typename T>
bool verifyResult(const T* ref, const T* got, size_t n, float tol = 1e-4f) {
    float  max_err = 0.0f;
    size_t err_cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        float err = std::abs(static_cast<float>(ref[i]) - static_cast<float>(got[i]));
        if (err > tol) {
            ++err_cnt;
            if (err > max_err) max_err = err;
        }
    }
    if (err_cnt == 0) {
        LOG_INFO("Verification PASSED  (n=%zu, tol=%.1e)", n, static_cast<double>(tol));
        return true;
    }
    LOG_ERROR("Verification FAILED  %zu/%zu elements exceed tol=%.1e, max_err=%.6f",
              err_cnt, n, static_cast<double>(tol), max_err);
    return false;
}

// =============================================================================
// Section 10: Device Info Printer
// =============================================================================

inline void printDeviceInfo() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    // Memory bandwidth estimate: 2 * clock(kHz) * bus_width(bytes) / 1e6 → GB/s
    double bw_gbs = 2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6;
    LOG_INFO("GPU[%d]: %-30s  CC=%d.%d  SMs=%d  MaxThreads/SM=%d  "
             "SharedMem/SM=%zu KB  GlobalMem=%zu MB  PeakBW=%.0f GB/s",
             dev, p.name, p.major, p.minor,
             p.multiProcessorCount,
             p.maxThreadsPerMultiProcessor,
             p.sharedMemoryPerMultiprocessor / 1024,
             p.totalGlobalMem / (1024ULL * 1024ULL),
             bw_gbs);
}

} // namespace cuda_template
