// =============================================================================
// cuda_kernel_template.cu
// Two example kernels wired to the full infrastructure in cuda_kernel_template.cuh.
//
//   Kernel 1: scaleBias    — elementwise  out[i] = in[i]*scale + bias  (1D)
//   Kernel 2: tiledMatMul  — tiled SGEMM  C = A(M×K) × B(K×N)          (2D)
//
// Compile:
//   # Release
//   nvcc -O3 -arch=sm_89 -o kernel_template cuda_kernel_template.cu
//
//   # Debug  (device printf + host verify, verbose logging)
//   nvcc -O0 -g -G -DCUDA_DEBUG -DCUDA_LOG_LEVEL=3 \
//        -o kernel_template cuda_kernel_template.cu
//
//   # Profile  (NVTX ranges visible in Nsight Systems)
//   nvcc -O3 -arch=sm_89 -DCUDA_PROFILE \
//        -o kernel_template cuda_kernel_template.cu -lnvtx3
//
// Run:
//   ./kernel_template
// =============================================================================

#include "cuda_kernel_template.cuh"
#include <cstring>
#include <numeric>
#include <vector>

using namespace cuda_template;

// =============================================================================
// Kernel 1: scaleBias  (elementwise, 1-D)
// out[i] = in[i] * scale + bias
// =============================================================================

__global__ void scaleBiasKernel(const float* __restrict__ in,
                                 float* __restrict__       out,
                                 float                     scale,
                                 float                     bias,
                                 int                       n) {
    // Prints once from the very first thread — use to verify launch parameters.
    KERNEL_LOG("scaleBiasKernel  n=%d  scale=%.3f  bias=%.3f", n, scale, bias);

    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * scale + bias;
    }
}

// CPU reference implementation for correctness checking.
static void scaleBiasCPU(const float* in, float* out,
                          float scale, float bias, int n) {
    for (int i = 0; i < n; ++i) out[i] = in[i] * scale + bias;
}

// Host launch wrapper — insulates main() from kernel call syntax.
static void launchScaleBias(const float* d_in, float* d_out,
                             float scale, float bias, int n,
                             const LaunchConfig& cfg) {
    NVTX_PUSH("scaleBias");
    CUDA_LAUNCH(scaleBiasKernel, cfg, d_in, d_out, scale, bias, n);
    NVTX_POP();
}

// =============================================================================
// Kernel 2: tiledMatMul  (2-D, shared memory)
// C(M×N) = A(M×K) × B(K×N)  using TILE×TILE shared-memory tiles.
//
// Note: shared_mem in LaunchConfig should be 0 here because sA/sB are
//       statically declared (__shared__ float), not dynamic (extern __shared__).
// =============================================================================

static constexpr int TILE = 16;

__global__ void tiledMatMulKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__       C,
                                  int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = static_cast<int>(blockIdx.y) * TILE + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * TILE + threadIdx.x;

    KERNEL_LOG("tiledMatMulKernel  M=%d  N=%d  K=%d  tiles=%d",
               M, N, K, (K + TILE - 1) / TILE);

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Cooperative load: each thread loads one element of the tile.
        sA[threadIdx.y][threadIdx.x] = (row < M && t * TILE + threadIdx.x < K)
                                        ? A[row * K + t * TILE + threadIdx.x]
                                        : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE + threadIdx.y < K && col < N)
                                        ? B[(t * TILE + threadIdx.y) * N + col]
                                        : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// CPU reference — ikj loop order for cache-friendly B access.
static void matMulCPU(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

static void launchTiledMatMul(const float* d_A, const float* d_B, float* d_C,
                               int M, int N, int K, const LaunchConfig& cfg) {
    NVTX_PUSH("tiledMatMul");
    CUDA_LAUNCH(tiledMatMulKernel, cfg, d_A, d_B, d_C, M, N, K);
    NVTX_POP();
}

// =============================================================================
// Main: wire up both kernels with the full infrastructure
// =============================================================================

int main() {
    try {
        NVTX_PUSH("main");
        printDeviceInfo();

        // ----------------------------------------------------------------
        // Kernel 1: scaleBias
        // ----------------------------------------------------------------
        LOG_INFO("===== Kernel 1: scaleBias =====");
        {
            constexpr int   N     = 1 << 22;  // 4 M elements
            constexpr float SCALE = 2.5f;
            constexpr float BIAS  = -1.0f;

            // Host data
            std::vector<float> h_in(N), h_ref(N), h_out(N);
            std::iota(h_in.begin(), h_in.end(), 0.0f);  // 0, 1, 2, ...

            // Device allocation (RAII — freed automatically at scope exit)
            CudaBuffer<float> d_in(N), d_out(N);
            d_in.copyFromHost(h_in.data());
            d_out.zero();

            // Build and inspect launch config
            LaunchConfig cfg = make1DConfig(N);
            cfg.print("scaleBias");

            // Single correctness run
            launchScaleBias(d_in.get(), d_out.get(), SCALE, BIAS, N, cfg);
            CUDA_CHECK(cudaDeviceSynchronize());
            d_out.copyToHost(h_out.data());

            scaleBiasCPU(h_in.data(), h_ref.data(), SCALE, BIAS, N);
            verifyResult(h_ref.data(), h_out.data(), N, 1e-4f);

            // Structured benchmark: 1 mul + 1 add per element → 2N FLOPs
            benchmarkKernel(
                "scaleBias (4M elems)",
                [&]() {
                    launchScaleBias(d_in.get(), d_out.get(), SCALE, BIAS, N, cfg);
                },
                2LL * N,
                /*warmup=*/10, /*iters=*/200);
        }

        // ----------------------------------------------------------------
        // Kernel 2: tiledMatMul
        // ----------------------------------------------------------------
        LOG_INFO("===== Kernel 2: tiledMatMul =====");
        {
            constexpr int M = 512, N = 512, K = 256;

            // Host data
            std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N), h_C(M * N);
            for (auto& v : h_A) v = static_cast<float>(rand()) / RAND_MAX;
            for (auto& v : h_B) v = static_cast<float>(rand()) / RAND_MAX;

            // Device allocation
            CudaBuffer<float> d_A(M * K), d_B(K * N), d_C(M * N);
            d_A.copyFromHost(h_A.data());
            d_B.copyFromHost(h_B.data());
            d_C.zero();

            // make2DConfig(nx=N, ny=M):  blockIdx.x → column tiles, blockIdx.y → row tiles
            // shared_mem=0: sA/sB are statically declared inside the kernel.
            LaunchConfig cfg = make2DConfig(N, M, TILE, TILE, /*shared_mem=*/0);
            cfg.print("tiledMatMul");

            // Single correctness run
            launchTiledMatMul(d_A.get(), d_B.get(), d_C.get(), M, N, K, cfg);
            CUDA_CHECK(cudaDeviceSynchronize());
            d_C.copyToHost(h_C.data());

            matMulCPU(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
            // fp32 accumulation over K=256 terms → use slightly relaxed tolerance
            verifyResult(h_C_ref.data(), h_C.data(), static_cast<size_t>(M) * N, 1e-3f);

            // 2MNK FLOPs (one fma per inner-loop iteration)
            benchmarkKernel(
                "tiledMatMul (512x512x256)",
                [&]() {
                    launchTiledMatMul(d_A.get(), d_B.get(), d_C.get(), M, N, K, cfg);
                },
                2LL * M * N * K,
                /*warmup=*/10, /*iters=*/200);
        }

        NVTX_POP();  // main
        LOG_INFO("Done.");
        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("%s", e.what());
        return 1;
    }
}
