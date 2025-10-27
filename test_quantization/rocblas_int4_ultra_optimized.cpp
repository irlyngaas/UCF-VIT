#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error " << error << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ROCBLAS_CHECK(cmd) \
    do { \
        rocblas_status status = cmd; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS error " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Ultra-optimized kernels for minimal overhead

// Kernel 1: Register-level unpacking with vectorized loads
__global__ void ultra_unpack_int4_vectorized(
    const uint4* __restrict__ packed_data,    // 128-bit vectorized loads
    int8_t* __restrict__ unpacked_data,
    int num_packs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_packs) {
        // Load 128 bits = 32 INT4 values at once
        uint4 packed_vec = packed_data[tid];
        
        // Extract each uint32_t and unpack in registers
        uint32_t p0 = packed_vec.x, p1 = packed_vec.y, p2 = packed_vec.z, p3 = packed_vec.w;
        
        int8_t* out_base = unpacked_data + tid * 32;
        
        // Unroll completely for max performance
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // p0: first 8 INT4 values
            uint8_t bits = (p0 >> (i * 4)) & 0xF;
            out_base[i] = (bits & 0x8) ? (bits | 0xF0) : bits;
            
            // p1: next 8 INT4 values  
            bits = (p1 >> (i * 4)) & 0xF;
            out_base[i + 8] = (bits & 0x8) ? (bits | 0xF0) : bits;
            
            // p2: next 8 INT4 values
            bits = (p2 >> (i * 4)) & 0xF;
            out_base[i + 16] = (bits & 0x8) ? (bits | 0xF0) : bits;
            
            // p3: last 8 INT4 values
            bits = (p3 >> (i * 4)) & 0xF;
            out_base[i + 24] = (bits & 0x8) ? (bits | 0xF0) : bits;
        }
    }
}

// Kernel 2: Fused unpack + copy with warp-level cooperation
__global__ void warp_cooperative_unpack(
    const uint32_t* __restrict__ packed_data,
    int8_t* __restrict__ unpacked_data,
    int rows, int packed_cols, int unpacked_cols
) {
    __shared__ uint32_t shared_packed[1024];  // 32KB shared memory
    __shared__ int8_t shared_unpacked[8192];  // Pre-unpacked data
    
    int row = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    
    if (row < rows) {
        // Phase 1: Coalesced load to shared memory
        int packs_per_warp = (packed_cols + warps_per_block - 1) / warps_per_block;
        int pack_start = warp_id * packs_per_warp;
        
        for (int i = 0; i < packs_per_warp; i += 32) {
            int pack_idx = pack_start + i + lane_id;
            if (pack_idx < packed_cols) {
                shared_packed[warp_id * packs_per_warp + i + lane_id] = 
                    packed_data[row * packed_cols + pack_idx];
            }
        }
        __syncthreads();
        
        // Phase 2: Warp-cooperative unpacking
        int unpack_start = warp_id * packs_per_warp * 8;
        for (int i = 0; i < packs_per_warp; i++) {
            if (pack_start + i < packed_cols) {
                uint32_t packed = shared_packed[warp_id * packs_per_warp + i];
                
                // Each lane unpacks different bits
                if (lane_id < 8) {
                    uint8_t bits = (packed >> (lane_id * 4)) & 0xF;
                    int8_t value = (bits & 0x8) ? (bits | 0xF0) : bits;
                    int out_idx = unpack_start + i * 8 + lane_id;
                    if (out_idx < unpacked_cols) {
                        shared_unpacked[out_idx] = value;
                    }
                }
            }
        }
        __syncthreads();
        
        // Phase 3: Coalesced write to global memory
        for (int i = threadIdx.x; i < unpacked_cols; i += blockDim.x) {
            unpacked_data[row * unpacked_cols + i] = shared_unpacked[i];
        }
    }
}

// Kernel 3: Template-based unpack with compile-time optimization
template<int TILE_M, int TILE_K>
__global__ void templated_unpack_int4(
    const uint32_t* __restrict__ packed_data,
    int8_t* __restrict__ unpacked_data,
    int M, int K, int packed_K
) {
    // Thread tile mapping
    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_K;
    int thread_row = tile_row + threadIdx.y;
    int thread_col_pack = (tile_col / 8) + threadIdx.x;
    
    // Registers for unpacked data
    int8_t reg_data[8];
    
    // Each thread handles one packed uint32 → 8 INT8 values
    if (thread_row < M && thread_col_pack < packed_K) {
        uint32_t packed = packed_data[thread_row * packed_K + thread_col_pack];
        
        // Template unrolling for optimal register usage
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t bits = (packed >> (i * 4)) & 0xF;
            reg_data[i] = (bits & 0x8) ? (bits | 0xF0) : bits;
        }
        
        // Write back with proper bounds checking
        int base_col = thread_col_pack * 8;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = base_col + i;
            if (col < K) {
                unpacked_data[thread_row * K + col] = reg_data[i];
            }
        }
    }
}

class UltraOptimizedInt4Gemm {
private:
    rocblas_handle handle;
    hipStream_t stream;
    
public:
    UltraOptimizedInt4Gemm() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
        HIP_CHECK(hipStreamCreate(&stream));
        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    }
    
    ~UltraOptimizedInt4Gemm() {
        rocblas_destroy_handle(handle);
        hipStreamDestroy(stream);
    }
    
    void pack_matrix_int4_cpu(const std::vector<int8_t>& src, int rows, int cols,
                              std::vector<uint32_t>& packed, int& packed_cols) {
        packed_cols = (cols + 7) / 8;
        packed.resize(rows * packed_cols);
        
        for (int r = 0; r < rows; r++) {
            for (int pc = 0; pc < packed_cols; pc++) {
                uint32_t pack_val = 0;
                for (int i = 0; i < 8; i++) {
                    int c = pc * 8 + i;
                    if (c < cols) {
                        int8_t val = src[r * cols + c];
                        val = std::max(-8, std::min(7, (int)val));  // Clamp to INT4
                        pack_val |= ((val & 0xF) << (i * 4));
                    }
                }
                packed[r * packed_cols + pc] = pack_val;
            }
        }
    }
    
    void test_ultra_optimized_gemm(int M, int N, int K) {
        std::cout << "\n=== ULTRA-OPTIMIZED INT4 GEMM: " << M << "x" << K << " @ " << K << "x" << N << " ===" << std::endl;
        
        // Generate test data
        std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dis(-8, 7);
        
        std::vector<int8_t> h_A_int4(M * K), h_B_int4(K * N);
        for (auto& val : h_A_int4) val = dis(gen);
        for (auto& val : h_B_int4) val = dis(gen);
        
        // Pack matrices
        std::vector<uint32_t> h_A_packed, h_B_packed;
        int A_packed_cols, B_packed_cols;
        pack_matrix_int4_cpu(h_A_int4, M, K, h_A_packed, A_packed_cols);
        pack_matrix_int4_cpu(h_B_int4, K, N, h_B_packed, B_packed_cols);
        
        // Allocate device memory
        uint32_t *d_A_packed, *d_B_packed;
        int8_t *d_A_unpacked, *d_B_unpacked;
        int32_t *d_C;
        
        HIP_CHECK(hipMalloc(&d_A_packed, h_A_packed.size() * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_B_packed, h_B_packed.size() * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_A_unpacked, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B_unpacked, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(int32_t)));
        
        // Copy data asynchronously
        HIP_CHECK(hipMemcpyAsync(d_A_packed, h_A_packed.data(), 
                   h_A_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_B_packed, h_B_packed.data(), 
                   h_B_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice, stream));
        
        const int num_runs = 100;
        int32_t alpha = 1, beta = 0;
        
        // Test different optimization strategies
        std::vector<std::string> kernel_names = {
            "Vectorized Register Unpack",
            "Warp Cooperative Unpack", 
            "Template Optimized Unpack"
        };
        
        std::vector<double> kernel_times(3);
        
        // Kernel 1: Vectorized register unpacking
        {
            int block_size = 256;
            int grid_size_A = (h_A_packed.size() / 4 + block_size - 1) / block_size;
            int grid_size_B = (h_B_packed.size() / 4 + block_size - 1) / block_size;
            
            // Warmup
            for (int i = 0; i < 10; i++) {
                ultra_unpack_int4_vectorized<<<grid_size_A, block_size, 0, stream>>>(
                    (uint4*)d_A_packed, d_A_unpacked, h_A_packed.size() / 4);
                ultra_unpack_int4_vectorized<<<grid_size_B, block_size, 0, stream>>>(
                    (uint4*)d_B_packed, d_B_unpacked, h_B_packed.size() / 4);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_runs; i++) {
                ultra_unpack_int4_vectorized<<<grid_size_A, block_size, 0, stream>>>(
                    (uint4*)d_A_packed, d_A_unpacked, h_A_packed.size() / 4);
                ultra_unpack_int4_vectorized<<<grid_size_B, block_size, 0, stream>>>(
                    (uint4*)d_B_packed, d_B_unpacked, h_B_packed.size() / 4);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            kernel_times[0] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / num_runs;
        }
        
        // Kernel 2: Warp cooperative
        {
            dim3 block(256);
            dim3 grid((K + 2047) / 2048, M);
            int shared_mem = 32 * 1024;  // 32KB
            
            // Warmup
            for (int i = 0; i < 10; i++) {
                warp_cooperative_unpack<<<grid, block, shared_mem, stream>>>(
                    d_A_packed, d_A_unpacked, M, A_packed_cols, K);
                warp_cooperative_unpack<<<dim3((N + 2047) / 2048, K), block, shared_mem, stream>>>(
                    d_B_packed, d_B_unpacked, K, B_packed_cols, N);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_runs; i++) {
                warp_cooperative_unpack<<<grid, block, shared_mem, stream>>>(
                    d_A_packed, d_A_unpacked, M, A_packed_cols, K);
                warp_cooperative_unpack<<<dim3((N + 2047) / 2048, K), block, shared_mem, stream>>>(
                    d_B_packed, d_B_unpacked, K, B_packed_cols, N);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            kernel_times[1] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / num_runs;
        }
        
        // Kernel 3: Template optimized
        {
            dim3 block(32, 8);  // 256 threads
            dim3 grid_A((A_packed_cols + 4 - 1) / 4, (M + 8 - 1) / 8);
            dim3 grid_B((B_packed_cols + 4 - 1) / 4, (K + 8 - 1) / 8);
            
            // Warmup
            for (int i = 0; i < 10; i++) {
                templated_unpack_int4<8, 32><<<grid_A, block, 0, stream>>>(
                    d_A_packed, d_A_unpacked, M, K, A_packed_cols);
                templated_unpack_int4<8, 32><<<grid_B, block, 0, stream>>>(
                    d_B_packed, d_B_unpacked, K, N, B_packed_cols);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_runs; i++) {
                templated_unpack_int4<8, 32><<<grid_A, block, 0, stream>>>(
                    d_A_packed, d_A_unpacked, M, K, A_packed_cols);
                templated_unpack_int4<8, 32><<<grid_B, block, 0, stream>>>(
                    d_B_packed, d_B_unpacked, K, N, B_packed_cols);
                    
                ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    M, N, K, &alpha,
                    d_A_unpacked, rocblas_datatype_i8_r, M,
                    d_B_unpacked, rocblas_datatype_i8_r, K, &beta,
                    d_C, rocblas_datatype_i32_r, M,
                    d_C, rocblas_datatype_i32_r, M,
                    rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            auto end = std::chrono::high_resolution_clock::now();
            kernel_times[2] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / num_runs;
        }
        
        // Baseline INT8 test
        std::vector<int8_t> h_A_int8(h_A_int4), h_B_int8(h_B_int4);
        int8_t *d_A_int8, *d_B_int8;
        
        HIP_CHECK(hipMalloc(&d_A_int8, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B_int8, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMemcpyAsync(d_A_int8, h_A_int8.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_B_int8, h_B_int8.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice, stream));
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K, &alpha,
                d_A_int8, rocblas_datatype_i8_r, M,
                d_B_int8, rocblas_datatype_i8_r, K, &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K, &alpha,
                d_A_int8, rocblas_datatype_i8_r, M,
                d_B_int8, rocblas_datatype_i8_r, K, &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        double int8_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / num_runs;
        
        // Results
        std::cout << "\n--- ULTRA-OPTIMIZATION RESULTS ---" << std::endl;
        std::cout << "INT8 baseline: " << int8_time << " ms (" << (2.0 * M * N * K) / (int8_time * 1e-3) / 1e9 << " GFLOPS)" << std::endl;
        
        for (int i = 0; i < 3; i++) {
            double overhead = (kernel_times[i] - int8_time) / int8_time * 100;
            double gflops = (2.0 * M * N * K) / (kernel_times[i] * 1e-3) / 1e9;
            std::cout << kernel_names[i] << ": " << kernel_times[i] << " ms (" << gflops << " GFLOPS, " 
                      << overhead << "% overhead)" << std::endl;
        }
        
        // Find best kernel
        int best_kernel = std::min_element(kernel_times.begin(), kernel_times.end()) - kernel_times.begin();
        double best_overhead = (kernel_times[best_kernel] - int8_time) / int8_time * 100;
        
        std::cout << "\nBest kernel: " << kernel_names[best_kernel] << std::endl;
        std::cout << "Target: <10% overhead | Achieved: " << best_overhead << "%" << std::endl;
        std::cout << "Status: " << (best_overhead < 10 ? "SUCCESS ✅" : (best_overhead < 20 ? "CLOSE ⚠️" : "NEEDS MORE WORK ❌")) << std::endl;
        
        // Cleanup
        hipFree(d_A_packed);
        hipFree(d_B_packed);
        hipFree(d_A_unpacked);
        hipFree(d_B_unpacked);
        hipFree(d_A_int8);
        hipFree(d_B_int8);
        hipFree(d_C);
    }
};

int main() {
    std::cout << "=== ULTRA-OPTIMIZED INT4 GEMM TEST ===" << std::endl;
    std::cout << "Target: Reduce overhead from 30% to <10%" << std::endl;
    std::cout << "Techniques: Vectorized loads, warp cooperation, template optimization" << std::endl;
    
    HIP_CHECK(hipInit(0));
    
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "\nFound " << device_count << " HIP devices" << std::endl;
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
    std::cout << "Max Shared Memory: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    
    try {
        UltraOptimizedInt4Gemm tester;
        
        // Test on the largest size where we had 30% overhead
        tester.test_ultra_optimized_gemm(1024, 3072, 768);
        
        std::cout << "\n=== ULTRA-OPTIMIZATION TEST COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}