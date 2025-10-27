#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

// Error checking macros
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

// GPU kernel for unpacking INT4 to INT8
__global__ void unpack_int4_to_int8_kernel(
    const uint32_t* __restrict__ packed_data,
    int8_t* __restrict__ unpacked_data,
    int num_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pack_idx = tid;
    
    // Each thread handles one packed uint32 (8 INT4 values)
    if (pack_idx < (num_elements + 7) / 8) {
        uint32_t packed = packed_data[pack_idx];
        int base_idx = pack_idx * 8;
        
        // Unpack 8 INT4 values
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int out_idx = base_idx + i;
            if (out_idx < num_elements) {
                // Extract 4 bits
                uint8_t bits = (packed >> (i * 4)) & 0xF;
                
                // Sign extend 4-bit to 8-bit
                int8_t value = (bits & 0x8) ? (bits | 0xF0) : bits;
                unpacked_data[out_idx] = value;
            }
        }
    }
}

// Optimized kernel using shared memory for coalesced access
__global__ void unpack_int4_matrix_optimized(
    const uint32_t* __restrict__ packed_data,
    int8_t* __restrict__ unpacked_data,
    int rows, int cols, int packed_cols
) {
    extern __shared__ uint32_t shared_packed[];
    
    int row = blockIdx.y;
    int col_block = blockIdx.x;
    int tid = threadIdx.x;
    
    const int BLOCK_SIZE = 256;
    const int PACKS_PER_BLOCK = BLOCK_SIZE / 8;  // Each pack produces 8 values
    
    if (row < rows && col_block * BLOCK_SIZE < cols) {
        // Load packed data to shared memory
        int pack_start = col_block * PACKS_PER_BLOCK;
        int pack_offset = row * packed_cols + pack_start;
        
        if (tid < PACKS_PER_BLOCK && pack_start + tid < packed_cols) {
            shared_packed[tid] = packed_data[pack_offset + tid];
        }
        __syncthreads();
        
        // Each warp handles unpacking
        if (tid < PACKS_PER_BLOCK) {
            uint32_t packed = shared_packed[tid];
            int base_col = col_block * BLOCK_SIZE + tid * 8;
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col = base_col + i;
                if (col < cols) {
                    // Extract and sign-extend
                    uint8_t bits = (packed >> (i * 4)) & 0xF;
                    int8_t value = (bits & 0x8) ? (bits | 0xF0) : bits;
                    unpacked_data[row * cols + col] = value;
                }
            }
        }
    }
}

class GpuInt4Gemm {
private:
    rocblas_handle handle;
    
public:
    GpuInt4Gemm() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }
    
    ~GpuInt4Gemm() {
        rocblas_destroy_handle(handle);
    }
    
    // Pack INT4 matrix on CPU (for testing)
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
                        // Clamp to INT4 range
                        val = (val < -8) ? -8 : (val > 7) ? 7 : val;
                        pack_val |= ((val & 0xF) << (i * 4));
                    }
                }
                packed[r * packed_cols + pc] = pack_val;
            }
        }
    }
    
    void test_gpu_unpack_gemm(int M, int N, int K) {
        std::cout << "\n=== Testing GPU-Unpacked INT4 GEMM: " << M << "x" << K << " @ " << K << "x" << N << " ===" << std::endl;
        
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
        
        // Copy packed data to device
        HIP_CHECK(hipMemcpy(d_A_packed, h_A_packed.data(), h_A_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B_packed, h_B_packed.data(), h_B_packed.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
        
        // Configure kernels
        dim3 blockDim(256);
        dim3 gridDim_A((M * A_packed_cols + blockDim.x - 1) / blockDim.x);
        dim3 gridDim_B((K * B_packed_cols + blockDim.x - 1) / blockDim.x);
        
        // Optimized kernel configuration
        dim3 blockDim_opt(32);
        dim3 gridDim_A_opt((K + 255) / 256, M);
        dim3 gridDim_B_opt((N + 255) / 256, K);
        
        int32_t alpha = 1, beta = 0;
        const int num_runs = 100;
        
        // Test 1: Basic kernel
        std::cout << "Testing basic GPU unpacking kernel..." << std::endl;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            unpack_int4_to_int8_kernel<<<gridDim_A, blockDim>>>(d_A_packed, d_A_unpacked, M * K);
            unpack_int4_to_int8_kernel<<<gridDim_B, blockDim>>>(d_B_packed, d_B_unpacked, K * N);
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            unpack_int4_to_int8_kernel<<<gridDim_A, blockDim>>>(d_A_packed, d_A_unpacked, M * K);
            unpack_int4_to_int8_kernel<<<gridDim_B, blockDim>>>(d_B_packed, d_B_unpacked, K * N);
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double basic_time_ms = duration.count() / 1000.0 / num_runs;
        
        // Test 2: Optimized kernel
        std::cout << "Testing optimized GPU unpacking kernel..." << std::endl;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            int shared_size = 32 * sizeof(uint32_t);
            unpack_int4_matrix_optimized<<<gridDim_A_opt, blockDim_opt, shared_size>>>(
                d_A_packed, d_A_unpacked, M, K, A_packed_cols);
            unpack_int4_matrix_optimized<<<gridDim_B_opt, blockDim_opt, shared_size>>>(
                d_B_packed, d_B_unpacked, K, N, B_packed_cols);
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            int shared_size = 32 * sizeof(uint32_t);
            unpack_int4_matrix_optimized<<<gridDim_A_opt, blockDim_opt, shared_size>>>(
                d_A_packed, d_A_unpacked, M, K, A_packed_cols);
            unpack_int4_matrix_optimized<<<gridDim_B_opt, blockDim_opt, shared_size>>>(
                d_B_packed, d_B_unpacked, K, N, B_packed_cols);
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double opt_time_ms = duration.count() / 1000.0 / num_runs;
        
        // Compare with pure INT8
        std::cout << "Running pure INT8 baseline..." << std::endl;
        
        // Convert to INT8 directly for baseline
        std::vector<int8_t> h_A_int8(h_A_int4), h_B_int8(h_B_int4);
        int8_t *d_A_int8, *d_B_int8;
        
        HIP_CHECK(hipMalloc(&d_A_int8, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B_int8, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMemcpy(d_A_int8, h_A_int8.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B_int8, h_B_int8.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice));
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_int8, rocblas_datatype_i8_r, M,
                d_B_int8, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_int8, rocblas_datatype_i8_r, M,
                d_B_int8, rocblas_datatype_i8_r, K,
                &beta,
                d_C, rocblas_datatype_i32_r, M,
                d_C, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double int8_time_ms = duration.count() / 1000.0 / num_runs;
        
        // Calculate metrics
        double basic_gflops = (2.0 * M * N * K) / (basic_time_ms * 1e-3) / 1e9;
        double opt_gflops = (2.0 * M * N * K) / (opt_time_ms * 1e-3) / 1e9;
        double int8_gflops = (2.0 * M * N * K) / (int8_time_ms * 1e-3) / 1e9;
        
        double basic_overhead = (basic_time_ms - int8_time_ms) / int8_time_ms * 100;
        double opt_overhead = (opt_time_ms - int8_time_ms) / int8_time_ms * 100;
        
        // Results
        std::cout << "\n--- RESULTS ---" << std::endl;
        std::cout << "Pure INT8 baseline:" << std::endl;
        std::cout << "  Time: " << int8_time_ms << " ms" << std::endl;
        std::cout << "  GFLOPS: " << int8_gflops << std::endl;
        
        std::cout << "\nINT4 with basic GPU unpacking:" << std::endl;
        std::cout << "  Time: " << basic_time_ms << " ms" << std::endl;
        std::cout << "  GFLOPS: " << basic_gflops << std::endl;
        std::cout << "  Overhead vs INT8: " << basic_overhead << "%" << std::endl;
        
        std::cout << "\nINT4 with optimized GPU unpacking:" << std::endl;
        std::cout << "  Time: " << opt_time_ms << " ms" << std::endl;
        std::cout << "  GFLOPS: " << opt_gflops << std::endl;
        std::cout << "  Overhead vs INT8: " << opt_overhead << "%" << std::endl;
        
        std::cout << "\nMemory savings:" << std::endl;
        std::cout << "  INT4 vs INT8: 2x reduction" << std::endl;
        std::cout << "  INT4 vs FP32: 8x reduction" << std::endl;
        
        std::cout << "\nStatus: " << (opt_overhead < 20 ? "SUCCESS ✅" : "NEEDS OPTIMIZATION ⚠️") << std::endl;
        
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
    std::cout << "=== GPU-OPTIMIZED INT4 UNPACKING TEST ===" << std::endl;
    std::cout << "Testing INT4 with GPU-side unpacking to INT8" << std::endl;
    
    // Initialize HIP
    HIP_CHECK(hipInit(0));
    
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP devices" << std::endl;
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
    std::cout << "Compute Units: " << prop.multiProcessorCount << std::endl;
    
    try {
        GpuInt4Gemm tester;
        
        // Test different matrix sizes
        tester.test_gpu_unpack_gemm(256, 3072, 768);   // Small
        tester.test_gpu_unpack_gemm(512, 3072, 768);   // Medium
        tester.test_gpu_unpack_gemm(1024, 3072, 768);  // Large
        
        std::cout << "\n=== GPU INT4 UNPACKING TESTS COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}