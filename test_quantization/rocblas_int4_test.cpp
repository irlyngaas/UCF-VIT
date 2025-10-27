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

class Int4PackedGemm {
private:
    rocblas_handle handle;
    
public:
    Int4PackedGemm() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }
    
    ~Int4PackedGemm() {
        rocblas_destroy_handle(handle);
    }
    
    // Pack 8 INT4 values into one 32-bit integer
    __device__ __host__ uint32_t pack_int4x8(const int8_t* values) {
        uint32_t packed = 0;
        for (int i = 0; i < 8; i++) {
            // Clamp to 4-bit range [-8, 7]
            int8_t val = values[i];
            val = (val < -8) ? -8 : (val > 7) ? 7 : val;
            
            // Pack into 4 bits (signed, so we need to handle sign extension)
            uint32_t bits = (val & 0xF);
            packed |= (bits << (i * 4));
        }
        return packed;
    }
    
    // Unpack one 32-bit integer into 8 INT4 values (extended to INT8)
    __device__ __host__ void unpack_int4x8(uint32_t packed, int8_t* values) {
        for (int i = 0; i < 8; i++) {
            // Extract 4 bits
            uint8_t bits = (packed >> (i * 4)) & 0xF;
            
            // Sign extend 4-bit to 8-bit
            if (bits & 0x8) {
                values[i] = bits | 0xF0;  // Extend sign bit
            } else {
                values[i] = bits;
            }
        }
    }
    
    // Convert INT4 matrix to packed format
    void pack_matrix_int4(const std::vector<int8_t>& src, int rows, int cols, 
                         std::vector<uint32_t>& packed, int& packed_cols) {
        // Columns must be multiple of 8 for packing
        packed_cols = (cols + 7) / 8;
        packed.resize(rows * packed_cols);
        
        for (int r = 0; r < rows; r++) {
            for (int pc = 0; pc < packed_cols; pc++) {
                int8_t values[8] = {0};
                
                // Extract 8 values for packing
                for (int i = 0; i < 8; i++) {
                    int c = pc * 8 + i;
                    if (c < cols) {
                        values[i] = src[r * cols + c];
                    }
                }
                
                packed[r * packed_cols + pc] = pack_int4x8(values);
            }
        }
    }
    
    // Unpack and convert to INT8 for rocBLAS
    void unpack_to_int8(const std::vector<uint32_t>& packed, int rows, int packed_cols,
                       std::vector<int8_t>& unpacked, int target_cols) {
        unpacked.resize(rows * target_cols);
        
        for (int r = 0; r < rows; r++) {
            for (int pc = 0; pc < packed_cols; pc++) {
                int8_t values[8];
                unpack_int4x8(packed[r * packed_cols + pc], values);
                
                for (int i = 0; i < 8 && pc * 8 + i < target_cols; i++) {
                    unpacked[r * target_cols + (pc * 8 + i)] = values[i];
                }
            }
        }
    }
    
    void test_int4_gemm(int M, int N, int K) {
        std::cout << "\n=== Testing INT4 Packed GEMM: " << M << "x" << K << " @ " << K << "x" << N << " ===" << std::endl;
        
        // Generate INT4 data (-8 to 7)
        std::random_device rd;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dis(-8, 7);
        
        std::vector<int8_t> h_A_int4(M * K), h_B_int4(K * N);
        for (auto& val : h_A_int4) val = dis(gen);
        for (auto& val : h_B_int4) val = dis(gen);
        
        // === Test 1: Baseline INT8 (for comparison) ===
        std::cout << "Running INT8 baseline..." << std::endl;
        
        std::vector<int8_t> h_A_int8(h_A_int4), h_B_int8(h_B_int4);
        
        int8_t *d_A_int8, *d_B_int8;
        int32_t *d_C_int8;
        
        HIP_CHECK(hipMalloc(&d_A_int8, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B_int8, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_C_int8, M * N * sizeof(int32_t)));
        
        HIP_CHECK(hipMemcpy(d_A_int8, h_A_int8.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B_int8, h_B_int8.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice));
        
        int32_t alpha = 1, beta = 0;
        
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
                d_C_int8, rocblas_datatype_i32_r, M,
                d_C_int8, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        const int num_runs = 100;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_int8, rocblas_datatype_i8_r, M,
                d_B_int8, rocblas_datatype_i8_r, K,
                &beta,
                d_C_int8, rocblas_datatype_i32_r, M,
                d_C_int8, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double int8_time_ms = duration.count() / 1000.0 / num_runs;
        double int8_gflops = (2.0 * M * N * K) / (int8_time_ms * 1e-3) / 1e9;
        
        std::cout << "  INT8 time: " << int8_time_ms << " ms" << std::endl;
        std::cout << "  INT8 GFLOPS: " << int8_gflops << std::endl;
        
        // === Test 2: INT4 Packed Implementation ===
        std::cout << "Running INT4 packed implementation..." << std::endl;
        
        // Pack matrices
        std::vector<uint32_t> h_A_packed, h_B_packed;
        int A_packed_cols, B_packed_cols;
        
        pack_matrix_int4(h_A_int4, M, K, h_A_packed, A_packed_cols);
        pack_matrix_int4(h_B_int4, K, N, h_B_packed, B_packed_cols);
        
        // Unpack to INT8 for MFMA (this would be done on GPU in real implementation)
        std::vector<int8_t> h_A_unpacked, h_B_unpacked;
        unpack_to_int8(h_A_packed, M, A_packed_cols, h_A_unpacked, K);
        unpack_to_int8(h_B_packed, K, B_packed_cols, h_B_unpacked, N);
        
        // For now, let's time the unpacking + GEMM (simulating GPU kernel)
        int8_t *d_A_unpacked, *d_B_unpacked;
        int32_t *d_C_int4;
        
        HIP_CHECK(hipMalloc(&d_A_unpacked, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B_unpacked, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_C_int4, M * N * sizeof(int32_t)));
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            // Simulate: copy packed data + unpack + GEMM
            HIP_CHECK(hipMemcpy(d_A_unpacked, h_A_unpacked.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_B_unpacked, h_B_unpacked.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice));
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C_int4, rocblas_datatype_i32_r, M,
                d_C_int4, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing (including unpack overhead for now)
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            // This simulates the full INT4 pipeline
            HIP_CHECK(hipMemcpy(d_A_unpacked, h_A_unpacked.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_B_unpacked, h_B_unpacked.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice));
            
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A_unpacked, rocblas_datatype_i8_r, M,
                d_B_unpacked, rocblas_datatype_i8_r, K,
                &beta,
                d_C_int4, rocblas_datatype_i32_r, M,
                d_C_int4, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double int4_time_ms = duration.count() / 1000.0 / num_runs;
        double int4_gflops = (2.0 * M * N * K) / (int4_time_ms * 1e-3) / 1e9;
        
        std::cout << "  INT4 time: " << int4_time_ms << " ms" << std::endl;
        std::cout << "  INT4 GFLOPS: " << int4_gflops << std::endl;
        
        // Results
        double speedup = int8_time_ms / int4_time_ms;
        double memory_reduction = 2.0; // INT4 vs INT8
        double memory_reduction_vs_fp32 = 8.0; // INT4 vs FP32
        
        std::cout << "\n--- RESULTS ---" << std::endl;
        std::cout << "  INT4 vs INT8 speedup: " << speedup << "x" << std::endl;
        std::cout << "  Memory reduction vs INT8: " << memory_reduction << "x" << std::endl;
        std::cout << "  Memory reduction vs FP32: " << memory_reduction_vs_fp32 << "x" << std::endl;
        std::cout << "  Status: " << (speedup > 0.8 ? "PROMISING ✅" : "NEEDS WORK ⚠️") << std::endl;
        
        // Accuracy check
        std::vector<int32_t> result_int8(M * N), result_int4(M * N);
        HIP_CHECK(hipMemcpy(result_int8.data(), d_C_int8, M * N * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(result_int4.data(), d_C_int4, M * N * sizeof(int32_t), hipMemcpyDeviceToHost));
        
        // Check first few values
        int max_diff = 0;
        for (int i = 0; i < std::min(100, M * N); i++) {
            int diff = abs(result_int8[i] - result_int4[i]);
            max_diff = std::max(max_diff, diff);
        }
        std::cout << "  Max difference (first 100 elements): " << max_diff << std::endl;
        
        // Cleanup
        hipFree(d_A_int8);
        hipFree(d_B_int8);
        hipFree(d_C_int8);
        hipFree(d_A_unpacked);
        hipFree(d_B_unpacked);
        hipFree(d_C_int4);
    }
};

int main() {
    std::cout << "=== INT4 PACKED GEMM TEST ===" << std::endl;
    std::cout << "Testing INT4 using packed format + MFMA INT8" << std::endl;
    
    // Initialize HIP
    HIP_CHECK(hipInit(0));
    
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " HIP devices" << std::endl;
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
    
    try {
        Int4PackedGemm tester;
        
        // Test different matrix sizes
        tester.test_int4_gemm(256, 3072, 768);   // Small
        tester.test_int4_gemm(512, 3072, 768);   // Medium
        tester.test_int4_gemm(1024, 3072, 768);  // Large
        
        std::cout << "\n=== INT4 PACKED TESTS COMPLETED ===" << std::endl;
        std::cout << "Note: This is proof-of-concept. Real implementation would" << std::endl;
        std::cout << "do unpacking on GPU to avoid memory transfer overhead." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}