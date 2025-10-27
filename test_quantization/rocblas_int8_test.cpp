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

class RocBlasInt8Tester {
private:
    rocblas_handle handle;
    
public:
    RocBlasInt8Tester() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle));
    }
    
    ~RocBlasInt8Tester() {
        rocblas_destroy_handle(handle);
    }
    
    void test_int8_gemm(int M, int N, int K) {
        std::cout << "\n=== Testing INT8 GEMM: " << M << "x" << K << " @ " << K << "x" << N << " ===" << std::endl;
        
        // Host data
        std::vector<int8_t> h_A(M * K);
        std::vector<int8_t> h_B(K * N);
        std::vector<float> h_C_fp32(M * N);
        std::vector<int32_t> h_C_int32(M * N);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dis(-127, 127);
        
        for (auto& val : h_A) val = static_cast<int8_t>(dis(gen));
        for (auto& val : h_B) val = static_cast<int8_t>(dis(gen));
        
        // Device memory
        int8_t *d_A, *d_B;
        int32_t *d_C_int32;
        float *d_C_fp32;
        
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(int8_t)));
        HIP_CHECK(hipMalloc(&d_C_int32, M * N * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&d_C_fp32, M * N * sizeof(float)));
        
        // Copy to device
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), hipMemcpyHostToDevice));
        
        // Test 1: INT8 GEMM with INT32 output
        std::cout << "Testing INT8 -> INT32 GEMM..." << std::endl;
        
        int32_t alpha = 1, beta = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A, rocblas_datatype_i8_r, M,
                d_B, rocblas_datatype_i8_r, K,
                &beta,
                d_C_int32, rocblas_datatype_i32_r, M,
                d_C_int32, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        const int num_runs = 100;
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha,
                d_A, rocblas_datatype_i8_r, M,
                d_B, rocblas_datatype_i8_r, K,
                &beta,
                d_C_int32, rocblas_datatype_i32_r, M,
                d_C_int32, rocblas_datatype_i32_r, M,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double int8_time_ms = duration.count() / 1000.0 / num_runs;
        
        // Calculate GFLOPS
        double gflops = (2.0 * M * N * K) / (int8_time_ms * 1e-3) / 1e9;
        
        std::cout << "  INT8 GEMM time: " << int8_time_ms << " ms" << std::endl;
        std::cout << "  INT8 GEMM GFLOPS: " << gflops << std::endl;
        
        // Test 2: FP32 baseline for comparison
        std::cout << "Testing FP32 baseline..." << std::endl;
        
        std::vector<float> h_A_fp32(M * K), h_B_fp32(K * N);
        for (int i = 0; i < M * K; i++) h_A_fp32[i] = static_cast<float>(h_A[i]);
        for (int i = 0; i < K * N; i++) h_B_fp32[i] = static_cast<float>(h_B[i]);
        
        float *d_A_fp32, *d_B_fp32;
        HIP_CHECK(hipMalloc(&d_A_fp32, M * K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B_fp32, K * N * sizeof(float)));
        
        HIP_CHECK(hipMemcpy(d_A_fp32, h_A_fp32.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B_fp32, h_B_fp32.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
        
        float alpha_fp32 = 1.0f, beta_fp32 = 0.0f;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha_fp32,
                d_A_fp32, M,
                d_B_fp32, K,
                &beta_fp32,
                d_C_fp32, M
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        // Timing
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                M, N, K,
                &alpha_fp32,
                d_A_fp32, M,
                d_B_fp32, K,
                &beta_fp32,
                d_C_fp32, M
            ));
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double fp32_time_ms = duration.count() / 1000.0 / num_runs;
        double fp32_gflops = (2.0 * M * N * K) / (fp32_time_ms * 1e-3) / 1e9;
        
        std::cout << "  FP32 GEMM time: " << fp32_time_ms << " ms" << std::endl;
        std::cout << "  FP32 GEMM GFLOPS: " << fp32_gflops << std::endl;
        
        // Performance comparison
        double speedup = fp32_time_ms / int8_time_ms;
        double memory_reduction = 4.0; // INT8 vs FP32
        
        std::cout << "\n--- RESULTS ---" << std::endl;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        std::cout << "  Memory reduction: " << memory_reduction << "x" << std::endl;
        std::cout << "  Status: " << (speedup > 1.0 ? "SUCCESS ✅" : "FAILED ❌") << std::endl;
        
        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C_int32);
        hipFree(d_C_fp32);
        hipFree(d_A_fp32);
        hipFree(d_B_fp32);
    }
};

int main() {
    std::cout << "=== rocBLAS INT8 DIRECT TEST ===" << std::endl;
    std::cout << "Testing native AMD GPU INT8 acceleration" << std::endl;
    
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
        RocBlasInt8Tester tester;
        
        // Test different matrix sizes
        tester.test_int8_gemm(256, 3072, 768);   // Small
        tester.test_int8_gemm(512, 3072, 768);   // Medium  
        tester.test_int8_gemm(1024, 3072, 768);  // Large
        
        std::cout << "\n=== ALL TESTS COMPLETED ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}