#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include "../../Include/ptx_common.h"
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <memory>
#include <string>
#include <map>
#include "gemm_kernels.cuh"
#include "gemm_cute.cuh"
#include "gemm_cute2.cuh"
#include "gemm_kernel6.cuh"
// ===== Configuration Constants =====
constexpr float ERROR_THRESHOLD_SMALL = 1e-3f;
constexpr float ERROR_THRESHOLD_MEDIUM = 1e-2f;
constexpr float ERROR_THRESHOLD_MATCH = 1e-1f;

// ===== Type Conversion Utilities =====

template <typename T>
struct TypeConverter
{
  static T from_float(float val) { return static_cast<T>(val); }
  static float to_float(T val) { return static_cast<float>(val); }
};

// Specialized converters for special types
template <>
struct TypeConverter<nv_bfloat16>
{
  static nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }
  static float to_float(nv_bfloat16 val) { return __bfloat162float(val); }
};

template <>
struct TypeConverter<half>
{
  static half from_float(float val) { return __float2half(val); }
  static float to_float(half val) { return __half2float(val); }
};

// ===== cuBLAS Wrapper =====

template <typename T>
class CublasWrapper
{
private:
  cublasHandle_t handle_;

  cudaDataType_t get_cuda_data_type() const
  {
    if constexpr (std::is_same_v<T, nv_bfloat16>)
      return CUDA_R_16BF;
    else if constexpr (std::is_same_v<T, half>)
      return CUDA_R_16F;
    else if constexpr (std::is_same_v<T, float>)
      return CUDA_R_32F;
    else
    {
      throw std::runtime_error("Unsupported data type");
    }
  }

public:
  CublasWrapper() { cublasCreate(&handle_); }
  ~CublasWrapper() { cublasDestroy(handle_); }

  void gemm(const T *A, const T *B, T *C, int M, int N, int K,
            float alpha = 1.0f, float beta = 0.0f)
  {
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    cudaDataType_t data_type = get_cuda_data_type();

    cublasStatus_t status = cublasGemmEx(
        handle_, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        &alpha, B, data_type, K,
        A, data_type, K,
        &beta, C, data_type, N,
        compute_type, algo);
    cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      throw std::runtime_error("cuBLAS GEMM failed with status: " + std::to_string(status));
    }
  }
};

// ===== Kernel Interface =====

template <typename T>
using KernelFunction = std::function<void(const T *, const T *, T *, int, int, int)>;

template <typename T>
class KernelRegistry
{
private:
  std::map<std::string, KernelFunction<T>> kernels_;

public:
  void register_kernel(const std::string &name, KernelFunction<T> kernel)
  {
    kernels_[name] = kernel;
  }

  KernelFunction<T> get_kernel(const std::string &name) const
  {
    auto it = kernels_.find(name);
    if (it == kernels_.end())
    {
      throw std::runtime_error("Kernel not found: " + name);
    }
    return it->second;
  }

  std::vector<std::string> get_kernel_names() const
  {
    std::vector<std::string> names;
    for (const auto &pair : kernels_)
    {
      names.push_back(pair.first);
    }
    return names;
  }
};

// ===== Matrix Class for Managing Data =====

template <typename T>
class Matrix
{
private:
  std::vector<T> host_data_;
  T *device_data_;
  int rows_, cols_;

public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), host_data_(rows * cols)
  {
    cudaMalloc(&device_data_, rows * cols * sizeof(T));
  }

  ~Matrix()
  {
    cudaFree(device_data_);
  }

  // Initialize with sequential values for testing
  void init_sequential()
  {
    for (int i = 0; i < rows_ * cols_; ++i)
    {
      host_data_[i] = TypeConverter<T>::from_float(static_cast<float>(i));
    }
    copy_to_device();
  }

  // Initialize with random values
  void init_random(float min_val = -1.0f, float max_val = 1.0f)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < rows_ * cols_; ++i)
    {
      host_data_[i] = TypeConverter<T>::from_float(dis(gen));
    }
    copy_to_device();
  }

  // Initialize with zeros
  void init_zeros()
  {
    std::fill(host_data_.begin(), host_data_.end(), TypeConverter<T>::from_float(0.0f));
    copy_to_device();
  }

  void copy_to_device()
  {
    cudaMemcpy(device_data_, host_data_.data(), rows_ * cols_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copy_from_device()
  {
    cudaMemcpy(host_data_.data(), device_data_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost);
  }

  T *device_ptr() { return device_data_; }
  const T *device_ptr() const { return device_data_; }
  std::vector<T> &host_data() { return host_data_; }
  const std::vector<T> &host_data() const { return host_data_; }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int size() const { return rows_ * cols_; }
};

// ===== Comparison and Analysis =====

template <typename T>
class ResultComparator
{
public:
  struct ComparisonResult
  {
    float max_abs_diff;
    float avg_abs_diff;
    float max_rel_diff;
    float avg_rel_diff;
    int exact_matches;
    int total_elements;
    float match_percentage;

    // Error distribution
    int small_error_count;
    int medium_error_count;
    int large_error_count;

    // Top 10 largest differences
    std::vector<std::tuple<int, float, float, float, float>> top_10_differences;
  };

  static ComparisonResult compare(const Matrix<T> &mat1, const Matrix<T> &mat2,
                                  const std::string &name1 = "Result 1",
                                  const std::string &name2 = "Result 2")
  {
    if (mat1.size() != mat2.size())
    {
      throw std::runtime_error("Matrix sizes don't match for comparison");
    }

    ComparisonResult result = {};
    result.total_elements = mat1.size();

    const auto &data1 = mat1.host_data();
    const auto &data2 = mat2.host_data();

    float sum_abs_diff = 0.0f;
    float sum_rel_diff = 0.0f;

    // 收集所有差异信息用于找出最大的十个
    std::vector<std::tuple<int, float, float, float, float>> all_differences;
    all_differences.reserve(result.total_elements);

    for (int i = 0; i < result.total_elements; ++i)
    {
      float val1 = TypeConverter<T>::to_float(data1[i]);
      float val2 = TypeConverter<T>::to_float(data2[i]);

      float abs_diff = std::abs(val1 - val2);
      float rel_diff = (val1 != 0.0f) ? abs_diff / std::abs(val1) : 0.0f;

      result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);
      result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
      sum_abs_diff += abs_diff;
      sum_rel_diff += rel_diff;

      if (abs_diff < ERROR_THRESHOLD_MATCH)
        result.exact_matches++;

      // Count error distribution
      if (rel_diff < ERROR_THRESHOLD_SMALL)
        result.small_error_count++;
      else if (rel_diff < ERROR_THRESHOLD_MEDIUM)
        result.medium_error_count++;
      else
        result.large_error_count++;

      // 存储差异信息
      all_differences.emplace_back(i, val1, val2, abs_diff, rel_diff);
    }

    result.avg_abs_diff = sum_abs_diff / result.total_elements;
    result.avg_rel_diff = sum_rel_diff / result.total_elements;
    result.match_percentage = 100.0f * result.exact_matches / result.total_elements;

    // 按相对误差排序，找出最大的十个
    std::sort(all_differences.begin(), all_differences.end(),
              [](const auto &a, const auto &b)
              {
                return std::get<4>(a) > std::get<4>(b); // 按相对误差降序排序
              });

    // 取前十个
    int top_count = std::min(10, static_cast<int>(all_differences.size()));
    result.top_10_differences.assign(all_differences.begin(), all_differences.begin() + top_count);

    return result;
  }

  static void print_comparison(const ComparisonResult &result,
                               const std::string &name1 = "Result 1",
                               const std::string &name2 = "Result 2")
  {
    std::cout << "\n=== " << name1 << " vs " << name2 << " ===" << std::endl;
    std::cout << "Total elements: " << result.total_elements << std::endl;
    std::cout << "Max absolute error: " << std::scientific << std::setprecision(6) << result.max_abs_diff << std::endl;
    std::cout << "Average absolute error: " << std::scientific << std::setprecision(6) << result.avg_abs_diff << std::endl;
    std::cout << "Max relative error: " << std::scientific << std::setprecision(6) << result.max_rel_diff << std::endl;
    std::cout << "Average relative error: " << std::scientific << std::setprecision(6) << result.avg_rel_diff << std::endl;
    std::cout << "Exact matches: " << result.exact_matches << " (" << std::fixed << std::setprecision(2)
              << result.match_percentage << "%)" << std::endl;

    std::cout << "\nError distribution:" << std::endl;
    std::cout << "Small errors (< " << ERROR_THRESHOLD_SMALL << "): " << result.small_error_count
              << " (" << std::fixed << std::setprecision(2) << (100.0f * result.small_error_count / result.total_elements) << "%)" << std::endl;
    std::cout << "Medium errors (" << ERROR_THRESHOLD_SMALL << " ~ " << ERROR_THRESHOLD_MEDIUM << "): " << result.medium_error_count
              << " (" << std::fixed << std::setprecision(2) << (100.0f * result.medium_error_count / result.total_elements) << "%)" << std::endl;
    std::cout << "Large errors (> " << ERROR_THRESHOLD_MEDIUM << "): " << result.large_error_count
              << " (" << std::fixed << std::setprecision(2) << (100.0f * result.large_error_count / result.total_elements) << "%)" << std::endl;

    // 打印差距最大的十个值
    if (!result.top_10_differences.empty())
    {
      std::cout << "\nTop 10 largest relative differences:" << std::endl;
      std::cout << std::left << std::setw(8) << "Index"
                << std::setw(15) << name1 + " Value"
                << std::setw(15) << name2 + " Value"
                << std::setw(15) << "Abs Diff"
                << std::setw(15) << "Rel Diff" << std::endl;
      std::cout << std::string(70, '-') << std::endl;

      for (const auto &diff : result.top_10_differences)
      {
        int index = std::get<0>(diff);
        float val1 = std::get<1>(diff);
        float val2 = std::get<2>(diff);
        float abs_diff = std::get<3>(diff);
        float rel_diff = std::get<4>(diff);

        std::cout << std::left << std::setw(8) << index
                  << std::setw(15) << std::scientific << std::setprecision(6) << val1
                  << std::setw(15) << std::scientific << std::setprecision(6) << val2
                  << std::setw(15) << std::scientific << std::setprecision(6) << abs_diff
                  << std::setw(15) << std::scientific << std::setprecision(6) << rel_diff << std::endl;
      }
    }
  }
};

// ===== Performance Benchmarking =====

template <typename T>
class PerformanceBenchmark
{
public:
  struct BenchmarkResult
  {
    std::string name;
    double avg_time_ms;
    double throughput_gflops;
    int test_runs;
    double total_time_ms;
  };

  static BenchmarkResult benchmark_kernel(const std::string &name,
                                          KernelFunction<T> kernel,
                                          const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C,
                                          int M, int N, int K,
                                          int warmup_runs = 10, int test_runs = 100)
  {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i)
    {
      kernel(A.device_ptr(), B.device_ptr(), C.device_ptr(), M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_runs; ++i)
    {
      kernel(A.device_ptr(), B.device_ptr(), C.device_ptr(), M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    BenchmarkResult result;
    result.name = name;
    result.test_runs = test_runs;
    result.total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result.avg_time_ms = result.total_time_ms / test_runs;
    result.throughput_gflops = (2.0 * M * N * K * test_runs) / (result.total_time_ms * 1e6);

    return result;
  }

  static BenchmarkResult benchmark_cublas(CublasWrapper<T> &cublas,
                                          const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C,
                                          int M, int N, int K,
                                          int warmup_runs = 10, int test_runs = 100)
  {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i)
    {
      cublas.gemm(A.device_ptr(), B.device_ptr(), C.device_ptr(), M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_runs; ++i)
    {
      cublas.gemm(A.device_ptr(), B.device_ptr(), C.device_ptr(), M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    BenchmarkResult result;
    result.name = "cuBLAS";
    result.test_runs = test_runs;
    result.total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result.avg_time_ms = result.total_time_ms / test_runs;
    result.throughput_gflops = (2.0 * M * N * K * test_runs) / (result.total_time_ms * 1e6);

    return result;
  }

  static void print_benchmark_result(const BenchmarkResult &result)
  {
    std::cout << result.name << ":" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << result.avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << result.throughput_gflops << " GFLOPS" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << result.total_time_ms << " ms" << std::endl;
    std::cout << "  Test runs: " << result.test_runs << std::endl;
  }
};

// ===== Test Suite =====

template <typename T>
class GemmTestSuite
{
private:
  KernelRegistry<T> kernel_registry_;
  CublasWrapper<T> cublas_;

public:
  GemmTestSuite()
  {
    register_default_kernels();
  }

  void register_default_kernels()
  {
    // Register naive kernel
    kernel_registry_.register_kernel("naive", [](const T *A, const T *B, T *C, int M, int N, int K)
                                     {
            dim3 block_size(16, 16);
            dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
            naive_gemm_kernel<T><<<grid_size, block_size>>>(A, B, C, M, N, K); });
  }

  void register_kernel(const std::string &name, KernelFunction<T> kernel)
  {
    kernel_registry_.register_kernel(name, kernel);
  }

  void run_single_test(int M, int N, int K, const std::string &kernel_name = "naive")
  {
    std::cout << "\n=== Single Test: " << kernel_name << " (" << M << "x" << N << "x" << K << ") ===" << std::endl;

    // Create matrices
    Matrix<T> A(M, K), B(K, N), C_kernel(M, N), C_cublas(M, N);

    // Initialize data
    A.init_random(-10.0f, 10.0f);
    B.init_random(-10.0f, 10.0f);
    C_kernel.init_zeros();
    C_cublas.init_zeros();

    // Run kernel
    auto kernel = kernel_registry_.get_kernel(kernel_name);
    kernel(A.device_ptr(), B.device_ptr(), C_kernel.device_ptr(), M, N, K);

    // Run cuBLAS
    cublas_.gemm(A.device_ptr(), B.device_ptr(), C_cublas.device_ptr(), M, N, K);

    // Copy results back
    C_kernel.copy_from_device();
    C_cublas.copy_from_device();

    // Compare results
    auto comparison = ResultComparator<T>::compare(C_cublas, C_kernel, "cuBLAS", kernel_name);
    ResultComparator<T>::print_comparison(comparison, "cuBLAS", kernel_name);
  }

  void run_benchmark_suite(const std::vector<std::tuple<int, int, int>> &test_sizes,
                           int warmup_runs = 5, int test_runs = 100)
  {
    std::cout << "\n=== Benchmark Suite ===" << std::endl;
    std::cout << std::string(120, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Size"
              << std::setw(15) << "Kernel"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "vs cuBLAS"
              << std::setw(15) << "Accuracy" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    auto kernel_names = kernel_registry_.get_kernel_names();

    for (const auto &size : test_sizes)
    {
      int M, N, K;
      std::tie(M, N, K) = size;

      std::string size_str = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

      // Create matrices
      Matrix<T> A(M, K), B(K, N), C_kernel(M, N), C_cublas(M, N);
      A.init_random(-10.0f, 10.0f);
      B.init_random(-10.0f, 10.0f);

      // Benchmark cuBLAS first
      C_cublas.init_zeros();
      auto cublas_result = PerformanceBenchmark<T>::benchmark_cublas(cublas_, A, B, C_cublas, M, N, K, warmup_runs, test_runs);
      C_cublas.copy_from_device();

      std::cout << std::left << std::setw(15) << size_str
                << std::setw(15) << "cuBLAS"
                << std::setw(15) << std::fixed << std::setprecision(3) << cublas_result.avg_time_ms
                << std::setw(15) << std::fixed << std::setprecision(2) << cublas_result.throughput_gflops
                << std::setw(15) << "1.00"
                << std::setw(15) << "Reference" << std::endl;

      // Benchmark each kernel
      for (const auto &kernel_name : kernel_names)
      {
        C_kernel.init_zeros();
        auto kernel = kernel_registry_.get_kernel(kernel_name);
        auto kernel_result = PerformanceBenchmark<T>::benchmark_kernel(kernel_name, kernel, A, B, C_kernel, M, N, K, warmup_runs, test_runs);
        C_kernel.copy_from_device();

        // Calculate speedup
        double speedup = cublas_result.avg_time_ms / kernel_result.avg_time_ms;

        // Calculate accuracy
        auto comparison = ResultComparator<T>::compare(C_cublas, C_kernel);
        std::string accuracy_str = (comparison.avg_rel_diff < ERROR_THRESHOLD_SMALL) ? "Excellent" : (comparison.avg_rel_diff < ERROR_THRESHOLD_MEDIUM) ? "Good"
                                                                                                                                                        : "Poor";

        std::cout << std::left << std::setw(15) << ""
                  << std::setw(15) << kernel_name
                  << std::setw(15) << std::fixed << std::setprecision(3) << kernel_result.avg_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << kernel_result.throughput_gflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(15) << accuracy_str << std::endl;
      }

      std::cout << std::string(120, '-') << std::endl;
    }
  }
};

// ===== Example Usage =====

int main()
{
  // Test with different data types
  std::cout << "=== Testing with float ===" << std::endl;
  GemmTestSuite<float> float_suite;

  float_suite.register_kernel("kernel1", [](const float *A, const float *B, float *C, int M, int N, int K)
                              {
        dim3 block_size(32*32/4);
        dim3 grid_size((N + 32 - 1) / 32, (M + 32 - 1) / 32);
        kernel1<float,32,32,32,2,2><<<grid_size, block_size>>>(A, B, C, M, N, K); });
  // Run a single test
  // float_suite.run_single_test(1024, 1024, 1024, "kernel1");

  // // You can easily add your own kernel like this:

  // // Run comprehensive benchmark
  std::vector<std::tuple<int, int, int>> test_sizes = {
      {256, 256, 256},
      {512, 512, 512},
      {1024, 1024, 1024},
      {2048, 2048, 2048}};

  // float_suite.run_benchmark_suite(test_sizes);

  // Test with bfloat16
  std::cout << "\n=== Testing with bfloat16 ===" << std::endl;
  GemmTestSuite<nv_bfloat16> bf16_suite;
  bf16_suite.register_kernel("kernel1", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
        dim3 block_size(32*32/4);
        dim3 grid_size((N + 32 - 1) / 32, (M + 32 - 1) / 32);
        kernel1<nv_bfloat16,32,32,32,2,2><<<grid_size, block_size>>>(A, B, C, M, N, K); 
               cudaDeviceSynchronize(); });
  // bf16_suite.register_kernel("kernel1_1x1", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
  //                            {
  //       dim3 block_size(32*32/1);
  //       dim3 grid_size((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  //       kernel1<nv_bfloat16,32,32,32,1,1><<<grid_size, block_size>>>(A, B, C, M, N, K);

  //              cudaDeviceSynchronize(); });
  bf16_suite.register_kernel("kernel2", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
                              constexpr int BM = 32;
                              constexpr int BN = 64;
                              constexpr int BK = 64;
                              constexpr int WMMA_M = 16;
                              constexpr int WMMA_N = 16;
                              constexpr int WMMA_K = 16;
                              constexpr int WAPR_NUM = 8;
        dim3 block_size(WAPR_NUM*32);
        dim3 grid_size((M + BM - 1) / BM,(N + BN - 1) / BN);
        kernel2<nv_bfloat16,BM,BN,BK,WMMA_M,WMMA_N,WMMA_K,WAPR_NUM><<<grid_size, block_size>>>(A, B, C, M, N, K); 
        
               cudaDeviceSynchronize(); });
  bf16_suite.register_kernel("kernel3", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
                  constexpr int BM = 16;
                  constexpr int BN = 16;
                  constexpr int BK = 16;
                  constexpr int WMMA_M = 16;
                  constexpr int WMMA_N = 16;
                  constexpr int WMMA_K = 16;
                  constexpr int K_STAGE = 2;
                  constexpr int WARP_TILE_M = 1;
                  constexpr int WARP_TILE_N = 1;
                  constexpr int WAPR_NUM = BM/WMMA_M*BN/WMMA_N /WARP_TILE_M/WARP_TILE_N;
        dim3 block_size(WAPR_NUM*32);
        dim3 grid_size((M + BM - 1) / BM,(N + BN - 1) / BN);

        kernel3<nv_bfloat16,BM,BN,BK,WMMA_M,WMMA_N,WMMA_K,WAPR_NUM,K_STAGE,WARP_TILE_M,WARP_TILE_N><<<grid_size, block_size>>>(A, B, C, M, N, K); 
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
        } });
  bf16_suite.register_kernel("kernel4", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
                  constexpr int BM = 64;
                  constexpr int BN = 64;
                  constexpr int BK = 16;
                  constexpr int WMMA_M = 16;
                  constexpr int WMMA_N = 8;
                  constexpr int WMMA_K = 16;
                  constexpr int K_STAGE = 2;
                  constexpr int WARP_TILE_M = 1;
                  constexpr int WARP_TILE_N = 1;
                  constexpr int WAPR_NUM = BM/WMMA_M*BN/WMMA_N /WARP_TILE_M/WARP_TILE_N;
        dim3 block_size(WAPR_NUM*32);
        dim3 grid_size((M + BM - 1) / BM,(N + BN - 1) / BN);

        kernel4<nv_bfloat16,BM,BN,BK,WMMA_M,WMMA_N,WMMA_K,WAPR_NUM,K_STAGE,WARP_TILE_M,WARP_TILE_N><<<grid_size, block_size>>>(A, B, C, M, N, K); 
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
        } });
  bf16_suite.register_kernel("cute", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
        launch_simple_cute<nv_bfloat16>(const_cast<nv_bfloat16*>(A), const_cast<nv_bfloat16*>(B), C, M, N, K);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUTE kernel launch failed: %s\n", cudaGetErrorString(error));
        } });
  bf16_suite.register_kernel("cute2", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
        launch_gemm_mma_stages_block_swizzle_tn_cute2<nv_bfloat16, 2, false>(const_cast<nv_bfloat16*>(A), const_cast<nv_bfloat16*>(B), C, M, N, K, 0);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUTE2 kernel launch failed: %s\n", cudaGetErrorString(error));
        } });
  bf16_suite.register_kernel("kernel5", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
                  constexpr int BM = 128;
                  constexpr int BN = 64;
                  constexpr int BK = 16;
                  constexpr int WMMA_M = 16;
                  constexpr int WMMA_N = 8;
                  constexpr int WMMA_K = 16;
                  constexpr int K_STAGE = 2;
                  constexpr int WARP_TILE_M = 4;
                  constexpr int WARP_TILE_N = 4;
                  constexpr int WAPR_NUM = BM/WMMA_M*BN/WMMA_N /WARP_TILE_M/WARP_TILE_N;
        dim3 block_size(WAPR_NUM*32);
        dim3 grid_size((M + BM - 1) / BM,(N + BN - 1) / BN);

        kernel5<nv_bfloat16,BM,BN,BK,WMMA_M,WMMA_N,WMMA_K,WAPR_NUM,K_STAGE,WARP_TILE_M,WARP_TILE_N><<<grid_size, block_size>>>(A, B, C, M, N, K); 
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
        } });
  bf16_suite.register_kernel("kernel6", [](const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K)
                             {
                  constexpr int BM = 128;
                  constexpr int BN = 64;
                  constexpr int BK = 16;
                  constexpr int WMMA_M = 16;
                  constexpr int WMMA_N = 16;  // 16x16x16 MMA
                  constexpr int WMMA_K = 16;
                  constexpr int K_STAGE = 2;
                  constexpr int WARP_TILE_M = 4;
                  constexpr int WARP_TILE_N = 2;  // Adjusted for 16x16 N dimension
                  constexpr int WAPR_NUM = BM/WMMA_M*BN/WMMA_N /WARP_TILE_M/WARP_TILE_N;
        dim3 block_size(WAPR_NUM*32);
        dim3 grid_size((M + BM - 1) / BM,(N + BN - 1) / BN);

        kernel6<nv_bfloat16,BM,BN,BK,WMMA_M,WMMA_N,WMMA_K,WAPR_NUM,K_STAGE,WARP_TILE_M,WARP_TILE_N><<<grid_size, block_size>>>(A, B, C, M, N, K); 
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("Kernel6 launch failed: %s\n", cudaGetErrorString(error));
        } });

  bf16_suite.run_single_test(2048, 2048, 2048, "cute2");
  // bf16_suite.run_single_test(1024, 1024, 1024, "kernel2");
  bf16_suite.run_benchmark_suite(test_sizes);

  return 0;
}