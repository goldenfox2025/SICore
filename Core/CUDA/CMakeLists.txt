# CMakeLists.txt
# Place this file in Core/cuda/

cmake_minimum_required(VERSION 3.18)
project(GemmTest LANGUAGES CXX CUDA)

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Add the executable
add_executable(gemm_test gemm.cu)

# Set the C++ standard
target_compile_features(gemm_test PRIVATE cxx_std_17)

# The PTX assembly instructions (mma.sync, cp.async) require a specific
# GPU architecture. SM_80 (Ampere) is the minimum for cp.async.
# You can add other architectures like SM_86, SM_89 (Ada), SM_90 (Hopper)
# e.g., "-gencode=arch=compute_86,code=sm_86"
target_compile_options(gemm_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_89,code=sm_89>")

# Add lineinfo for debugging support
target_compile_options(gemm_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3>")

# Link against the cuBLAS library
target_link_libraries(gemm_test PRIVATE CUDA::cublas)

# Add the include directory for CUTLASS
# This path is relative to the `Core/cuda` directory where this CMakeLists.txt is located.
target_include_directories(gemm_test PRIVATE ../../third_party/cutlass/include)
target_include_directories(gemm_test PRIVATE ../../Include)
# Optional: If you need to link against CUTLASS libraries (if they exist)
# target_link_libraries(gemm_test PRIVATE cutlass)

# Set CUDA specific properties
set_target_properties(gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)

# Enable verbose output for debugging (optional)
# set_property(TARGET gemm_test PROPERTY CUDA_VERBOSE_PTXAS ON) 