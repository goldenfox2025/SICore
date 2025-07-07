#!/bin/bash

# Build script for GEMM test
# Place this file in Core/cuda/

set -e  # Exit on any error

echo "=== Building GEMM Test ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Building the project..."
make -j$(nproc)

echo "=== Build completed successfully! ==="
echo ""
echo "To run the program:"
echo "  cd build"
echo "  ./gemm_test"
echo ""
echo "Or run directly from here:"
echo "  ./build/gemm_test" 