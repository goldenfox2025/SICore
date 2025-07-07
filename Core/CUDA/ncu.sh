#!/bin/bash

# Kernel name - modify this as needed
KERNEL_NAME="kernel5"

# Executable path
EXECUTABLE="./build/gemm_test"

# NCU output file
OUTPUT_FILE="./data/gemm_profile_$(date +%Y%m%d_%H%M%S).ncu-rep"

# NCU command with comprehensive metrics
ncu \
    --kernel-name "${KERNEL_NAME}" \
    --set full\
    --target-processes all \
    --replay-mode kernel \
    --export "${OUTPUT_FILE}" \
    "${EXECUTABLE}"

# If export fails, try without it for basic profiling
if [ $? -ne 0 ]; then
    echo "Export failed, trying basic profiling..."
    ncu \
        --kernel-name "${KERNEL_NAME}" \
        --section SpeedOfLight \
        "${EXECUTABLE}"
fi

echo "Profile saved to: ${OUTPUT_FILE}"
echo "Use 'ncu-ui ${OUTPUT_FILE}' to view the results in GUI"