#!/usr/bin/env bash
# This script should be run from the scripts folder

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WMITER_VALUES=(1 2 4 8)
WNITER_VALUES=(1 2 4 8)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/kernels/1311_kernel_lab6_tensor.cuh"
OUTPUT="../benchmark_results/kernel_1311_autotune_results.txt"

# Clear the output file
echo "" > "$OUTPUT"

# GPU device and constants
export DEVICE="0"
WARPSIZE=32

# Compute total number of configurations
TOTAL_CONFIGS=$(( ${#BK_VALUES[@]} \
                 * ${#BM_VALUES[@]} \
                 * ${#BN_VALUES[@]} \
                 * ${#WMITER_VALUES[@]} \
                 * ${#WNITER_VALUES[@]} ))
CONFIG_NUM=0

# Sweep over all parameter combinations
for BK in "${BK_VALUES[@]}"; do
  for BM in "${BM_VALUES[@]}"; do
    for BN in "${BN_VALUES[@]}"; do
      for WMITER in "${WMITER_VALUES[@]}"; do
        for WNITER in "${WNITER_VALUES[@]}"; do

          CONFIG_NUM=$(( CONFIG_NUM + 1 ))
          echo ""
          echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$BK BM=$BM BN=$BN WMITER=$WMITER WNITER=$WNITER" |& tee -a "$OUTPUT"

          # Patch the kernel source with the current values
          sed -i "s/const uint BK = .*/const uint BK = $BK;/" "$RUNNER"
          sed -i "s/const uint BM = .*/const uint BM = $BM;/" "$RUNNER"
          sed -i "s/const uint BN = .*/const uint BN = $BN;/" "$RUNNER"
          sed -i "s/const uint WMITER = .*/const uint WMITER = $WMITER;/" "$RUNNER"
          sed -i "s/const uint WNITER = .*/const uint WNITER = $WNITER;/" "$RUNNER"

          # Rebuild
          make

          # Run benchmark (time out if it hangs)
          timeout -v 8 ./sgemm 1311 |& tee -a "$OUTPUT"

        done
      done
    done
  done
done