// From kernel 1017, not warptiling, but split 8x8 into 4x4s for better bank conflicting
// 19.5TFlops
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Template parameters:
// BM: block tile height (in rows of C)
// BN: block tile width (in columns of C)
// BK: block tile depth (tile width of A, tile height of B)
// TM: microtile (per thread) height (number of rows computed per thread)
// TN: microtile (per thread) width (number of columns computed per thread)
// DM: block dimension in M direction
// DN: block dimension in N direction
// NTM: number of microtiles in the M direction per thread
// NTN: number of microtiles in the N direction per thread
// Note: Currently, this code assumes TM=TN=primitiveWidth lol, the loops aren't smart enough because I got lazy...
template <const int BM, const int BN, const int BK, const int TM, const int TN, const int NTM, const int NTN>
__global__ __launch_bounds__((BM * BN) / (TM * NTM * TN * NTN), 1)
void sgemmLab4RegSplitTiling(int M, int N, int K, float alpha,
                  const float *A, const float *B, float beta, float *C) {

    // 2
    // const int NTM = BM / (TM * DM); // Number of microtiles in the M direction per thread
    // const int NTN = BN / (TN * DN); // Number of microtiles in the N direction per thread
    constexpr int DM = BM / (TM * NTM); // Number of threads in the M direction per block
    constexpr int DN = BN / (TN * NTN); // Number of threads in the N direction per block

    // 64
    constexpr int STM = TM * DM; // Stride between microtiles owned by each thread in the M direction
    constexpr int STN = TN * DN; // Stride between microtiles owned by each thread in the N direction

    // DM x DN grid of threads each owning NTM x NTN grid of TMxTN microtiles
    // Strided at STM x STN.
    const int threadCol = threadIdx.x % DN;
    const int threadRow = threadIdx.x / DN;

    // Allocate shared memory for A_tile and B_tile.
    // A_tile: dimensions BM x BK, stored in row-major order.
    // But As is stored in BK x BM order
    // B_tile: dimensions BK x BN, stored in row-major order.
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Determine block tile indices (each block computes a BM x BN tile of C)
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Adjust pointers to point to the beginning of this block's tile.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // Allocate registers to accumulate the per-thread microtile results.
    float threadResult[(TM*NTM) * (TN*NTN)] = {0.0f}; // BM / DM x BN / DN
    float regA[TM*NTM] = {0.0f};
    float regB[TN*NTN] = {0.0f};


    // float4 (LDS.128) is 4 floats wide
    constexpr uint primitiveWidth = 4;

    // These variables are for their loop codes!
    constexpr uint totalResultsBlocktile = BM * BN;
    constexpr uint numThreadsBlocktile = DM*DN;
    // assert(numThreadsBlocktile == blockDim.x);
    
    // Multiply everything by primitiveWidth since we load 4 at a time
    // Assumption: We need dimensions of As, Bs to be multiples of 4!!!
    // This code is all the same as 1017
    const uint innerRowA = primitiveWidth * threadIdx.x / BK;
    const uint innerColA = (primitiveWidth * threadIdx.x) % BK;
    constexpr uint strideA = primitiveWidth * numThreadsBlocktile / BK;

    const uint innerRowB = primitiveWidth * threadIdx.x / BN;
    const uint innerColB = (primitiveWidth * threadIdx.x) % BN;
    constexpr uint strideB = primitiveWidth * numThreadsBlocktile / BN;


    // Outer loop over the depth dimension of the multiplication.
    for (int bk = 0; bk < K; bk += BK) {
        // --- Load A tile into shared memory ---
        // We transpose A to avoid bank conflicts when loading shared -> registers
        // Load BM x BK from A to BK x BM in As
        // Each instruction, all threads collectively load strideA rows size BK
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            // TODO: The loop doesn't work since loadOffset is in the slowest dimension
            float4 tmp = *reinterpret_cast<const float4*>(&A[(innerRowA + loadOffset) * K + innerColA]);
            As[(innerColA + 0) * BM + (innerRowA + loadOffset)] = tmp.x;
            As[(innerColA + 1) * BM + (innerRowA + loadOffset)] = tmp.y;
            As[(innerColA + 2) * BM + (innerRowA + loadOffset)] = tmp.z;
            As[(innerColA + 3) * BM + (innerRowA + loadOffset)] = tmp.w;
            // *reinterpret_cast<float4*>(&As[(innerRowA + loadOffset) * BK + innerColA]) =
            //     *reinterpret_cast<const float4*>(&A[(innerRowA + loadOffset) * K + innerColA]);
        }
        // --- Load B tile into shared memory ---
        // Load BK x BN
        // Each instruction, all threads collectively load strideB rows size BN
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            *reinterpret_cast<float4*>(&Bs[(innerRowB + loadOffset) * BN + innerColB]) =
                *reinterpret_cast<const float4*>(&B[(innerRowB + loadOffset) * N + innerColB]);
        }

        __syncthreads();

        // --- Compute microtile multiplication ---
        // Loop over the BK dimension of the current block tile.
        // micro_chunk_size == 1 here
        for (int r_t = 0; r_t < BK; ++r_t) {
            // Load a column of A tile into registers for this microtile.
            for (int i = 0; i < NTM; i += 1) {
                int col = threadRow * TM + i*STM; // starting column in the shared tile
                int row = r_t;                // current row in the shared tile
                *reinterpret_cast<float4*>(&regA[i*primitiveWidth]) =
                    *reinterpret_cast<const float4*>(&As[row * BM + col]);
            }

            // Vectorized load of a row from Bs into regB
            for (int j = 0; j < NTN; j += 1) {
                int row = r_t;                // current row in the shared tile
                int col = threadCol * TN + j*STN;   // starting column in the shared tile
                *reinterpret_cast<float4*>(&regB[j*primitiveWidth]) =
                    *reinterpret_cast<const float4*>(&Bs[row * BN + col]);
            }

            // Unvectorizing here doesn't help:
            // for (uint wSubRowIdx = 0; wSubRowIdx < NTM; ++wSubRowIdx) {
            //     for (uint i = 0; i < TM; ++i) {
            //       regA[wSubRowIdx * TM + i] =
            //           As[(r_t * BM) + wSubRowIdx * STM +
            //              threadRow * TM + i];
            //     }
            //   }
            //   for (uint wSubColIdx = 0; wSubColIdx < STN; ++wSubColIdx) {
            //     for (uint i = 0; i < TN; ++i) {
            //       regB[wSubColIdx * TN + i] =
            //           Bs[(r_t * BN) + wSubColIdx * STN +
            //              threadCol * TN + i];
            //     }
            //   }

            // Multiply and accumulate.
            // We don't need to worry about any strides or anything
            for (int i = 0; i < TM*NTM; ++i) {
                for (int j = 0; j < TN*NTN; ++j) {
                    threadResult[i * TN*NTN + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();

        // Advance A and B pointers to the next block tile in the depth dimension.
        A += BK;
        B += BK * N;
    }

    // Write the final results to global memory with alpha and beta scaling.
    // We assume TN is multiple of 4, and shape of C are multiples of 4!!
    for (int microGroupIdxM = 0; microGroupIdxM < NTM; microGroupIdxM += 1) {
        for (int i = 0; i < TM; ++i) {
            int globalRow = threadRow * TM + microGroupIdxM * STM + i;
            int regRow = microGroupIdxM * primitiveWidth + i;
            for (int microGroupIdxN = 0; microGroupIdxN < NTN; microGroupIdxN += 1) {
                int globalCol = threadCol * TN + microGroupIdxN * STN;
                int regCol = microGroupIdxN * primitiveWidth;
                // if (globalRow + blockRow * BM < M && globalCol + blockCol * BN < N) {
                int idx = globalRow * N + globalCol;
                // C[idx] = alpha * threadResult[i * TN + j] + beta * C[idx];
                float4 a_val = *reinterpret_cast<float4*>(&threadResult[regRow * TN*NTN + regCol]);
                float4 b_val = *reinterpret_cast<float4*>(&C[idx]);
                float4 tmp;
                tmp.x = alpha * a_val.x + beta * b_val.x;
                tmp.y = alpha * a_val.y + beta * b_val.y;
                tmp.z = alpha * a_val.z + beta * b_val.z;
                tmp.w = alpha * a_val.w + beta * b_val.w;
                *reinterpret_cast<float4*>(&C[idx]) = tmp;
            }
        }
    }
}

void runSgemmLab4RegSplitTiling(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C) {
    const uint BK = 8; // 16
    const uint TM = 4; // 10
    const uint TN = 4;
    const uint NTM = 2;
    const uint NTN = 2;
    if (M >= 128 and N >= 128) {
        const uint BM = 128; // 160
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * NTM * TN * NTN));
        sgemmLab4RegSplitTiling<BM, BN, BK, TM, TN, NTM, NTN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * NTM * TN * NTN));
        sgemmLab4RegSplitTiling<BM, BN, BK, TM, TN, NTM, NTN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}