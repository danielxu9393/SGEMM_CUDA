// From kernel 1112, modifying things to match their warptiling kernel
// Important things:
// 1.
// constexpr is very important!!! On warpBlockSizeMN, strideAB
// Their loops for smem -> registers, fma are important to go together!!
// 2.
// My load + My fma: 20.8TFlops
// My load + their fma: 20.8TFlops
// Their load + my fma: 21.2TFlops
// Their load + their fma: 21.5TFlops
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
// Warp microblock: (NWM*NWN = 32)
// NWM: number of microtiles in the M direction per warp
// NWN: number of microtiles in the N direction per warp
// Warp block:
// WMITER: number of microcblocks in the M direction per warp
// WNITER: number of microcblocks in the N direction per warp
// DM*DN is number of warps in the block
// DM: block dimension in M direction
// DN: block dimension in N direction
template <const int BM, const int BN, const int BK, const int TM, const int TN, const int NWM, const int NWN, const int WMITER, const int WNITER>
__global__ __launch_bounds__((BM * BN) / (TM * WMITER * TN * WNITER))
void sgemmLab4RegWarpTilingMatch(int M, int N, int K, float alpha,
                  const float *A, const float *B, float beta, float *C) {

    // 2
    const int DM = BM / (TM * NWM * WMITER); // Number of warps in the M direction per block
    const int DN = BN / (TN * NWN * WNITER); // Number of warps in the N direction per block

    // 32
    constexpr int warpBlockSizeM = TM * NWM; // Stride between warp microblocks owned by each warp in the M direction
    constexpr int warpBlockSizeN = TN * NWN; // Stride between warp microblocks owned by each warp in the N direction

    // 64
    const int warpOwnSizeM = warpBlockSizeM * WMITER; // Size of everything owned by 1 warp in M direction
    const int warpOwnSizeN = warpBlockSizeN * WNITER; // Size of everything owned by 1 warp in N direction

    // DM x DN grid of warps each owning WMITER x WNITER grid of warp microblocks
    // Each warp microblocks is a NWM x NWN grid of TMxTN microtiles
    const int warpIdx = threadIdx.x / 32;
    const int warpCol = warpIdx % DN;
    const int warpRow = warpIdx / DN;

    const int threadIdxInWarp = threadIdx.x % 32;
    const int threadColInWarp = threadIdxInWarp % NWN;
    const int threadRowInWarp = threadIdxInWarp / NWN;

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
    // C += (blockRow * BM + warpRow * warpOwnSizeM) * N + (blockCol * BN + warpCol * warpOwnSizeN);

    // Allocate registers to accumulate the per-thread microtile results.
    float threadResult[(TM*WMITER) * (TN*WNITER)] = {0.0f};
    float regA[TM*WMITER] = {0.0f};
    float regB[TN*WNITER] = {0.0f};


    // float4 (LDS.128) is 4 floats wide
    constexpr uint primitiveWidth = 4;

    // These variables are for their loop codes!
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = 32*DM*DN;
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
            float4 tmp = *reinterpret_cast<const float4*>(&A[(innerRowA + loadOffset) * K + innerColA]);
            As[(innerColA + 0) * BM + (innerRowA + loadOffset)] = tmp.x;
            As[(innerColA + 1) * BM + (innerRowA + loadOffset)] = tmp.y;
            As[(innerColA + 2) * BM + (innerRowA + loadOffset)] = tmp.z;
            As[(innerColA + 3) * BM + (innerRowA + loadOffset)] = tmp.w;
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
        for (int r_t = 0; r_t < BK; ++r_t) {
            // for (int i = 0; i < WMITER; i += 1) {
            //     for (int k = 0; k < TM; k += primitiveWidth) {
            //         int col = warpRow * warpOwnSizeM + i*warpBlockSizeM + threadRowInWarp * TM + k;
            //         int row = r_t;                // current row in the shared tile
            //         *reinterpret_cast<float4*>(&regA[k + i*TM]) =
            //             *reinterpret_cast<const float4*>(&As[row * BM + col]);
            //     }
            // }
            // for (int j = 0; j < WNITER; j += 1) {
            //     for (int k = 0; k < TN; k += primitiveWidth) {
            //         int row = r_t;                // current row in the shared tile
            //         int col = warpCol * warpOwnSizeN + j*warpBlockSizeN + threadColInWarp * TN + k;   // starting column in the shared tile
            //         *reinterpret_cast<float4*>(&regB[k + j*TN]) =
            //             *reinterpret_cast<const float4*>(&Bs[row * BN + col]);
            //     }
            // }

            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                  regA[wSubRowIdx * TM + i] =
                      As[(r_t * BM) + warpRow * warpOwnSizeM + wSubRowIdx * warpBlockSizeM +
                         threadRowInWarp * TM + i];
                }
              }
              for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                  regB[wSubColIdx * TN + i] =
                      Bs[(r_t * BN) + warpCol * warpOwnSizeN + wSubColIdx * warpBlockSizeN +
                         threadColInWarp * TN + i];
                }
              }

            /************/

            // for (int i = 0; i < TM*WMITER; ++i) {
            //     for (int j = 0; j < TN*WNITER; ++j) {
            //         threadResult[i * TN*WNITER + j] += regA[i] * regB[j];
            //     }
            // }

            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                  // calculate per-thread results
                  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                      threadResult[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                    (wSubColIdx * TN) + resIdxN] +=
                          regA[wSubRowIdx * TM + resIdxM] *
                          regB[wSubColIdx * TN + resIdxN];
                    }
                  }
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
    for (int warpBlockIdxM = 0; warpBlockIdxM < WMITER; warpBlockIdxM++){
        for (int warpBlockIdxN = 0; warpBlockIdxN < WNITER; warpBlockIdxN++){
            for (int i = 0; i < TM; i += 1) {
                for (int j = 0; j < TN; j += primitiveWidth){
                    int globalRow = warpRow * warpOwnSizeM + warpBlockIdxM*warpBlockSizeM + threadRowInWarp * TM + i;
                    int globalCol = warpCol * warpOwnSizeN + warpBlockIdxN*warpBlockSizeN + threadColInWarp * TN + j;
                    int regRow = warpBlockIdxM * TM + i;
                    int regCol = warpBlockIdxN * TN + j;
                    int idx = globalRow * N + globalCol;
                    float4 a_val = *reinterpret_cast<float4*>(&threadResult[regRow * TN*WNITER + regCol]);
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

}

void runSgemmLab4RegWarpTilingMatch(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C) {
    const uint BK = 16; // 16
    const uint WMITER = 1;
    const uint WNITER = 4;
    const uint NWM = 8;
    const uint NWN = 4;
    const uint TM = 8; // 10
    const uint TN = 4;
    const uint BM = 128; // 160
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * WMITER * TN * WNITER));
    sgemmLab4RegWarpTilingMatch<BM, BN, BK, TM, TN, NWM, NWN, WMITER, WNITER>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}