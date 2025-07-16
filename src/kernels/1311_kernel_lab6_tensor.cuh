// From kernel 1113, adding tensor cores
// 31TFlops!!
// TODOs: Reorganize before writing to global memory
// Do we still need to tranpose A?
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__device__ inline void mma_16x8x8(uint32_t const *a, uint32_t const *b, uint32_t *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3},     /* 'D' matrix */ \n\t"
        "{%4, %5, %6, %7},     /* 'A' matrix */ \n\t"
        "{%8, %9},             /* 'B' matrix */ \n\t"
        "{%0, %1, %2, %3};     /* 'C' matrix (read-modify-write) */ \n\t"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}


// Template parameters:
// BM: block tile height (in rows of C)
// BN: block tile width (in columns of C)
// BK: block tile depth (tile width of A, tile height of B)
// TM = 16
// TN = 8
// Warp block:
// WMITER: number of microcblocks in the M direction per warp
// WNITER: number of microcblocks in the N direction per warp
// DM*DN is number of warps in the block
// DM: block dimension in M direction
// DN: block dimension in N direction
template <const int BM, const int BN, const int BK, const int TM, const int TN, const int TK, const int WMITER, const int WNITER>
__global__ __launch_bounds__(32 * (BM * BN) / (TM * WMITER * TN * WNITER))
void sgemmLab6Tensor(int M, int N, int K, float alpha,
                  const float *A, const float *B, float beta, float *C) {

    // 2
    const int DM = BM / (TM * WMITER); // Number of warps in the M direction per block
    const int DN = BN / (TN * WNITER); // Number of warps in the N direction per block

    // 64
    const int warpOwnSizeM = TM * WMITER; // Size of everything owned by 1 warp in M direction
    const int warpOwnSizeN = TN * WNITER; // Size of everything owned by 1 warp in N direction

    // DM x DN grid of warps each owning WMITER x WNITER grid of warp microblocks
    // Each warp microblocks is a NWM x NWN grid of TMxTN microtiles
    const int warpIdx = threadIdx.x / 32;
    const int warpCol = warpIdx % DN;
    const int warpRow = warpIdx / DN;

    const int threadIdxInWarp = threadIdx.x % 32;

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
    // float threadResult[(TM*WMITER) * (TN*WNITER)] = {0.0f};

    // TODO: Change this to do the 4,2 thingy
    uint32_t regA[4*WMITER] = {0};
    uint32_t regB[2*WNITER] = {0};
    uint32_t regC[4*WMITER*WNITER] = {0};


    // float4 (LDS.128) is 4 floats wide
    constexpr uint primitiveWidth = 4;

    // These variables are for their loop codes!
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = 32*DM*DN;
    assert(numThreadsBlocktile == blockDim.x);
    
    // Multiply everything by primitiveWidth since we load 4 at a time
    // Assumption: We need dimensions of As, Bs to be multiples of 4!!!
    // This code is all the same as 1017
    // To avoid errors, we need primitiveWidth * numThreadsBlocktile <= BM*BK
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
        for (int r_t = 0; r_t < BK; r_t += TK) {

            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                // for (uint i = 0; i < TM; ++i) {
                //   regA[wSubRowIdx * TM + i] =
                //       As[(r_t * BM) + warpRow * warpOwnSizeM + wSubRowIdx * warpBlockSizeM +
                //          threadRowInWarp * TM + i];
                // }
                for (int q = 0; q < 4; q++) {
                    int q_row_start = (q % 2) ? 8 : 0; // 8 if q odd
                    int q_col_start = (q >= 2) ? 4 : 0; // 4 if q >= 2
                    int col = warpRow * warpOwnSizeM + wSubRowIdx * TM + q_row_start + threadIdxInWarp / 4;
                    int row = r_t + q_col_start + threadIdxInWarp % 4;
                    regA[wSubRowIdx*4 + q] = __float_as_uint(As[row * BM + col]);
                }

              }
              for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // for (uint i = 0; i < TN; ++i) {
                //   regB[wSubColIdx * TN + i] =
                //       Bs[(r_t * BN) + warpCol * warpOwnSizeN + wSubColIdx * warpBlockSizeN +
                //          threadColInWarp * TN + i];
                // }
                for (int q = 0; q < 2; q++) {
                    int q_row_start = (q % 2) ? 4 : 0; // 4 if q odd
                    int row = r_t + q_row_start + threadIdxInWarp % 4;
                    int col = warpCol * warpOwnSizeN + wSubColIdx * TN + threadIdxInWarp / 4;
                    regB[wSubColIdx*2 + q] = __float_as_uint(Bs[row * BN + col]);
                }
              }

            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    mma_16x8x8(&regA[wSubRowIdx * 4], &regB[wSubColIdx * 2], &regC[(wSubRowIdx*WNITER + wSubColIdx)*4]);
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
            // for (int i = 0; i < TM; i += 1) {
            //     for (int j = 0; j < TN; j += primitiveWidth){
            //         int globalRow = warpRow * warpOwnSizeM + warpBlockIdxM*warpBlockSizeM + threadRowInWarp * TM + i;
            //         int globalCol = warpCol * warpOwnSizeN + warpBlockIdxN*warpBlockSizeN + threadColInWarp * TN + j;
            //         int regRow = warpBlockIdxM * TM + i;
            //         int regCol = warpBlockIdxN * TN + j;
            //         int idx = globalRow * N + globalCol;
            //         float4 a_val = *reinterpret_cast<float4*>(&threadResult[regRow * TN*WNITER + regCol]);
            //         float4 b_val = *reinterpret_cast<float4*>(&C[idx]);
            //         float4 tmp;
            //         tmp.x = alpha * a_val.x + beta * b_val.x;
            //         tmp.y = alpha * a_val.y + beta * b_val.y;
            //         tmp.z = alpha * a_val.z + beta * b_val.z;
            //         tmp.w = alpha * a_val.w + beta * b_val.w;
            //         *reinterpret_cast<float4*>(&C[idx]) = tmp;
            //     }
            // }
            // TODO: Just having this ugly loop for now
            // Later, I will reorganize the memory and coalesce global writes!!
            int q_col_stride = 2;
            for (int q = 0; q < 4; q++) {
                int q_row_start = (q >= 2) ? 8 : 0; // 8 if q >= 2
                int q_col_start = (q % 2) ? 1 : 0; // 1 if q odd
                int row = q_row_start + threadIdxInWarp / 4;
                int col = q_col_start + (threadIdxInWarp % 4) * q_col_stride;
                int globalRow = warpRow * warpOwnSizeM + warpBlockIdxM * TM + row;
                int globalCol = warpCol * warpOwnSizeN + warpBlockIdxN * TN + col;

                C[globalRow * N + globalCol] = alpha * __uint_as_float(regC[(warpBlockIdxM * WNITER + warpBlockIdxN) * 4 + q]) +
                    beta * C[globalRow * N + globalCol];
            }
        }
    }

}

void runSgemmLab6Tensor(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C) {
    const uint BK = 64;
    const uint WMITER = 8;
    const uint WNITER = 8;
    const uint TM = 16;
    const uint TN = 8;
    const uint TK = 8;
    const uint BM = 256;
    const uint BN = 256;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(32 * (BM * BN) / (TM * WMITER * TN * WNITER));
    sgemmLab6Tensor<BM, BN, BK, TM, TN, TK, WMITER, WNITER>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}