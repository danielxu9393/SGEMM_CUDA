// From kernel 1211, actually implemented doublebuffering 
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

namespace async_common_1212 {
    __device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
        const int BYTES = 16;
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        asm volatile(
            "{\n"
            "   cp.async.cg.shared.global [%0], [%1], %2;\n"
            "}\n" ::"r"(smem),
            "l"(glob_ptr),
            "n"(BYTES));
    }
    
    __device__ __forceinline__ void async_memcpy_waitall() {
        asm volatile("cp.async.wait_all;\n" ::);
    }

    template <const int BM, const int BN, const int BK, const int strideA,
          const int strideB>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B, float *As,
                                float *Bs, int innerRowA, int innerColA,
                                int innerRowB, int innerColB) {

        for (uint offset = 0; offset < BM; offset += strideA) {
            cp_async4(&As[(innerRowA + offset) * BK + innerColA],
                        &A[(innerRowA + offset) * K + innerColA]);
        }

        for (uint offset = 0; offset + strideB <= BK; offset += strideB) {
            cp_async4(&Bs[(innerRowB + offset) * BN + innerColB],
                            &B[(innerRowB + offset) * N + innerColB]);
        }
    }

    template <const int BM, const int BN, const int BK, const int strideA>
    __device__ void transposeA(int N, int K, const float *A_source, float *A_target, int innerRowA, int innerColA) {
        for (uint offset = 0; offset < BM; offset += strideA) {
            float4 tmp = *reinterpret_cast<const float4*>(&A_source[(innerRowA + offset) * BK + innerColA]);
            A_target[(innerColA + 0) * BM + (innerRowA + offset)] = tmp.x;
            A_target[(innerColA + 1) * BM + (innerRowA + offset)] = tmp.y;
            A_target[(innerColA + 2) * BM + (innerRowA + offset)] = tmp.z;
            A_target[(innerColA + 3) * BM + (innerRowA + offset)] = tmp.w;
        }
    }
}

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
void sgemmLab5DoubleBuffering(int M, int N, int K, float alpha,
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

    // Allocate shared memory for As and Bs.
    // A_buffer: dimensions BM x BK, stored in row-major order.
    // But As is stored in BK x BM order
    // Bs: dimensions BK x BN, stored in row-major order.
    __shared__ float A_buffers[2 * BM * BK];
    float *A_buffer_list[2] = {A_buffers, A_buffers + BM * BK};

    __shared__ float As[BK * BM];
    __shared__ float B_buffers[2 * BK * BN];
    float *B_buffer_list[2] = {B_buffers, B_buffers + BK * BN};

    // Flag to track which buffer currently using
    int cur_buffer = 0;

    // Determine block tile indices (each block computes a BM x BN tile of C)
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Adjust pointers to point to the beginning of this block's tile.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

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

    async_common_1212::loadFromGmem<BM, BN, BK, strideA, strideB>(
        N, K, A, B, A_buffer_list[cur_buffer], B_buffer_list[cur_buffer], innerRowA, innerColA,
        innerRowB, innerColB);
    async_common_1212::async_memcpy_waitall();

    // Outer loop over the depth dimension of the multiplication.
    for (int bk = 0; bk < K; bk += BK) {
        int next_buffer = 1 - cur_buffer;
        
        if (bk + BK < K) {
            // Load the next block tile into the next buffer.
            async_common_1212::loadFromGmem<BM, BN, BK, strideA, strideB>(
                N, K, A + BK, B + BK * N, A_buffer_list[next_buffer], B_buffer_list[next_buffer], innerRowA, innerColA,
            innerRowB, innerColB);
        }

        // Set so we are using current buffers
        async_common_1212::transposeA<BM, BK, BK, strideA>(N, K, A_buffer_list[cur_buffer], As, innerRowA, innerColA);
        float *Bs = B_buffer_list[cur_buffer];

        __syncthreads();

        // --- Compute microtile multiplication ---
        for (int r_t = 0; r_t < BK; ++r_t) {
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
        async_common_1212::async_memcpy_waitall();
        __syncthreads();
        cur_buffer = next_buffer;

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

void runSgemmLab5DoubleBuffering(int M, int N, int K, float alpha, float *A, float *B,
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
    sgemmLab5DoubleBuffering<BM, BN, BK, TM, TN, NWM, NWN, WMITER, WNITER>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}