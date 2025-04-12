// Starting from warptiling, moving towards 1112 my warptiling implementation lol
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
// template <const int BM, const int BN, const int BK, const int WM, const int WN,
//           const int WNITER, const int TM, const int TN, const int NUM_THREADS>
template <const int BM, const int BN, const int BK, const int TM, const int TN, const int NWM, const int NWN, const int WMITER, const int WNITER>
// __global__ void __launch_bounds__(NUM_THREADS)
__global__ void __launch_bounds__((BM * BN) / (TM * WMITER * TN * WNITER))
sgemmWarpTilingBackwards(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpOwnSizeM = TM * NWM * WMITER; // Stride between warp microblocks owned by each warp in the M direction
  const uint warpOwnSizeN = TN * NWN * WNITER; // Stride between warp microblocks owned by each warp in the N direction
  const uint NUM_THREADS = (BM * BN) / (TM * WMITER * TN * WNITER);

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / 32; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / warpOwnSizeN);
  const uint warpRow = warpIdx / (BN / warpOwnSizeN);

  // size of the warp subtile
//   constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
  constexpr uint warpBlockSizeM = warpOwnSizeM / WMITER; // 64/2=32
  constexpr uint warpBlockSizeN = warpOwnSizeN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % 32;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (warpBlockSizeN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (warpBlockSizeN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * warpOwnSizeM) * N + cCol * BN + warpCol * warpOwnSizeN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint strideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint strideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResult[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regA[WMITER * TM] = {0.0};
  float regB[WNITER * TN] = {0.0};

  const uint primitiveWidth = 4;

  // outer-most loop over block tiles
  for (uint bk = 0; bk < K; bk += BK) {
    for (uint offset = 0; offset + strideA <= BM; offset += strideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
      }
    
      for (uint offset = 0; offset + strideB <= BK; offset += strideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
      }
    __syncthreads();
    for (uint r_t = 0; r_t < BK; ++r_t) {
        // populate registers for whole warptile
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
    
        // execute warptile matmul
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

    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * warpBlockSizeM) * N + wSubColIdx * warpBlockSizeN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResult[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResult[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResult[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResult[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

void runSgemmWarpTilingBackwards(int M, int N, int K, float alpha, float *A, float *B,
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
    sgemmWarpTilingBackwards<BM, BN, BK, TM, TN, NWM, NWN, WMITER, WNITER>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}