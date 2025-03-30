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
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN), 1)
void sgemmLab4RegTileAsyncLoop(int M, int N, int K, float alpha,
                  const float *A, const float *B, float beta, float *C) {
    // Determine block tile indices (each block computes a BM x BN tile of C)
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Total number of threads per block is defined as (BM * BN)/(TM * TN).
    // Each thread computes a microtile of size TM x TN.
    // threadRow and threadCol index which microtile this thread computes within the block tile.
    // const int threadCol = threadIdx.x;
    // const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Allocate shared memory for A_tile and B_tile.
    // A_tile: dimensions BM x BK, stored in row-major order.
    // B_tile: dimensions BK x BN, stored in row-major order.
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Adjust pointers to point to the beginning of this block's tile.
    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // Allocate registers to accumulate the per-thread microtile results.
    float threadResult[TM * TN] = {0.0f};
    float regA[TM] = {0.0f};
    float regB[TN] = {0.0f};

    // We use simple strides: each thread loads (and later computes) its microtile.
    // The number of microtiles per block along a dimension is BM/TM or BN/TN.
    // const int threadsPerRow = BM / TM; // same as BN / TN by assumption
    const int blockSizeM = BM / TM; // should be == blockDim.y;
    const int blockSizeN = BN / TN; 
    // const int blockSizeM = blockDim.y;
    // const int blockSizeN = blockDim.x;

    // These variables are for their loop codes!
    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    // const uint numThreadsBlocktile = blockDim.x;
    assert(numThreadsBlocktile == blockDim.x);
    
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;


    // Outer loop over the depth dimension of the multiplication.
    for (int bk = 0; bk < K; bk += BK) {
        // --- Load A tile into shared memory ---
        // for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        //     As[(innerRowA + loadOffset) * BK + innerColA] =
        //     A[(innerRowA + loadOffset) * K + innerColA];
        // }
        // for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        //     Bs[(innerRowB + loadOffset) * BN + innerColB] =
        //     B[(innerRowB + loadOffset) * N + innerColB];
        // }
        
        // Load A
        const uint tid = threadIdx.x;
        const uint total_chunks_A = BM * BK;
        for (uint chunk = tid; chunk < total_chunks_A; chunk += numThreadsBlocktile) {
            uint row = chunk / BK;
            uint col = chunk % BK;
            As[row * BK + col] = A[row * K + col];
        }

        const uint total_chunks_B = BK * BN;
        for (uint chunk = tid; chunk < total_chunks_B; chunk += numThreadsBlocktile) {
            uint row = chunk / BN;
            uint col = chunk % BN;
            Bs[row * BN + col] = B[row * N + col];
        }

        __syncthreads();

        // --- Compute microtile multiplication ---
        // Loop over the BK dimension of the current block tile.
        // micro_chunk_size == 1 here
        for (int r_t = 0; r_t < BK; ++r_t) {
            // Load a column of A tile into registers for this microtile.
            for (int i = 0; i < TM; ++i) {
                int row = threadRow * TM + i;
                int col = r_t;
                regA[i] = As[row * BK + col];
            }
            // Load a row of B tile into registers.
            for (int j = 0; j < TN; ++j) {
                int row = r_t;
                int col = threadCol * TN + j;
                regB[j] = Bs[row * BN + col];
            }
            // Multiply and accumulate.
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    threadResult[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();

        // Advance A and B pointers to the next block tile in the depth dimension.
        A += BK;
        B += BK * N;
    }

    // Write the final results to global memory with alpha and beta scaling.
    for (int i = 0; i < TM; ++i) {
        // int globalRow = blockRow * BM + threadRow * TM + i;
        int globalRow = threadRow * TM + i;
        for (int j = 0; j < TN; ++j) {
            // int globalCol = blockCol * BN + threadCol * TN + j;
            int globalCol = threadCol * TN + j;
            // if (globalRow + blockRow * BM < M && globalCol + blockCol * BN < N) {
            int idx = globalRow * N + globalCol;
            C[idx] = alpha * threadResult[i * TN + j] + beta * C[idx];
        }
    }
}