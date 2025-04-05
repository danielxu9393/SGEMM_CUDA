// Tranpose A so shared -> registers load is coalesced
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
void sgemmLab4RegVectorizeTransposeA(int M, int N, int K, float alpha,
                  const float *A, const float *B, float beta, float *C) {
    // Determine block tile indices (each block computes a BM x BN tile of C)
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Total number of threads per block is defined as (BM * BN)/(TM * TN).
    // Each thread computes a microtile of size TM x TN.
    // threadRow and threadCol index which microtile this thread computes within the block tile.
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
    const int blockSizeM = BM / TM;
    const int blockSizeN = BN / TN; 

    // float4 (LDS.128) is 4 floats wide
    const uint primitiveWidth = 4;

    // These variables are for their loop codes!
    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    // const uint numThreadsBlocktile = blockDim.x;
    assert(numThreadsBlocktile == blockDim.x);
    
    // Multiply everything by primitiveWidth since we load 4 at a time
    // Assumption: We need dimensions of As, Bs to be multiples of 4!!!
    const uint innerRowA = primitiveWidth * threadIdx.x / BK;
    const uint innerColA = (primitiveWidth * threadIdx.x) % BK;
    const uint strideA = primitiveWidth * numThreadsBlocktile / BK;

    const uint innerRowB = primitiveWidth * threadIdx.x / BN;
    const uint innerColB = (primitiveWidth * threadIdx.x) % BN;
    const uint strideB = primitiveWidth * numThreadsBlocktile / BN;


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
            for (int i = 0; i < TM; ++i) {
                int col = threadRow * TM + i;
                int row = r_t;
                regA[i] = As[row * BM + col];
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
    // We assume TN is multiple of 4, and shape of C are multiples of 4!!
    for (int i = 0; i < TM; ++i) {
        // int globalRow = blockRow * BM + threadRow * TM + i;
        int globalRow = threadRow * TM + i;
        for (int j = 0; j < TN; j+=4) {
            // int globalCol = blockCol * BN + threadCol * TN + j;
            int globalCol = threadCol * TN + j;
            // if (globalRow + blockRow * BM < M && globalCol + blockCol * BN < N) {
            int idx = globalRow * N + globalCol;
            // C[idx] = alpha * threadResult[i * TN + j] + beta * C[idx];
            float4 a_val = *reinterpret_cast<float4*>(&threadResult[i * TN + j]);
            float4 b_val = *reinterpret_cast<float4*>(&C[idx]);
            float4 tmp;
            tmp.x = alpha * a_val.x + beta * b_val.x;
            tmp.y = alpha * a_val.y + beta * b_val.y;
            tmp.z = alpha * a_val.z + beta * b_val.z;
            tmp.w = alpha * a_val.w + beta * b_val.w;

            // Their write code is slower than mine!!!
            // 16.3GFlops on 4096 with their code, but 16.6GFlops on 4096 with mine!!
            
            // float4 tmp = reinterpret_cast<float4 *>(
            //     &C[idx])[0];
            // // perform GEMM update in reg
            // tmp.x = alpha * threadResult[i * TN + j] + beta * tmp.x;
            // tmp.y = alpha * threadResult[i * TN + j + 1] + beta * tmp.y;
            // tmp.z = alpha * threadResult[i * TN + j + 2] + beta * tmp.z;
            // tmp.w = alpha * threadResult[i * TN + j + 3] + beta * tmp.w;
            *reinterpret_cast<float4*>(&C[idx]) = tmp;
        }
    }
}

void runSgemmLab4RegVectorizeTransposeA(int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C) {
    const uint BK = 8; // 16
    const uint TM = 8; // 10
    const uint TN = 8;
    if (M >= 128 and N >= 128) {
        const uint BM = 128; // 160
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        // dim3 blockDim((BN /TN), (BM / TM));
        sgemmLab4RegVectorizeTransposeA<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        // dim3 blockDim((BN /TN), (BM / TM));
        sgemmLab4RegVectorizeTransposeA<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}