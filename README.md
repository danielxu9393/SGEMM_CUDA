# Daniel Xu's implementations:
This repo is a fork of (https://github.com/siboehm/SGEMM_CUDA)[https://github.com/siboehm/SGEMM_CUDA] where I have implemented my own kernels to get better intuition on GPU performance. My kernel 1211 with async global loading performs 1.6% better then the best original implementation.

I also implemented a version with tensor cores. (Didn't spend much time on this yet)

My kernels for GFLOPs at matrix size 4096x4096 no Tensor Cores:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
|1011: My first draft with tiling |`5079.9`| 21.8% |
|1012: A,B,C pointer increment in loop |`5117.3`| 22.0% |
|1013: Collapse block dim to 1D |`8145.0`| 35.0% |
|1014: Made inner loops have each warp load consecutive floats |`12374.3`| 53.2% |
|1015: Copied the inner loops from kernel 5 (cleaner indexing) |`15756.8`| 67.8% |
|1016: Vectorized memory access |`16639.9`| 71.6% |
|1017: Tranpose A so shared -> registers load is coalesced |`18594.3`| 80.0% |
|1018: Tried back to 2D block dim (slower) |`18210.2`| 78.3% |
|1019: Start from 1017, try vectorizing loading shared -> registers|`18522.1`| 79.7% |
|1111: Split tiling |`20163.5`| 86.7% |
|1112: My warptiling implementation |`20777.4`| 89.4% |
|1113: Debugging why their warptiling better |`21659.8`| 93.2% |
|1114: Debugging why their warptiling better 2 |`21900.5`|94.2% |
|1211: Async Global Loading | `22135.9` | 95.2%   |
|1212: Double Buffering (slower) |`16270.0`|70.0% |
| 0: cuBLAS (No Tensor Cores)   | `23249.6` | 100.0%   |
<!-- benchmark_results -->

My kernels for GFLOPs at matrix size 4096x4096 with Tensor Cores:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
|1311: Added Tensor Cores first draft |`31001.8` | 53.8% | 
|-1: cuBLAS (With Tensor Cores)   | `57634.7` | 100.0%   |
<!-- benchmark_results -->


(For CuBLAS with tensor core, use kernel -1)

Build: `mkdir build && cd build && cmake .. && cmake --build .`

Rebuild: `cmake .. && cmake --build .`

Clean: `cmake --build . --target clean`

Run: `DEVICE=5 ./sgemm 1311`


# Original README

# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
