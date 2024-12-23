# CUDA Matrix Multiplication Benchmarking

This project implements and benchmarks different approaches to matrix multiplication using CUDA:
- Sequential CPU implementation
- Naive GPU implementation
- Tiled GPU implementation with shared memory
- More implementations with further optimizations coming in the future

The implementations and results are discussed in detail in these blog posts on my personal site:
- [Part 1: Naive GPU Implementation, Explanation, and CPU vs naive GPU Benchmarking](https://andreasholt.com/posts/gpu-vs-cpu-matmul/)
- [Part 2: Tiled Matrix Multiplication Explained and Implemented, Benchmarking against naive GPU, and Performance Analysis with Nsight Compute](https://andreasholt.com/posts/shared-tiled-matmul/)



## Building the Project

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

The executable supports different modes:

### Run Full Benchmark Suite
```bash
./matmul
```
This runs benchmarks for all implementations across matrix sizes: 32×32, 256×256, 1024×1024, and 2048×2048.

### Profile Specific Implementation
```bash
./matmul profile <type> <dim>
```
- `<type>`: Implementation type (`naive_gpu` or `tiled_gpu`)
- `<dim>`: Matrix dimension (creates dim×dim matrices)

Example:
```bash
./matmul profile tiled_gpu 1024
```

### NVIDIA Nsight Compute Profiling
For detailed GPU metrics:
```bash
ncu --set full -o naive_2048_full.ncu-rep ./matmul profile naive_gpu 2048
ncu --set full -o tiled_2048_full.ncu-rep ./matmul profile tiled_gpu 2048
```

## Implementation Details

The project implements matrix multiplication using different approaches:
- Each thread computes one element of the output matrix (`naive_gpu`)
- Uses shared memory tiling to improve memory access patterns (`tiled_gpu`)
- Basic CPU implementation for baseline comparison (`sequential_cpu`)

Each implementation can be benchmarked and profiled independently to compare performance across different metrics. For now these metrics include GFLOPS and time (ms).