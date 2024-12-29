//
// Created by andwh on 14/12/2024.
//

#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

struct BenchmarkResult {
    double time_ms;
    double gflops;
    bool correct;
};

enum class MatMulType {
    SEQUENTIAL_CPU,
    NAIVE_GPU,
    TILED_GPU,
    TILED_COALESCED_GPU,
};

const char* get_MatMulType_name(MatMulType type);



BenchmarkResult benchmark_matmul(MatMulType type, int M, int N, int K, bool verify_correctness = false, int num_iterations = 100);



#endif //BENCHMARK_CUH
