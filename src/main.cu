#include <iostream>
#include "./matmul_helpers.cuh"
#include "naive_matmul.cuh"
#include "benchmark.cuh"

void run_benchmarks() {
    int dims[] = {
        32,
        256,
        1024,
        2048
    };
    bool verify_correctness = false;
    // since we are testing for correctness against CPU, this can take very long on large matrices
    MatMulType types[] = {
        // Specify all the types that should be benchmarked
        MatMulType::NAIVE_GPU,
        MatMulType::COALESCED_GPU,
        MatMulType::TILED_COALESCED_GPU,
        MatMulType::TILED_GPU,
        MatMulType::SEQUENTIAL_CPU
    };;

    for (const int &dim: dims) {
        std::cout << "-----------" << std::endl;
        std::cout << "BENCHMARKING ON: " << dim << "×" << dim << "(A) " << "@" << dim << "×" << dim << " (B) " <<
                std::endl;

        for (const auto &type: types) {
            auto naive_gpu_result = benchmark_matmul(type, dim, dim, dim, verify_correctness);
            std::cout << "RESULTS FOR " << get_MatMulType_name(type) << ":" << std::endl;
            std::cout << "   - Time taken (ms): " << naive_gpu_result.time_ms << std::endl;
            std::cout << "   - GFLOPS: " << naive_gpu_result.gflops << std::endl;
            if (verify_correctness) {
                std::cout << "   - Correct: " << naive_gpu_result.correct << std::endl;
            }
        }
        std::cout << "-----------" << std::endl;
        std::cout << "" << std::endl;
    }
}

void test_3x2_matmul() {
    int M = 3; // rows of A
    int N = 2; // cols of B
    int K = 2; // cols of A and rows of B

    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    alloc_matrices(&A, &B, &C, &d_A, &d_B, &d_C, M, N, K);

    float A_data[] = {
        1.0f, 2.0f,
        4.0f, 5.0f,
        6.0f, 3.0f
    };

    float B_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    memcpy(A, A_data, M * K * sizeof(float));
    memcpy(B, B_data, K * N * sizeof(float));

    std::cout << "matrix A (3x2):\n";
    print_matrix(A, M, K);

    std::cout << "\nmatrix B (2x2):\n";
    print_matrix(B, K, N);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    naive_matmul(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nResult Matrix C (3x2):\n";
    print_matrix(C, M, N);

    int is_correct = verify_against_cpu_matmul(MatMulType::NAIVE_GPU, A, B, C, M, N, K);
    if (is_correct) {
        std::cout << "Correct matmul kernel" << std::endl;
    } else {
        std::cout << "Incorrect matmul kernel" << std::endl;
    }
}

void run_profile(MatMulType type, int dim) {
    for (int i = 0; i < 3; i++) {
        // we run the profile 3 times
        auto result = benchmark_matmul(type, dim, dim, dim);
        std::cout << "Run " << i + 1 << " - Time: " << result.time_ms << "ms, GFLOPS: " << result.gflops << std::endl;
    }
}


int main(int argc, char *argv[]) {
    if (argc == 1) {
        // if we have no arguments, we just run the entire suite that we also did benchmarking with
        run_benchmarks();
        return 0;
    }

    if (argc == 4 && std::string(argv[1]) == "profile") {
        MatMulType type;
        if (std::string(argv[2]) == "naive_gpu") {
            type = MatMulType::NAIVE_GPU;
        } else if (std::string(argv[2]) == "tiled_gpu") {
            type = MatMulType::TILED_GPU;
        } else if (std::string(argv[2]) == "tiled_coalesced_gpu") {
            type = MatMulType::TILED_COALESCED_GPU;
        }

        int dim = std::atoi(argv[3]);
        run_profile(type, dim);
        return 0;
    }

    std::cout << "Usage:" << std::endl;
    std::cout << "  ./matmul                    - Run full benchmark suite" << std::endl;
    std::cout << "  ./matmul profile <type> <dim> - Profile specific implementation." << std::endl;
    std::cout << "      <type>: naive_gpu, tiled_gpu" << std::endl;
    std::cout << "      <dim>: any integer" << std::endl;

    return 1;
}
