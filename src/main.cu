#include <iostream>
#include "./matmul_helpers.cuh"
#include "naive_matmul.cuh"
#include "benchmark.cuh"

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

int main() {
    int dims[] = {
        32,
        256,
        1024,
        2048
    };
    bool verify_correctness = false; // since we are testing for correctness against CPU, this can take very long on large matrices
    MatMulType types[] = { // Specify all the types that should be benchmarked
        MatMulType::NAIVE_GPU,
        MatMulType::TILED_GPU,
        MatMulType::SEQUENTIAL_CPU
    };;

    for (const int& dim : dims) {
        std::cout << "-----------" << std::endl;
        std::cout << "BENCHMARKING ON: " << dim << "×" << dim << "(A) " << "@" << dim << "×" << dim << " (B) " << std::endl;

        for (const auto& type : types) {

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


    // test_3x2_matmul();
    return 0;
}
