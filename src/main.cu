#include <iostream>
#include "./matmul_helpers.cuh"
#include "naive_matmul.cuh"

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

    int is_correct = verify_against_cpu_matmul(A, B, C, M, N, K);
    if (is_correct) {
        std::cout << "Correct matmul kernel" << std::endl;
    } else {
        std::cout << "Incorrect matmul kernel" << std::endl;
    }
}

int main() {
    test_3x2_matmul();
    return 0;
}
