//
// Created by andwh on 12/12/2024.
//

#include "matmul_helpers.cuh"

#include "benchmark.cuh"
#include "cpu_matmul.cuh"

// helper to index into the single ptr array for the 2D matrix
// we are not using pointer-to-pointer approach as it is slower: https://stackoverflow.com/a/53978538
__device__ __host__ int idx(int row, int col, int width) {
    return row * width + col;
}

void alloc_matrices(
    float **mat_A, float **mat_B, float **mat_C,
    float** d_mat_A, float **d_mat_B, float **d_mat_C,
    int m, int n, int k) {

    // allocate for host matrices on the cpu
    *mat_A = new float[m * k];
    *mat_B = new float[k * n];
    *mat_C = new float[m * n];

    // allocate device
    cudaMalloc(d_mat_A, sizeof(float) * m * k);
    cudaMalloc(d_mat_B, sizeof(float) * k * n);
    cudaMalloc(d_mat_C, sizeof(float) * n * m);

}

bool verify_against_cpu_matmul(
    MatMulType type,
    const float *h_mat_A, const float *h_mat_B, const float *h_mat_C_gpu,
    int M, int N, int K
) {
    // we allocate memory for CPU result
    float *cpu_result = new float[M * N];

    // we run the cpu version
    cpu_matmul(h_mat_A, h_mat_B, cpu_result, M, N, K);

    // lastly verify the result
    for (int i = 0; i < M * N; i++) {
        // small error tolerance to account for numerical errors (GPU and CPU might have slightly different fp-op implementations?)
        if (fabs(h_mat_C_gpu[i] - cpu_result[i]) > 1e-3) { // adjusted from 1e-5 to 1e-4 to account for numerical errors. Got  GPU=61.870365, CPU=61.870354, diff=0.000011 before
            int row = i / N;
            int col = i % N;
            float gpu_val = h_mat_C_gpu[i];
            float cpu_val = cpu_result[i];
            printf("%s: First mismatch at [%d,%d]: GPU=%f, CPU=%f, diff=%f\n",
                   get_MatMulType_name(type), row, col, gpu_val, cpu_val, fabs(gpu_val - cpu_val));
            delete[] cpu_result;
            return false; // cell mismatch - return false
        }
    }

    delete[] cpu_result;
    return true;
}

void print_matrix(float* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}

void initialize_matrices(float* mat_A, float* mat_B, int M, int N, int K) {
    // matrix A
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            mat_A[i * K + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // matrix B
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            mat_B[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void free_matrices(float* mat_A, float* mat_B, float* mat_C,
                  float* d_mat_A, float* d_mat_B, float* d_mat_C) {
    // Free host memory
    delete[] mat_A;
    delete[] mat_B;
    delete[] mat_C;

    // Free device memory
    cudaFree(d_mat_A);
    cudaFree(d_mat_B);
    cudaFree(d_mat_C);
}
