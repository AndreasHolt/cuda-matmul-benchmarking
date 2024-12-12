//
// Created by andwh on 12/12/2024.
//

#include "matmul_helpers.cuh"

#include "cpu_matmul.h"

// helper to index into the single ptr array for the 2D matrix
// we are not using pointer-to-pointer approach as it is slower: https://stackoverflow.com/a/53978538
__device__ __host__ inline int idx_in_flattened(int row, int col, int width) {
    return row * width + col;
}

void alloc_matrices(
    float **mat_A, float **mat_B, float **mat_C,
    int m, int n, int k, float
    **d_mat_A, float **d_mat_B, float **d_mat_C) {

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
    const float *h_mat_A, const float *h_mat_B, const float *h_mat_C_gpu,
    const float *h_mat_C_cpu, int M, int N, int K
) {
    // we allocate memory for CPU result
    float *cpu_result = new float[M * N];

    // we run the cpu version
    cpu_matmul(h_mat_A, h_mat_B, cpu_result, M, N, K);

    // lastly verify the result
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_mat_C_gpu[i] - cpu_result[i]) > 1e-5) {
            delete[] cpu_result;
            return false; // cell mismatch - return false
        }
    }

    delete[] cpu_result;
    return true;
}



