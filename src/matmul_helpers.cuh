//
// Created by andwh on 12/12/2024.
//

#ifndef MATMUL_HELPERS_CUH
#define MATMUL_HELPERS_CUH
#include <iostream>
#include "benchmark.cuh"

__device__ __host__ int idx(int row, int col, int width);

void alloc_matrices(
    float** mat_A, float** mat_B, float** mat_C, // Host matrices
    float** d_mat_A, float** d_mat_B, float** d_mat_C, // Device matrices
    int m, int n, int k // Dimensions
    );

bool verify_against_cpu_matmul(
    MatMulType type,
    const float *h_mat_A, const float *h_mat_B, const float *h_mat_C_gpu,
    int M, int N, int K
);

void print_matrix(float* matrix, int rows, int cols);

void initialize_matrices(float* mat_A, float* mat_B, int M, int N, int K);

void free_matrices(float* mat_A, float* mat_B, float* mat_C,
                  float* d_mat_A, float* d_mat_B, float* d_mat_C);

#endif //MATMUL_HELPERS_CUH
