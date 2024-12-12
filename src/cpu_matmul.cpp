//
// Created by andwh on 12/12/2024.
//

#include "cpu_matmul.h"

#include "naive_matmul.cuh"

void cpu_matmul(
    const float *mat_A, const float *mat_B, float *mat_C,
    int M, int N, int K
    )
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += mat_A[idx_in_flattened(i, k, K)] * mat_B[idx_in_flattened(k, j, N)];
            }
            mat_C[idx_in_flattened(i, j, N)] = sum;
        }
    }
}
