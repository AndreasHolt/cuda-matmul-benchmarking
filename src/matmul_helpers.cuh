//
// Created by andwh on 12/12/2024.
//

#ifndef MATMUL_HELPERS_CUH
#define MATMUL_HELPERS_CUH


void alloc_matrices(
    float** mat_A, float** mat_B, float** mat_C, // Host matrices
    int m, int n, int k, // Dimensions
    float** d_mat_A, float** d_mat_B, float** d_mat_C // Device matrices
    );




#endif //MATMUL_HELPERS_CUH
