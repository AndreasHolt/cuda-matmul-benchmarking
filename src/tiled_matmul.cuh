//
// Created by andwh on 16/12/2024.
//

#ifndef TILED_MATMUL_CUH
#define TILED_MATMUL_CUH

void tiled_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K);



#endif //TILED_MATMUL_CUH
