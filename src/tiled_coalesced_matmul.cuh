//
// Created by ucloud on 12/29/24.
//

#ifndef TILED_COALESCED_MATMUL_CUH
#define TILED_COALESCED_MATMUL_CUH

void tiled_coalesced_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K);


#endif //TILED_COALESCED_MATMUL_CUH
