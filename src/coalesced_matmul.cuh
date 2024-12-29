
#ifndef COALESCED_MATMUL_CUH
#define COALESCED_MATMUL_CUH

void coalesced_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K
);

#endif //COALESCED_MATMUL_CUH
