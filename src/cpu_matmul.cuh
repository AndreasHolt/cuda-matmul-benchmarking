#ifndef CPU_MATMUL_CUH
#define CPU_MATMUL_CUH

void cpu_matmul(
    const float *mat_A, const float *mat_B, float *mat_C,
    int M, int N, int K
    );

#endif //CPU_MATMUL_CUH
