//
// Created by andwh on 12/12/2024.
//

#ifndef NAIVE_MATMUL_CUH
#define NAIVE_MATMUL_CUH



__global__ void naive_matmul_kernel(
  const float* mat_A, const float* mat_B, float* mat_C, // we make matrix A and B const for compiler optimizations (caching mainly)
  int M, int N, int K
  );

void naive_matmul(
  const float* mat_A, const float* mat_B, float* mat_C,
  int M, int N, int K
  );



#endif //NAIVE_MATMUL_CUH
