//
// Created by andwh on 12/12/2024.
//

#include "naive_matmul.cuh"

#include "matmul_helpers.cuh"

__global__ void naive_matmul_kernel(
  const float* mat_A, const float* mat_B, float* mat_C, // we make matrix A and B const for compiler optimizations (caching mainly)
  int M, int N, int K
  )
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float cell_sum = 0.0f;

    for(int k = 0; k < K; k++) {
      cell_sum += mat_A[idx(row, k, K)] * mat_B[idx(k, col, N)];
    }

    mat_C[idx(row, col, N)] = cell_sum;
  }
}

void naive_matmul(
  const float* mat_A, const float* mat_B, float* mat_C,
  int M, int N, int K
  )
{
  dim3 block_size(16, 16);
  dim3 grid_size(
    (N + block_size.x - 1) / block_size.x, // trick for ceil division. ensure matrix N is covered by the grid of blocks
    (M + block_size.y - 1) / block_size.y
  ); // calculate grid size

  naive_matmul_kernel<<<grid_size, block_size>>>(mat_A, mat_B, mat_C, M, N, K);

}