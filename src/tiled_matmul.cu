//
// Created by andwh on 16/12/2024.
//

#include "tiled_matmul.cuh"
#include "matmul_helpers.cuh"

template<int TILE_SIZE>
__global__ void tiled_matmul_kernel(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K)
{
    // shared memory for the 16x16 tiles
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // load tile from mat_A into shared memory
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            int a_row = row;
            int a_col = tile * TILE_SIZE + tx;
            shared_A[ty][tx] = mat_A[idx_in_flattened(a_row, a_col, K)];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        // load tile from mat_B into shared memory
        if (col < N && (tile * TILE_SIZE + ty) < K) {
            int b_row = tile * TILE_SIZE + ty;
            int b_col = col;
            shared_B[ty][tx] = mat_B[idx_in_flattened(b_row, b_col, N)];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();  // wait for all threads to load their data

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    // write result
    if (row < M && col < N) {
        mat_C[idx_in_flattened(row, col, N)] = sum;
    }
}

void tiled_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K)
{
    constexpr int TILE_SIZE = 16;

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );

    tiled_matmul_kernel<TILE_SIZE><<<grid_size, block_size>>>(
        mat_A, mat_B, mat_C, M, N, K);
}