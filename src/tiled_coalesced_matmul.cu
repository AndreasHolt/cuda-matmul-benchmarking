#include "tiled_coalesced_matmul.cuh"
#include "matmul_helpers.cuh"


template<int TILE_SIZE>
__global__ void tiled_coalesced_matmul_kernel(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K)
{
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    // we now store B transposed to enable coalesced global memory loads.
    // threads with consecutive thread indices will access consecutive memory locations when loading from matrix B
    __shared__ float shared_B_transposed[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            int a_row = row;
            int a_col = tile * TILE_SIZE + tx;
            shared_A[ty][tx] = mat_A[idx(a_row, a_col, K)];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        if ((tile * TILE_SIZE + ty) < K && col < N ) {
            int b_row = tile * TILE_SIZE + ty;
            int b_col = col;
            // we swap tx and ty. Each thread in the warp no longer accesses elements spaced by N
            shared_B_transposed[tx][ty] = mat_B[idx(b_row, b_col, N)];
        } else {
            shared_B_transposed[tx][ty] = 0.0f;
        }

        __syncthreads();  // wait for all threads to load their data

        for (int k = 0; k < TILE_SIZE; k++) {
            // we adjust the indexing for the transposed B matrix
            sum += shared_A[ty][k] * shared_B_transposed[tx][k];
        }

        __syncthreads();
    }

    // write result
    if (row < M && col < N) {
        mat_C[idx(row, col, N)] = sum;
    }
}

void tiled_coalesced_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K)
{
    constexpr int TILE_SIZE = 16;

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );

    tiled_coalesced_matmul_kernel<TILE_SIZE><<<grid_size, block_size>>>(
        mat_A, mat_B, mat_C, M, N, K);
}