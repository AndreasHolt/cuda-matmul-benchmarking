#include "coalesced_matmul.cuh"
#include "matmul_helpers.cuh"

template<int BLOCKSIZE = 16>
__global__ void coalesced_matmul_kernel(
    int M, int N, int K,
    float alpha,                    // scaling factor for matrices A and B
    const float* mat_A,
    const float* mat_B,
    float beta,                     // scaling factor for matrix C
    float* mat_C
)
{
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; k++) {
            sum += mat_A[idx(x, k, K)] * mat_B[idx(k, y, N)];


        }
        // alpha and beta scaling
        mat_C[idx(x, y, N)] = alpha * sum + beta * mat_C[idx(x, y, N)];

    }
}


void coalesced_matmul(
    const float* mat_A, const float* mat_B, float* mat_C,
    int M, int N, int K
)
{
    constexpr int BLOCKSIZE = 16;

    dim3 grid_size(
        (M + BLOCKSIZE - 1) / BLOCKSIZE,
        (N + BLOCKSIZE - 1) / BLOCKSIZE
    );

    // blockDim becomes 1D: total threads = BLOCKSIZE * BLOCKSIZE
    // we make it 1D instead as 2D for more straightforward mapping
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);

    // testing out scaling factors
    float alpha = 1.0f;
    float beta = 0.5f;

    coalesced_matmul_kernel<BLOCKSIZE><<<grid_size, block_size>>>(M, N, K, alpha, mat_A, mat_B, beta, mat_C);
}