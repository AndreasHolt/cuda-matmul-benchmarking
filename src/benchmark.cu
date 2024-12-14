//
// Created by andwh on 14/12/2024.
//

#include "benchmark.cuh"

#include "matmul_helpers.cuh"
#include "naive_matmul.cuh"


BenchmarkResult benchmark_matmul(int M, int N, int K, int num_iterations) {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    alloc_matrices(&A, &B, &C, &d_A, &d_B, &d_C, M, N, K);
    initialize_matrices(A, B, M, N, K);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    naive_matmul(d_A, d_B, d_C, M, N, K); // we do a warmup run (we don't want to include how long it takes the gpu to spin up etc., gpu power state?)

    // we use cudaEvent to measure the time it takes in ms
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < num_iterations; i++) {
        naive_matmul(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= num_iterations;

    // calculate the GFLOPS
    // note that each output element needs 2*K FLOPs (each iter. does 1 multiply + 1 add)
    double total_flops = 2.0 * M * N * K;
    double gflops = (total_flops * 1e-9) / (milliseconds * 1e-3); // convert flops to glops. We just divide by a billion (1e-9). Also convert ms to sec

    // Verify correctness
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_correct = verify_against_cpu_matmul(A, B, C, M, N, K);

    // Cleanup
    free_matrices(A, B, C, d_A, d_B, d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {milliseconds, static_cast<float>(gflops), is_correct};
}
