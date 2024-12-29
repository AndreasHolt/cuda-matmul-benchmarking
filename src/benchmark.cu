//
// Created by andwh on 14/12/2024.
//

#include "benchmark.cuh"
#include <chrono>

#include "cpu_matmul.cuh"
#include "matmul_helpers.cuh"
#include "naive_matmul.cuh"
#include "tiled_coalesced_matmul.cuh"
#include "tiled_matmul.cuh"
#include "coalesced_matmul.cuh"

const char* get_MatMulType_name(MatMulType type) {
    switch(type) {
        case MatMulType::SEQUENTIAL_CPU:
            return "SEQUENTIAL_CPU";
        case MatMulType::NAIVE_GPU:
            return "NAIVE_GPU";
        case MatMulType::TILED_GPU:
            return "TILED_GPU";
        case MatMulType::TILED_COALESCED_GPU:
            return "TILED_COALESCED_GPU";
        case MatMulType::COALESCED_GPU:
            return "COALESCED_GPU";
        default:
            return "UNKNOWN";
    }
}


BenchmarkResult benchmark_matmul(MatMulType type, int M, int N, int K, bool verify_correctness, int num_iterations) {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;


    if (type == MatMulType::SEQUENTIAL_CPU) { // as it's cpu, we just need host memory to be allocated
        A = new float[M * K];
        B = new float[K * N];
        C = new float[M * N];
        initialize_matrices(A, B, M, N, K);
    } else { // otherwise it's on gpu, i.e. we need to allocate on host and device, and copy to device
        alloc_matrices(&A, &B, &C, &d_A, &d_B, &d_C, M, N, K);
        initialize_matrices(A, B, M, N, K);
        cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    // variables for timing
    float milliseconds;

    if (type == MatMulType::SEQUENTIAL_CPU) {
        // as it's on cpu just use chrono to measure
        auto start = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < num_iterations; i++) {
            cpu_matmul(A, B, C, M, N, K);
        }
        auto end = std::chrono::high_resolution_clock::now();
        milliseconds = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    } else {
        // for on-gpu use cudaEvent to measure the time it takes in ms
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // first a warmup on the relevant kernel function
        switch(type) {
            case MatMulType::NAIVE_GPU: // for now i've only implemented naive matmul
                naive_matmul(d_A, d_B, d_C, M, N, K);
            case MatMulType::TILED_GPU:
                tiled_matmul(d_A, d_B, d_C, M, N, K);
            case MatMulType::TILED_COALESCED_GPU:
                tiled_coalesced_matmul(d_A, d_B, d_C, M, N, K);
            case MatMulType::COALESCED_GPU:
                coalesced_matmul(d_A, d_B, d_C, M, N, K);
            break;
        }

        cudaEventRecord(start);
        for(int i = 0; i < num_iterations; i++) {
            switch (type) {
                case MatMulType::NAIVE_GPU:
                    naive_matmul(d_A, d_B, d_C, M, N, K);
                    break;
                case MatMulType::TILED_GPU:
                    tiled_matmul(d_A, d_B, d_C, M, N, K);
                    break;
                case MatMulType::TILED_COALESCED_GPU:
                    tiled_coalesced_matmul(d_A, d_B, d_C, M, N, K);
                    break;
                case MatMulType::COALESCED_GPU:
                    coalesced_matmul(d_A, d_B, d_C, M, N, K);
                    break;
            }

        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        milliseconds /= num_iterations;


    }

    // calculate the GFLOPS
    // note that each output element needs 2*K FLOPs (each iter. does 1 multiply + 1 add)
    double total_flops = 2.0 * M * N * K;
    double gflops = (total_flops * 1e-9) / (milliseconds * 1e-3); // convert flops to glops. We just divide by a billion (1e-9). Also convert ms to sec

    // i decided to just verify correctness for the gpu versions, as the cpu version is trivially correct
    bool is_correct = true; // default to true since cpu will not enter below
    if (verify_correctness && type != MatMulType::SEQUENTIAL_CPU) {
        cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        is_correct = verify_against_cpu_matmul(type, A, B, C, M, N, K);
    }

    // for good measure, cleanup
    if (type == MatMulType::SEQUENTIAL_CPU) {
        delete[] A;
        delete[] B;
        delete[] C;
    } else {
        free_matrices(A, B, C, d_A, d_B, d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return {milliseconds, static_cast<float>(gflops), is_correct};
}


