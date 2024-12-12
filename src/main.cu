#include <iostream>
#include "./matmul_helpers.cuh"
#include "naive_matmul.cuh"



int main() {
  int M = 3;  // rows of A
  int N = 2;  // cols of B
  int K = 2;  // cols of A and rows of B

  // Host matrices (flat arrays)
  float *A, *B, *C;
  // Device matrices
  float *d_A, *d_B, *d_C;

  alloc_matrices(&A, &B, &C, &d_A, &d_B, &d_C, M, N, K);

  // manual initialization of matrix A
  A[0] = 1.0f; A[1] = 2.0f;  // row 1
  A[2] = 4.0f; A[3] = 5.0f;  // row 2
  A[4] = 6.0f; A[5] = 3.0f;  // row 3

  // and matrix B
  B[0] = 1.0f; B[1] = 2.0f; // row 1
  B[2] = 3.0f; B[3] = 4.0f; // row 2

  // Print input matrices
  std::cout << "matrix A (3x2):\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      std::cout << A[i * K + j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "\nmatrix B (2x2):\n";
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << B[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);


  naive_matmul(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Print result matrix
  std::cout << "\nResult Matrix C (3x2):\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << C[i * N + j] << " ";
    }
    std::cout << "\n";
  }

  int is_correct = verify_against_cpu_matmul(A, B, C, M, N, K);
  if(is_correct) {
    std::cout << "Correct matmul kernel" << std::endl;

  } else {
    std::cout << "Incorrect matmul kernel" << std::endl;
  }



  return 0;
}

