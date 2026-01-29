#include <iostream>
#include <cmath>

#define CUDA_CHECK(call)                                                                        \
do {                                                                                             \
    cudaError_t error = call;                                                                   \
    if (error != cudaSuccess) {                                                                 \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));                        \
        exit(1);                                                                                \
    }                                                                                           \
} while(0)

// CUDA kernel: adds two vectors element-wise
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    const int block_size = 256;
    
    // Allocate host memory
    float* h_A = (float*)malloc(N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));
    float* h_C = (float*)malloc(N * sizeof(float));
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid size
    int grid_size = (N + block_size - 1) / block_size;
    
    // Launch kernel
    vectorAdd<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "Vector Addition Results:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
