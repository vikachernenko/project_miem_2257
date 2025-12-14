#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void normalizeRowKernel(float* A, float* I, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float pivot = A[k * n + k];
        A[k * n + idx] /= pivot;
        I[k * n + idx] /= pivot;
    }
}

__global__ void eliminateRowsKernel(float* A, float* I, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        if (row != k) {
            float factor = A[row * n + k];

            float valA = A[k * n + col];
            float valI = I[k * n + col];
            
            A[row * n + col] -= factor * valA;
            I[row * n + col] -= factor * valI;
        }
    }
}

void readMatrix(const char* filename, std::vector<float>& matrix, int& n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    matrix.resize(n * n);
    file.read(reinterpret_cast<char*>(matrix.data()), n * n * sizeof(float));
    file.close();
}

void writeMatrix(const char* filename, const std::vector<float>& matrix, int n) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(&n), sizeof(int));
    file.write(reinterpret_cast<const char*>(matrix.data()), n * n * sizeof(float));
    file.close();
}

int main() {
    clock_t start = clock();

    int n;
    std::vector<float> h_A;
    std::cout << "Reading T.dat..." << std::endl;
    readMatrix("T.dat", h_A, n);
    
    std::vector<float> h_I(n * n, 0.0f);
    for (int i = 0; i < n; ++i) {
        h_I[i * n + i] = 1.0f;
    }

    float *d_A, *d_I;
    CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CHECK(cudaMalloc(&d_I, n * n * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_I, h_I.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize1D = 256;
    int gridSize1D = (n + blockSize1D - 1) / blockSize1D;

    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((n + blockSize2D.x - 1) / blockSize2D.x, (n + blockSize2D.y - 1) / blockSize2D.y);

    std::cout << "Inverting matrix (Gauss-Jordan)..." << std::endl;

    for (int k = 0; k < n; ++k) {
        normalizeRowKernel<<<gridSize1D, blockSize1D>>>(d_A, d_I, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        eliminateRowsKernel<<<gridSize2D, blockSize2D>>>(d_A, d_I, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    }


    std::vector<float> h_Result(n * n);
    CHECK(cudaMemcpy(h_Result.data(), d_I, n * n * sizeof(float), cudaMemcpyDeviceToHost));

  
    writeMatrix("M3.dat", h_Result, n);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_I));

    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Inversion Task Completed. Time: " << elapsed_secs << " seconds" << std::endl;

    return 0;
}