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

__global__ void matrixMultKernel(float* M1, float* M2, float* M3, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        M3[row * n + col] = sum;
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

    int n1, n2;
    std::vector<float> M1, M2;
    std::cout << "Reading M1..." << std::endl;
    readMatrix("M1.dat", M1, n1);
    std::cout << "Reading M2..." << std::endl;
    readMatrix("M2.dat", M2, n2);
    
    if (n1 != n2) {
        std::cerr << "Error: Matrix sizes don't match!" << std::endl;
        return 1;
    }
    const int n = n1;
    std::vector<float> M3(n * n);

    float *d_M1, *d_M2, *d_M3;
    CHECK(cudaMalloc(&d_M1, n * n * sizeof(float)));
    CHECK(cudaMalloc(&d_M2, n * n * sizeof(float)));
    CHECK(cudaMalloc(&d_M3, n * n * sizeof(float)));

    std::cout << "Copying data to GPU..." << std::endl;
    CHECK(cudaMemcpy(d_M1, M1.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_M2, M2.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    std::cout << "Executing kernel..." << std::endl;
    matrixMultKernel<<<gridSize, blockSize>>>(d_M1, d_M2, d_M3, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::cout << "Copying result back..." << std::endl;
    CHECK(cudaMemcpy(M3.data(), d_M3, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    writeMatrix("M3.dat", M3, n);

    CHECK(cudaFree(d_M1));
    CHECK(cudaFree(d_M2));
    CHECK(cudaFree(d_M3));

    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Multiplication Task Completed. Time: " << elapsed_secs << " seconds" << std::endl;

    return 0;
}