#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>

void writeToFile(const char* filename, const std::vector<float>& matrix, int n) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(&n), sizeof(int));
    file.write(reinterpret_cast<const char*>(matrix.data()), n * n * sizeof(float));
    file.close();
    std::cout << "Generated " << filename << " (" << n << "x" << n << ")" << std::endl;
}

void generateDenseMatrix(const char* filename, int n) {
    std::vector<float> matrix(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int i = 0; i < n * n; ++i) {
        matrix[i] = dist(gen);
    }
    writeToFile(filename, matrix, n);
}

// Генерация тридиагональной матрицы с диагональным преобладанием
void generateTridiagonalMatrix(const char* filename, int n) {
    std::vector<float> matrix(n * n, 0.0f); // Заполняем нулями
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 5.0f);

    for (int i = 0; i < n; ++i) {
        float diagVal = 0.0f;
        
        // Левый сосед
        if (i > 0) {
            float val = dist(gen);
            matrix[i * n + (i - 1)] = val;
            diagVal += std::abs(val);
        }
        // Правый сосед
        if (i < n - 1) {
            float val = dist(gen);
            matrix[i * n + (i + 1)] = val;
            diagVal += std::abs(val);
        }

        // Диагональ должна быть больше суммы соседей (диагональное преобладание)
        matrix[i * n + i] = diagVal + dist(gen) + 2.0f; 
    }
    writeToFile(filename, matrix, n);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    int n = std::atoi(argv[1]);
    
    generateDenseMatrix("M1.dat", n);
    generateDenseMatrix("M2.dat", n);
    generateTridiagonalMatrix("T.dat", n); // Для задания 3

    return 0;
}