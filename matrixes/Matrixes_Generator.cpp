#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <cstdlib>
#include <limits>

using namespace std;

// Safe calculation of the range midpoint
const int sft = RAND_MAX / 2 + (RAND_MAX % 2);

int main() {
    const int N = 2000;
    const size_t total_elements = N * N;
    const size_t size = total_elements * sizeof(double);
    
    // Allocate aligned memory for AVX
    double* M1 = static_cast<double*>(_mm_malloc(size, 32));
    double* M2 = static_cast<double*>(_mm_malloc(size, 32));

    if (!M1 || !M2) {
        cerr << "Memory allocation failed!" << endl;
        if (M1) _mm_free(M1);
        if (M2) _mm_free(M2);
        return 1;
    }

    // Generate random numbers
    for (size_t i = 0; i < total_elements; ++i) {
        M1[i] = rand() - sft;
        M2[i] = rand() - sft;
    }

    // Save matrices to files
    auto save_matrix = [N](const string& filename, double* matrix) {
        ofstream out(filename);
        if (!out) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        out << N << " // Matrix Order\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                out << matrix[i * N + j] << " ";
            }
            out << "\n";
        }
        return true;
    };

    save_matrix("M1.dat", M1);
    save_matrix("M2.dat", M2);

    _mm_free(M1);
    _mm_free(M2);
    return 0;
}