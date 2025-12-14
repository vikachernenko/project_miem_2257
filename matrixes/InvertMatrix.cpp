#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "Timer.h" 

int main() {
    // USER CAN MODIFY THIS VALUE BASED ON THEIR CPU
    const int NUM_THREADS = 12;
    omp_set_num_threads(NUM_THREADS);
    
    start_timer();  

    // Read matrix from file
    std::ifstream in("T1.dat", std::ios::binary);
    if (!in) {
        std::cerr << "Error opening T1.dat" << std::endl;
        return 1;
    }

    int n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<double> a(n - 1), b(n), c(n - 1);
    in.read(reinterpret_cast<char*>(a.data()), (n - 1) * sizeof(double));
    in.read(reinterpret_cast<char*>(b.data()), n * sizeof(double));
    in.read(reinterpret_cast<char*>(c.data()), (n - 1) * sizeof(double));
    in.close();

    // Precompute coefficients
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> denom(n, 0.0);

    denom[0] = b[0];
    if (n > 1) {
        c_prime[0] = c[0] / denom[0];
        for (int i = 1; i < n - 1; ++i) {
            denom[i] = b[i] - a[i - 1] * c_prime[i - 1];
            c_prime[i] = c[i] / denom[i];
        }
        denom[n - 1] = b[n - 1] - a[n - 2] * c_prime[n - 2];
    } else {
        denom[0] = b[0];
    }

    // Create T2.dat with memory-mapped I/O
    int fd = open("T2.dat", O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd == -1) {
        std::cerr << "Error creating T2.dat" << std::endl;
        return 1;
    }

    // Set file size
    size_t file_size = sizeof(int) + static_cast<size_t>(n) * n * sizeof(double);
    if (ftruncate(fd, file_size) == -1) {
        close(fd);
        std::cerr << "Error setting file size" << std::endl;
        return 1;
    }

    // Memory mapping
    void* mapped = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        std::cerr << "Memory mapping failed" << std::endl;
        return 1;
    }

    // Write matrix size
    *reinterpret_cast<int*>(mapped) = n;
    double* data = reinterpret_cast<double*>(static_cast<char*>(mapped) + sizeof(int));

    // Parallel matrix inversion
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; ++j) {
        std::vector<double> d(n, 0.0);
        std::vector<double> d_prime(n, 0.0);
        std::vector<double> x(n, 0.0);

        d[j] = 1.0; // Right-hand side (identity matrix)

        // Forward sweep
        d_prime[0] = d[0] / denom[0];
        for (int i = 1; i < n; ++i) {
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom[i];
        }

        // Backward sweep
        x[n - 1] = d_prime[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }

        // Write column to memory
        double* column = data + j * n;
        for (int i = 0; i < n; ++i) {
            column[i] = x[i];
        }
    }

    // Cleanup
    munmap(mapped, file_size);
    close(fd);

    // Stop timer and get elapsed time
    double elapsed = stop_timer();
    std::cout << "Matrix inversion completed in: " << elapsed << " seconds" << std::endl;
    std::cout << "Threads used: " << NUM_THREADS << std::endl;

    return 0;
}