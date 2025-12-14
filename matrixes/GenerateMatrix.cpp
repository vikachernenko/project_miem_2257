#include <iostream>
#include <fstream>
#include <vector>
#include <random>

int main() {
    const int MAX_N = 2000; // Matrix size
    std::vector<double> a(MAX_N - 1); // Lower diagonal
    std::vector<double> b(MAX_N);     // Main diagonal
    std::vector<double> c(MAX_N - 1); // Upper diagonal

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    // Fill diagonals (ensure diagonal dominance)
    for (int i = 0; i < MAX_N - 1; ++i) {
        a[i] = dist(gen);
        c[i] = dist(gen);
        b[i] = a[i] + c[i] + 100.0; // Ensure diagonal dominance
    }
    b[MAX_N - 1] = dist(gen) + 100.0;

    // Save to file
    std::ofstream out("T1.dat", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&MAX_N), sizeof(MAX_N));
    out.write(reinterpret_cast<const char*>(a.data()), a.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(c.data()), c.size() * sizeof(double));

    return 0;
}