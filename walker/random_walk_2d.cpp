#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <cstdio>
#include <string>

struct AvxRng {
    __m256i s0, s1;

    AvxRng(uint64_t seed) {
        auto seed_gen = [&](int offset) {
            uint64_t z = (seed + offset * 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        s0 = _mm256_set_epi64x(seed_gen(1), seed_gen(2), seed_gen(3), seed_gen(4));
        s1 = _mm256_set_epi64x(seed_gen(5), seed_gen(6), seed_gen(7), seed_gen(8));
    }

    inline __m256i next() {
         __m256i x = s0;
         __m256i y = s1;
         s0 = y;
         x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));
         s1 = _mm256_xor_si256(_mm256_xor_si256(x, y), _mm256_xor_si256(_mm256_srli_epi64(x, 17), _mm256_srli_epi64(y, 26)));
         return _mm256_add_epi64(s1, y);
    }
};

int main(int argc, char** argv) {
    size_t n_walkers = 1000000;
    int n_steps = 1000;
    int threads_count = omp_get_max_threads();
    int bins = 100;
    int N_sample = 10;

    if (argc >= 2) n_walkers = std::stoull(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4) threads_count = std::atoi(argv[3]);
    if (argc >= 5) bins = std::atoi(argv[4]);
    if (argc >= 6) N_sample = std::atoi(argv[5]);

    omp_set_num_threads(threads_count);

    std::cout << "2D Random Walk (CPU Optimized)" << std::endl;
    std::cout << "Walkers: " << n_walkers << ", Steps: " << n_steps 
              << ", Bins: " << bins << ", Threads: " << threads_count << std::endl;

    std::vector<double> global_sum_r2(n_steps, 0.0);
    std::vector<unsigned long long> global_hist(bins, 0);
    std::vector<int> sample_paths_x(N_sample * n_steps);
    std::vector<int> sample_paths_y(N_sample * n_steps);

    double max_r = sqrt(2.0 * n_steps); 

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::vector<double> local_sum_r2(n_steps, 0.0);
        std::vector<unsigned long long> local_hist(bins, 0);

        int tid = omp_get_thread_num();
        size_t total_threads = omp_get_num_threads();
        size_t walkers_per_thread = n_walkers / total_threads;
        size_t start_w = tid * walkers_per_thread;
        size_t end_w = (tid == total_threads - 1) ? n_walkers : start_w + walkers_per_thread;

        AvxRng rng(start_w + 987654321ULL);
        __m256i v_ones = _mm256_set1_epi64x(1);

        size_t w = start_w;
        for (; w <= end_w - 4; w += 4) {
            __m256i v_x = _mm256_setzero_si256();
            __m256i v_y = _mm256_setzero_si256();

            for (int s = 0; s < n_steps; ++s) {
                __m256i r1 = rng.next(); 
                __m256i r2 = rng.next();

                __m256i bit_x = _mm256_and_si256(r1, v_ones);
                __m256i step_x = _mm256_sub_epi64(_mm256_slli_epi64(bit_x, 1), v_ones);
                v_x = _mm256_add_epi64(v_x, step_x);

                __m256i bit_y = _mm256_and_si256(r2, v_ones);
                __m256i step_y = _mm256_sub_epi64(_mm256_slli_epi64(bit_y, 1), v_ones);
                v_y = _mm256_add_epi64(v_y, step_y);

                long long x_arr[4], y_arr[4];
                _mm256_store_si256((__m256i*)x_arr, v_x);
                _mm256_store_si256((__m256i*)y_arr, v_y);

                for(int k=0; k<4; ++k) {
                    double r2 = (double)(x_arr[k]*x_arr[k] + y_arr[k]*y_arr[k]);
                    local_sum_r2[s] += r2;
                }
            }

            long long x_fin[4], y_fin[4];
            _mm256_store_si256((__m256i*)x_fin, v_x);
            _mm256_store_si256((__m256i*)y_fin, v_y);

            for(int k=0; k<4; ++k) {
                double r = sqrt((double)(x_fin[k]*x_fin[k] + y_fin[k]*y_fin[k]));
                int bin = (int)(r / max_r * bins);
                if (bin < 0) bin = 0;
                if (bin >= bins) bin = bins - 1;
                local_hist[bin]++;
            }
        }

        for (; w < end_w; ++w) {
            long long x = 0, y = 0;
            for (int s = 0; s < n_steps; ++s) {
                uint64_t r = _mm256_extract_epi64(rng.next(), 0);
                x += (r & 1) ? 1 : -1;
                y += (r & 2) ? 1 : -1;
                local_sum_r2[s] += (double)(x*x + y*y);
            }
            double r = sqrt((double)(x*x + y*y));
            int bin = (int)(r / max_r * bins);
            if(bin >= bins) bin = bins-1; else if(bin < 0) bin=0;
            local_hist[bin]++;
        }

        #pragma omp critical
        {
            for (int s = 0; s < n_steps; ++s) global_sum_r2[s] += local_sum_r2[s];
            for (int b = 0; b < bins; ++b) global_hist[b] += local_hist[b];
        }
    }

    if (N_sample > 0) {
        srand(42);
        for(int i=0; i<N_sample; ++i) {
            int x = 0, y = 0;
            for(int s=0; s<n_steps; ++s) {
                x += (rand() % 2) ? 1 : -1;
                y += (rand() % 2) ? 1 : -1;
                sample_paths_x[i*n_steps + s] = x;
                sample_paths_y[i*n_steps + s] = y;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end_time - start_time;
    std::cout << "Done in " << dur.count() << " ms." << std::endl;

    FILE *f_sigma = fopen("sigma_2d.csv", "w");
    fprintf(f_sigma, "step,sigma_empirical,sigma_theory\n");
    for (int s = 0; s < n_steps; ++s) {
        double mean_r2 = global_sum_r2[s] / n_walkers;
        fprintf(f_sigma, "%d,%.10g,%.10g\n", s + 1, sqrt(mean_r2), sqrt(2.0 * (s + 1)));
    }
    fclose(f_sigma);

    FILE *f_hist = fopen("histogram_2d.csv", "w");
    fprintf(f_hist, "bin,r_min,r_max,count\n");
    for (int b = 0; b < bins; ++b) {
        double rmin = (double)b * max_r / bins;
        double rmax = (double)(b + 1) * max_r / bins;
        fprintf(f_hist, "%d,%.5f,%.5f,%llu\n", b, rmin, rmax, global_hist[b]);
    }
    fclose(f_hist);

    if (N_sample > 0) {
        FILE *f_paths = fopen("sample_paths_2d.csv", "w");
        fprintf(f_paths, "walker,step,x,y\n");
        for (int w = 0; w < N_sample; ++w) {
            for (int s = 0; s < n_steps; ++s) {
                fprintf(f_paths, "%d,%d,%d,%d\n", w, s, 
                        sample_paths_x[w*n_steps + s], 
                        sample_paths_y[w*n_steps + s]);
            }
        }
        fclose(f_paths);
    }

    return 0;
}