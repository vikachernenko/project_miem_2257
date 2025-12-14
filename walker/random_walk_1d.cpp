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
    size_t n_walkers = 5000000;
    int n_steps = 1000;
    int threads_count = omp_get_max_threads();
    int N_sample = 10;

    if (argc >= 2) n_walkers = std::stoull(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4) threads_count = std::atoi(argv[3]);
    if (argc >= 5) N_sample = std::atoi(argv[4]);

    omp_set_num_threads(threads_count);

    std::cout << "1D Random Walk (CPU Optimized)" << std::endl;
    std::cout << "Walkers: " << n_walkers << ", Steps: " << n_steps 
              << ", Threads: " << threads_count << std::endl;

    std::vector<double> global_sum_pos(n_steps, 0.0);
    std::vector<double> global_sum_sq(n_steps, 0.0);
    std::vector<long long> global_hist(2 * n_steps + 1, 0);
    std::vector<int> sample_paths(N_sample * n_steps);

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::vector<double> local_sum(n_steps, 0.0);
        std::vector<double> local_sum_sq(n_steps, 0.0);
        std::vector<long long> local_hist(2 * n_steps + 1, 0);

        int tid = omp_get_thread_num();
        size_t total_threads = omp_get_num_threads();
        size_t walkers_per_thread = n_walkers / total_threads;
        size_t start_w = tid * walkers_per_thread;
        size_t end_w = (tid == total_threads - 1) ? n_walkers : start_w + walkers_per_thread;

        AvxRng rng(start_w + 123456789ULL);
        __m256i v_ones = _mm256_set1_epi64x(1);

        size_t w = start_w;
        for (; w <= end_w - 4; w += 4) {
            __m256i v_pos = _mm256_setzero_si256();

            for (int s = 0; s < n_steps; ++s) {
                __m256i r = rng.next();
                __m256i bit = _mm256_and_si256(r, v_ones);
                __m256i v_step = _mm256_sub_epi64(_mm256_slli_epi64(bit, 1), v_ones); 
                v_pos = _mm256_add_epi64(v_pos, v_step);

                long long pos_arr[4];
                _mm256_store_si256((__m256i*)pos_arr, v_pos);

                double p0 = (double)pos_arr[0];
                double p1 = (double)pos_arr[1];
                double p2 = (double)pos_arr[2];
                double p3 = (double)pos_arr[3];

                local_sum[s]    += p0 + p1 + p2 + p3;
                local_sum_sq[s] += p0*p0 + p1*p1 + p2*p2 + p3*p3;
            }

            long long final_pos[4];
            _mm256_store_si256((__m256i*)final_pos, v_pos);
            for(int i=0; i<4; ++i) local_hist[final_pos[i] + n_steps]++;
        }

        for (; w < end_w; ++w) {
            long long pos = 0;
            for (int s = 0; s < n_steps; ++s) {
                uint64_t r = _mm256_extract_epi64(rng.next(), 0);
                pos += (r & 1) ? 1 : -1;
                local_sum[s] += pos;
                local_sum_sq[s] += pos * pos;
            }
            local_hist[pos + n_steps]++;
        }

        #pragma omp critical
        {
            for (int s = 0; s < n_steps; ++s) {
                global_sum_pos[s] += local_sum[s];
                global_sum_sq[s] += local_sum_sq[s];
            }
            for (int i = 0; i < 2 * n_steps + 1; ++i) {
                global_hist[i] += local_hist[i];
            }
        }
    }

    if (N_sample > 0) {
        srand(12345);
        for (int w = 0; w < N_sample; ++w) {
            int pos = 0;
            for (int s = 0; s < n_steps; ++s) {
                pos += (rand() % 2) ? 1 : -1;
                sample_paths[w * n_steps + s] = pos;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "Done in " << duration.count() << " ms." << std::endl;

    FILE* f_sigma = fopen("sigma_1d.csv", "w");
    fprintf(f_sigma, "step,sigma_empirical,sigma_theory\n");
    for (int step = 0; step < n_steps; ++step) {
        double mean = global_sum_pos[step] / (double)n_walkers;
        double mean_sq = global_sum_sq[step] / (double)n_walkers;
        double var = mean_sq - mean * mean;
        if (var < 0) var = 0;
        fprintf(f_sigma, "%d,%.10g,%.10g\n", step + 1, sqrt(var), sqrt((double)step + 1.0));
    }
    fclose(f_sigma);

    FILE* f_hist = fopen("histogram_1d.csv", "w");
    fprintf(f_hist, "position,count\n");
    for (int i = 0; i < 2 * n_steps + 1; ++i) {
        if (global_hist[i] > 0) fprintf(f_hist, "%d,%lld\n", i - n_steps, global_hist[i]);
    }
    fclose(f_hist);

    if (N_sample > 0) {
        FILE* f_paths = fopen("sample_paths_1d.csv", "w");
        fprintf(f_paths, "walker,step,position\n");
        for (int w = 0; w < N_sample; ++w) {
            for (int s = 0; s < n_steps; ++s) {
                fprintf(f_paths, "%d,%d,%d\n", w, s, sample_paths[w * n_steps + s]);
            }
        }
        fclose(f_paths);
    }

    return 0;
}