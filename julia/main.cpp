#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <immintrin.h> 
#include <omp.h>       
#include <string>
#include <cmath>
#include <algorithm>

// Подключаем скачанную библиотеку
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Конфигурация
const int WIDTH = 1024;
const int HEIGHT = 768;
const int MAX_ITER = 500;

struct FractalConfig {
    std::string name;
    double xmin, xmax, ymin, ymax;
    double jc_re, jc_im; 
};

// Функция для выделения выровненной памяти (важно для AVX)
template<typename T>
T* alloc_aligned(size_t size) {
    return static_cast<T*>(_mm_malloc(size * sizeof(T), 32));
}

// Генерация палитры
std::vector<uint32_t> generate_palette(int max_iter) {
    std::vector<uint32_t> palette(max_iter + 1);
    for (int i = 0; i < max_iter; ++i) {
        double t = (double)i / max_iter;
        uint8_t r = (uint8_t)(fmod(100.0 * t + i * 0.5, 256.0));
        uint8_t g = (uint8_t)(100.0 * t);
        uint8_t b = (uint8_t)(100.0 * (1.0 - t) + 200.0 * t);
        palette[i] = (0xFF << 24) | (b << 16) | (g << 8) | r;
    }
    palette[max_iter] = 0xFF000000; 
    return palette;
}

// Ядро вычислений (AVX2)
void compute_fractal_avx2(const FractalConfig& config, int* output) {
    double dx = (config.xmax - config.xmin) / WIDTH;
    double dy = (config.ymax - config.ymin) / HEIGHT;

    __m256d v_two = _mm256_set1_pd(2.0);
    __m256d v_four = _mm256_set1_pd(4.0);
    __m256d v_jc_re = _mm256_set1_pd(config.jc_re);
    __m256d v_jc_im = _mm256_set1_pd(config.jc_im);
    __m256d v_xmin = _mm256_set1_pd(config.xmin);
    __m256d v_dx = _mm256_set1_pd(dx);
    __m256d v_offsets = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < HEIGHT; ++py) {
        double y0 = config.ymin + py * dy;
        __m256d v_y0 = _mm256_set1_pd(y0);

        for (int px = 0; px < WIDTH; px += 4) {
            __m256d v_px = _mm256_set1_pd((double)px);
            __m256d v_x0 = _mm256_fmadd_pd(_mm256_add_pd(v_px, v_offsets), v_dx, v_xmin);

            __m256d z_re, z_im, c_re, c_im;

            if (config.name == "julia") {
                z_re = v_x0; z_im = v_y0;
                c_re = v_jc_re; c_im = v_jc_im;
            } else {
                z_re = _mm256_setzero_pd(); z_im = _mm256_setzero_pd();
                c_re = v_x0; c_im = v_y0;
            }

            __m256i v_iters = _mm256_setzero_si256();
            __m256i v_ones = _mm256_set1_epi64x(1);

            for (int i = 0; i < MAX_ITER; ++i) {
                if (config.name == "bship") {
                    static const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
                    z_re = _mm256_and_pd(z_re, sign_mask);
                    z_im = _mm256_and_pd(z_im, sign_mask);
                }

                __m256d z_re2 = _mm256_mul_pd(z_re, z_re);
                __m256d z_im2 = _mm256_mul_pd(z_im, z_im);
                __m256d sum_sq = _mm256_add_pd(z_re2, z_im2);

                __m256d mask = _mm256_cmp_pd(sum_sq, v_four, _CMP_LE_OQ);
                int mask_int = _mm256_movemask_pd(mask);
                if (mask_int == 0) break;

                __m256i v_mask_i = _mm256_castpd_si256(mask);
                __m256i inc = _mm256_and_si256(v_mask_i, v_ones);
                v_iters = _mm256_add_epi64(v_iters, inc);

                __m256d new_re = _mm256_add_pd(_mm256_sub_pd(z_re2, z_im2), c_re);
                __m256d new_im = _mm256_add_pd(_mm256_mul_pd(v_two, _mm256_mul_pd(z_re, z_im)), c_im);

                z_re = new_re;
                z_im = new_im;
            }

            long long temp_iters[4];
            _mm256_store_si256((__m256i*)temp_iters, v_iters);

            output[py * WIDTH + px + 0] = (int)temp_iters[3];
            output[py * WIDTH + px + 1] = (int)temp_iters[2];
            output[py * WIDTH + px + 2] = (int)temp_iters[1];
            output[py * WIDTH + px + 3] = (int)temp_iters[0];
        }
    }
}

void save_image(const std::string& filename, int* data, const std::vector<uint32_t>& palette) {
    std::vector<uint8_t> pixels(WIDTH * HEIGHT * 4); 

    #pragma omp parallel for
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        int iter = data[i];
        if (iter >= MAX_ITER) iter = MAX_ITER;
        
        uint32_t color = palette[iter];
        
        pixels[i * 4 + 0] = (color) & 0xFF;         
        pixels[i * 4 + 1] = (color >> 8) & 0xFF;    
        pixels[i * 4 + 2] = (color >> 16) & 0xFF;   
        pixels[i * 4 + 3] = 255;                    
    }

    stbi_write_png(filename.c_str(), WIDTH, HEIGHT, 4, pixels.data(), WIDTH * 4);
}

int main() {
    // Включаем потоки вручную, если переменная среды не задана
    omp_set_num_threads(omp_get_num_procs());
    
    int* iteration_data = alloc_aligned<int>(WIDTH * HEIGHT);
    auto palette = generate_palette(MAX_ITER);

    std::vector<FractalConfig> fractals = {
        {"mandelbrot", -2.0, 1.0, -1.2, 1.2, 0, 0},
        {"julia",      -1.6, 1.6, -1.2, 1.2, -0.7, 0.27015},
        {"bship",      -2.2, 1.2, -2.0, 1.0, 0, 0}
    };

    std::cout << "Starting Fractal Generation (C++ AVX2 Optimized)..." << std::endl;
    std::cout << "Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "========================================" << std::endl;

    for (const auto& f : fractals) {
        std::string filename = f.name + "_cpp.png";
        std::cout << "Processing: " << f.name << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        
        compute_fractal_avx2(f, iteration_data);
        auto calc_end = std::chrono::high_resolution_clock::now();

        save_image(filename, iteration_data, palette);
        auto save_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dur_calc = calc_end - start;
        std::chrono::duration<double> total = save_end - start;

        std::cout << " Done. Calc: " << dur_calc.count() << "s | Total: " << total.count() << "s" << std::endl;
    }

    _mm_free(iteration_data);
    std::cout << "All tasks completed." << std::endl;
    return 0;
}