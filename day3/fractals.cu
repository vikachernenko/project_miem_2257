#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

// Параметры изображения
const int WIDTH = 4096;
const int HEIGHT = 4096;
const int MAX_ITER = 1000;

// Макрос для проверки ошибок CUDA
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ----------------------------------------------------------------------
// CUDA Ядра (Kernels)
// ----------------------------------------------------------------------

// 1. Фрактал Мандельброта
__global__ void mandelbrotKernel(int* d_out, int w, int h, float xmin, float xmax, float ymin, float ymax) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        // Преобразование координат пикселя в координаты комплексной плоскости
        float c_re = xmin + (float)px / (float)w * (xmax - xmin);
        float c_im = ymin + (float)py / (float)h * (ymax - ymin);
        
        float z_re = 0.0f;
        float z_im = 0.0f;
        
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;
            
            if (z_re2 + z_im2 > 4.0f) break;
            
            // z = z^2 + c
            z_im = 2.0f * z_re * z_im + c_im;
            z_re = z_re2 - z_im2 + c_re;
        }
        d_out[py * w + px] = iter;
    }
}

// 2. Фрактал Джулиа (для константы C)
__global__ void juliaKernel(int* d_out, int w, int h, float xmin, float xmax, float ymin, float ymax, float c_re_const, float c_im_const) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        // В Джулии C фиксировано, а Z меняется от координат
        float z_re = xmin + (float)px / (float)w * (xmax - xmin);
        float z_im = ymin + (float)py / (float)h * (ymax - ymin);
        
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;
            
            if (z_re2 + z_im2 > 4.0f) break;
            
            // z = z^2 + C_const
            z_im = 2.0f * z_re * z_im + c_im_const;
            z_re = z_re2 - z_im2 + c_re_const;
        }
        d_out[py * w + px] = iter;
    }
}

// 3. Фрактал "Горящий Корабль" (Burning Ship)
// Формула: z = (|Re(z)| + i|Im(z)|)^2 + c
__global__ void burningShipKernel(int* d_out, int w, int h, float xmin, float xmax, float ymin, float ymax) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < w && py < h) {
        float c_re = xmin + (float)px / (float)w * (xmax - xmin);
        float c_im = ymin + (float)py / (float)h * (ymax - ymin);
        
        float z_re = 0.0f;
        float z_im = 0.0f;
        
        int iter = 0;
        for (iter = 0; iter < MAX_ITER; ++iter) {
            float z_re2 = z_re * z_re;
            float z_im2 = z_im * z_im;
            
            if (z_re2 + z_im2 > 4.0f) break;
            
            // z_next = (|x| + i|y|)^2 + c
            // Real: |x|^2 - |y|^2 + c_re
            // Imag: 2*|x|*|y| + c_im
            
            float next_im = 2.0f * fabsf(z_re) * fabsf(z_im) + c_im;
            float next_re = z_re2 - z_im2 + c_re;
            
            z_re = next_re;
            z_im = next_im;
        }
        d_out[py * w + px] = iter;
    }
}

// ----------------------------------------------------------------------
// Вспомогательные функции
// ----------------------------------------------------------------------

// Функция сохранения изображения в формат PPM (P6)
// Также здесь применяется палитра ("Smooth coloring" аппроксимация)
void saveImagePPM(const char* filename, const std::vector<int>& data, int w, int h) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return;
    }

    // Заголовок PPM: P6 <width> <height> <max_val>
    file << "P6\n" << w << " " << h << "\n255\n";

    std::vector<unsigned char> pixels(w * h * 3);
    
    for (int i = 0; i < w * h; ++i) {
        int iter = data[i];
        unsigned char r, g, b;

        if (iter >= MAX_ITER) {
            // Точка внутри множества - черный цвет
            r = 0; g = 0; b = 0;
        } else {
            // Точка снаружи - генерируем красивый цвет
            // Используем синусоиды для плавных переходов
            double t = (double)iter / 100.0; // масштаб цвета
            r = (unsigned char)(9.0 * (1.0 - t) * t * t * t * 255.0);
            g = (unsigned char)(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0);
            b = (unsigned char)(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);
            
            // Альтернативная простая палитра для яркости:
            // r = (unsigned char)(sin(0.3 * iter + 0) * 127 + 128);
            // g = (unsigned char)(sin(0.3 * iter + 2) * 127 + 128);
            // b = (unsigned char)(sin(0.3 * iter + 4) * 127 + 128);
            
            // Палитра как в примере Go (примерно):
            double t_go = (double)iter / (double)MAX_ITER;
            r = (unsigned char)((int)(100.0 * t_go + iter * 0.5) % 256);
            g = (unsigned char)(100.0 * t_go);
            b = (unsigned char)(100.0 * (1.0 - t_go) + 200.0 * t_go);
        }
        
        pixels[i * 3 + 0] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }

    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    file.close();
}

// Структура для возврата замеров времени
struct BenchmarkResult {
    double calcTime;
    double saveTime;
};

// Универсальная функция запуска
BenchmarkResult runFractal(std::string name, int* d_out, std::vector<int>& h_out) {
    clock_t start_calc = clock();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    if (name == "Mandelbrot") {
        mandelbrotKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -2.0f, 1.0f, -1.5f, 1.5f);
    } 
    else if (name == "Julia") {
        // Константа как в примере Go: -0.7 + 0.27015i
        juliaKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -1.6f, 1.6f, -1.6f, 1.6f, -0.7f, 0.27015f);
    } 
    else if (name == "BurningShip") {
         burningShipKernel<<<gridSize, blockSize>>>(d_out, WIDTH, HEIGHT, -2.2f, 1.2f, -2.0f, 1.0f);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize()); // Ждем завершения вычислений
    
    // Копируем данные обратно
    CHECK(cudaMemcpy(h_out.data(), d_out, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));
    
    clock_t end_calc = clock();

    clock_t start_save = clock();
    std::string filename = name + ".ppm";
    saveImagePPM(filename.c_str(), h_out, WIDTH, HEIGHT);
    clock_t end_save = clock();

    BenchmarkResult res;
    res.calcTime = double(end_calc - start_calc) / CLOCKS_PER_SEC;
    res.saveTime = double(end_save - start_save) / CLOCKS_PER_SEC;
    return res;
}

int main() {
    int *d_out;
    size_t size = WIDTH * HEIGHT * sizeof(int);
    
    // Выделяем память на GPU один раз
    CHECK(cudaMalloc(&d_out, size));
    
    std::vector<int> h_out(WIDTH * HEIGHT);

    std::cout << "Starting Fractal Benchmark (" << WIDTH << "x" << HEIGHT << ")..." << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    printf("%-15s | %-12s | %-12s | %-12s\n", "Fractal", "Calc Time(s)", "Save Time(s)", "Total(s)");
    std::cout << "--------------------------------------------------------" << std::endl;

    std::vector<std::string> fractals = {"Mandelbrot", "Julia", "BurningShip"};

    for (const auto& name : fractals) {
        BenchmarkResult res = runFractal(name, d_out, h_out);
        double total = res.calcTime + res.saveTime;
        
        printf("%-15s | %-12.3f | %-12.3f | %-12.3f\n", 
               name.c_str(), res.calcTime, res.saveTime, total);
    }

    CHECK(cudaFree(d_out));

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Done. Results saved as .ppm files." << std::endl;
    return 0;
}