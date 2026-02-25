#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Состояния ячеек
#define EMPTY 0  // Скалы, реки, пустота
#define TREE 1   // Здоровый лес
#define FIRE 2   // Горит
#define BURNT 3  // Пепел

__global__ void initCurand(curandState* states, unsigned long seed, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cells) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Ядро клеточного автомата (Double Buffering)
__global__ void fireStep(const int* grid_in, int* grid_out, int Nx, int Ny, int t, 
                         float omega, float r, float p_base, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = y * Nx + x;
    int state = grid_in[idx];
    curandState localState = states[idx];

    // Вычисляем сезонную вероятность спонтанного возгорания (Слайд 26)
    // sin^2(omega * t) дает колебания от 0 (зима) до 1 (лето)
    float p_ign = p_base * powf(sinf(omega * t), 2.0f);

    if (state == EMPTY || state == BURNT) {
        // Камни и пепел не меняются
        grid_out[idx] = state;
    } 
    else if (state == FIRE) {
        // Дерево сгорает за 1 шаг и превращается в пепел
        grid_out[idx] = BURNT;
    } 
    else if (state == TREE) {
        bool catches_fire = false;
        
        // 1. Проверка соседей (Фон Нейман, закрытые границы)
        int neighbors[4][2] = {{x, y-1}, {x, y+1}, {x-1, y}, {x+1, y}};
        for (int i = 0; i < 4; i++) {
            int nx = neighbors[i][0];
            int ny = neighbors[i][1];
            
            // Проверка закрытых границ
            if (nx >= 0 && nx < Nx && ny >= 0 && ny < Ny) {
                int n_idx = ny * Nx + nx;
                if (grid_in[n_idx] == FIRE) {
                    // Сосед горит! Кидаем кубик на заражение огнем
                    if (curand_uniform(&localState) < r) {
                        catches_fire = true;
                        break;
                    }
                }
            }
        }
        
        // 2. Спонтанное возгорание (молния/жара), если не загорелось от соседа
        if (!catches_fire) {
            if (curand_uniform(&localState) < p_ign) {
                catches_fire = true;
            }
        }

        // Записываем результат в ВЫХОДНОЙ буфер
        grid_out[idx] = catches_fire ? FIRE : TREE;
    }

    states[idx] = localState;
}

int main(int argc, char** argv) {
    // Параметры по умолчанию
    int Nx = 200, Ny = 200, steps = 300;
    float density = 0.65f; // 65% карты - лес, 35% - реки/камни
    float r = 0.3f;        // Вероятность распространения огня от соседа
    float p_base = 0.0001f; // Базовая вероятность удара молнии
    float omega = 0.1f;    // Частота смены сезонов (0.1 = быстрая смена)

    if (argc >= 8) {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
        density = std::atof(argv[3]);
        r = std::atof(argv[4]);
        p_base = std::atof(argv[5]);
        omega = std::atof(argv[6]);
        steps = std::atoi(argv[7]);
    } else {
        std::cout << "Usage: ./fire_sim <Nx> <Ny> <Density> <SpreadProb(r)> <IgniteProb> <Season(omega)> <Steps>\n";
        std::cout << "Using default parameters...\n\n";
    }

    int grid_size = Nx * Ny;
    std::cout << "--- FOREST FIRE MODEL ---\n";
    std::cout << "Grid: " << Nx << "x" << Ny << " | Forest Density: " << (density * 100) << "%\n";

    // Инициализация на CPU
    std::vector<int> h_grid(grid_size, EMPTY);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Сажаем лес с заданной плотностью (Fire breaks / barriers)
    for (int i = 0; i < grid_size; i++) {
        if (dis(gen) < density) {
            h_grid[i] = TREE;
        }
    }
    // Подожжем одно дерево в центре для затравки
    h_grid[(Ny/2) * Nx + (Nx/2)] = FIRE;

    // Память на GPU
    int *d_grid_in, *d_grid_out;
    curandState* d_states;
    cudaMalloc(&d_grid_in, grid_size * sizeof(int));
    cudaMalloc(&d_grid_out, grid_size * sizeof(int));
    cudaMalloc(&d_states, grid_size * sizeof(curandState));

    cudaMemcpy(d_grid_in, h_grid.data(), grid_size * sizeof(int), cudaMemcpyHostToDevice);

    // Настройка блоков CUDA (2D сетка)
    dim3 blockSize(16, 16);
    dim3 numBlocks((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

    int total_threads = Nx * Ny;
    int numBlocks1D = (total_threads + 256 - 1) / 256;
    initCurand<<<numBlocks1D, 256>>>(d_states, 4321, total_threads);

    // Файлы
    std::ofstream history("fire_history.csv");
    history << "# " << Nx << "," << Ny << "\n";
    history << "Step,Trees,Fires,Burnt\n";
    std::ofstream frames("fire_frames.bin", std::ios::binary);

    std::vector<char> frame_data(grid_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_gpu_time = 0;

    // Главный цикл (Time Loop)
    for (int t = 0; t < steps; t++) {
        // Замеряем GPU (Только логика)
        cudaEventRecord(start);
        
        // Запускаем ядро: читаем из IN, пишем в OUT
        fireStep<<<numBlocks, blockSize>>>(d_grid_in, d_grid_out, Nx, Ny, t, omega, r, p_base, d_states);
        
        // Меняем указатели местами (Double Buffering)
        int* temp = d_grid_in;
        d_grid_in = d_grid_out;
        d_grid_out = temp;
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_gpu_time += ms;

        // Копируем на CPU для статистики и записи
        cudaMemcpy(h_grid.data(), d_grid_in, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

        int count_tree = 0, count_fire = 0, count_burnt = 0;
        for (int i = 0; i < grid_size; i++) {
            frame_data[i] = h_grid[i];
            if (h_grid[i] == TREE) count_tree++;
            else if (h_grid[i] == FIRE) count_fire++;
            else if (h_grid[i] == BURNT) count_burnt++;
        }

        history << t << "," << count_tree << "," << count_fire << "," << count_burnt << "\n";
        frames.write(frame_data.data(), grid_size);

        // Если гореть больше нечему и молний нет, можно прервать, но из-за молний оставим цикл
    }

    std::cout << "\n=== PERFORMANCE BENCHMARK ===\n";
    std::cout << "Pure GPU Compute Time: " << total_gpu_time << " ms\n";
    std::cout << "Average time per step: " << (total_gpu_time / steps) << " ms\n";
    
    cudaFree(d_grid_in); cudaFree(d_grid_out); cudaFree(d_states);
    history.close(); frames.close();
    return 0;
}

// nvcc -O3 -arch=sm_86 fire.cu -o fire_sim -Xlinker /SUBSYSTEM:CONSOLE
// ./fire_sim.exe 200 200 0.65 0.3 0.0001 0.1 300
// Сетка 200x200, Плотность леса 65%, Распространение огня 30%, 300 шагов