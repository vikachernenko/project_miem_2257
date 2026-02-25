#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DEAD 0
#define HUMAN 1
#define ZOMBIE 2

struct Agent {
    int id, type, x, y;
};

__global__ void initCurand(curandState* states, unsigned long seed, int max_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_agents) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void moveAgents(Agent* agents, int* grid, int num_agents, int Nx, int Ny, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    Agent a = agents[idx];
    if (a.type == DEAD) return;

    curandState localState = states[idx];
    int dir = curand(&localState) % 5; 
    
    int nx = a.x, ny = a.y;
    if (dir == 1) ny = (a.y - 1 + Ny) % Ny;
    else if (dir == 2) ny = (a.y + 1) % Ny;
    else if (dir == 3) nx = (a.x - 1 + Nx) % Nx;
    else if (dir == 4) nx = (a.x + 1) % Nx;

    int target_idx = ny * Nx + nx;
    int old = atomicCAS(&grid[target_idx], 0, idx + 1);

    if (old == 0 || old == idx + 1) {
        if (dir != 0 && old == 0) atomicCAS(&grid[a.y * Nx + a.x], idx + 1, 0);
        agents[idx].x = nx; agents[idx].y = ny;
    } else {
        atomicCAS(&grid[a.y * Nx + a.x], 0, idx + 1);
    }
    states[idx] = localState;
}

__global__ void interact(Agent* agents, int* grid, int num_agents, int Nx, int Ny, float p, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    Agent me = agents[idx];
    if (me.type != HUMAN) return;

    int neighbors[4] = {
        ((me.y - 1 + Ny) % Ny) * Nx + me.x,
        ((me.y + 1) % Ny) * Nx + me.x,
        me.y * Nx + ((me.x - 1 + Nx) % Nx),
        me.y * Nx + ((me.x + 1) % Nx)
    };

    curandState localState = states[idx];
    for (int i = 0; i < 4; i++) {
        int n_id = grid[neighbors[i]] - 1;
        if (n_id >= 0 && agents[n_id].type == ZOMBIE) {
            float r = curand_uniform(&localState);
            if (r < p) {
                agents[n_id].type = DEAD;
                grid[neighbors[i]] = 0; 
            } else {
                agents[idx].type = ZOMBIE;
                break;
            }
        }
    }
    states[idx] = localState;
}

int main(int argc, char** argv) {
    // 1. ЧТЕНИЕ ПАРАМЕТРОВ ПОЛЬЗОВАТЕЛЯ ИЛИ ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
    int Nx = 100, Ny = 100, NL = 800, NZ = 20, steps = 500;
    float p = 0.4f;

    if (argc >= 7) {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
        NL = std::atoi(argv[3]);
        NZ = std::atoi(argv[4]);
        p = std::atof(argv[5]);
        steps = std::atoi(argv[6]);
    } else {
        std::cout << "Usage: ./zombie_sim <Nx> <Ny> <Humans> <Zombies> <p> <steps>\n";
        std::cout << "Using default parameters...\n\n";
    }

    std::cout << "--- ZOMBIE APOCALYPSE ---\n";
    std::cout << "Grid: " << Nx << "x" << Ny << " | Humans: " << NL << " | Zombies: " << NZ << "\n";
    std::cout << "Win probability (p): " << p << " | Max steps: " << steps << "\n";

    int total_agents = NL + NZ;
    int grid_size = Nx * Ny;

    if (total_agents > grid_size) {
        std::cerr << "Error: Too many agents for this grid size!\n";
        return -1;
    }

    std::vector<Agent> h_agents(total_agents);
    std::vector<int> h_grid(grid_size, 0);

    std::vector<int> positions(grid_size);
    for (int i = 0; i < grid_size; i++) positions[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(positions.begin(), positions.end(), g);

    for (int i = 0; i < total_agents; i++) {
        h_agents[i].id = i;
        h_agents[i].type = (i < NL) ? HUMAN : ZOMBIE;
        h_agents[i].x = positions[i] % Nx;
        h_agents[i].y = positions[i] / Nx;
        h_grid[h_agents[i].y * Nx + h_agents[i].x] = i + 1;
    }

    Agent* d_agents; int* d_grid; curandState* d_states;
    cudaMalloc(&d_agents, total_agents * sizeof(Agent));
    cudaMalloc(&d_grid, grid_size * sizeof(int));
    cudaMalloc(&d_states, total_agents * sizeof(curandState));

    cudaMemcpy(d_agents, h_agents.data(), total_agents * sizeof(Agent), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, h_grid.data(), grid_size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (total_agents + blockSize - 1) / blockSize;

    initCurand<<<numBlocks, blockSize>>>(d_states, 1234, total_agents);

    std::ofstream history("history.csv");
    history << "# " << Nx << "," << Ny << "\n"; // Передаем размеры сетки в Python
    history << "Step,Humans,Zombies\n";
    std::ofstream frames("frames.bin", std::ios::binary);
    std::vector<char> frame_data(grid_size);

    // 2. НАСТРОЙКА ТАЙМЕРОВ CUDA
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float total_gpu_time_ms = 0;

    int actual_steps = 0;

    for (int t = 0; t < steps; t++) {
        // Замеряем ТОЛЬКО работу видеокарты, без записи на диск
        cudaEventRecord(start_event);
        moveAgents<<<numBlocks, blockSize>>>(d_agents, d_grid, total_agents, Nx, Ny, d_states);
        interact<<<numBlocks, blockSize>>>(d_agents, d_grid, total_agents, Nx, Ny, p, d_states);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float current_step_time;
        cudaEventElapsedTime(&current_step_time, start_event, stop_event);
        total_gpu_time_ms += current_step_time;

        // Копируем данные для записи (File I/O)
        cudaMemcpy(h_agents.data(), d_agents, total_agents * sizeof(Agent), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_grid.data(), d_grid, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

        int count_h = 0, count_z = 0;
        std::fill(frame_data.begin(), frame_data.end(), 0);

        for (int i = 0; i < total_agents; i++) {
            if (h_agents[i].type == HUMAN) count_h++;
            else if (h_agents[i].type == ZOMBIE) count_z++;
            
            if (h_agents[i].type != DEAD) {
                frame_data[h_agents[i].y * Nx + h_agents[i].x] = h_agents[i].type;
            }
        }

        history << t << "," << count_h << "," << count_z << "\n";
        frames.write(frame_data.data(), grid_size);
        actual_steps++;

        if (count_h == 0 || count_z == 0) {
            std::cout << "Simulation ended naturally at step " << t << ".\n";
            break;
        }
    }

    std::cout << "\n=== PERFORMANCE BENCHMARK ===\n";
    std::cout << "Pure GPU Compute Time: " << total_gpu_time_ms << " ms\n";
    std::cout << "Average time per step: " << (total_gpu_time_ms / actual_steps) << " ms\n";
    std::cout << "=============================\n";

    cudaFree(d_agents); cudaFree(d_grid); cudaFree(d_states);
    history.close(); frames.close();
    return 0;
}

// nvcc -O3 -arch=sm_86 zombie.cu -o zombie_sim -Xlinker /SUBSYSTEM:CONSOLE
// zombie_sim.exe <Ширина> <Высота> <Люди> <Зомби> <Шанс_Победы_Людей> <Макс_Шагов>
// ./zombie_sim.exe 150 150 1500 50 0.1 800
// python visualize.py
