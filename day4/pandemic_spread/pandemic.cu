#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SUSCEPTIBLE 0
#define INFECTED 1
#define RECOVERED 2

// Инициализация генераторов на GPU
__global__ void initCurand(curandState* states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) curand_init(seed, idx, 0, &states[idx]);
}

// Ядро эпидемии (Double Buffering)
__global__ void epidemicStep(const int* state_in, int* state_out, 
                             const int* offsets, const int* neighbors, 
                             int N, float lambda, float gamma, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int my_state = state_in[idx];
    curandState localState = states[idx];

    if (my_state == RECOVERED) {
        state_out[idx] = RECOVERED;
    } 
    else if (my_state == INFECTED) {
        // Шанс выздороветь (gamma)
        if (curand_uniform(&localState) < gamma) {
            state_out[idx] = RECOVERED;
        } else {
            state_out[idx] = INFECTED;
        }
    } 
    else if (my_state == SUSCEPTIBLE) {
        bool gets_infected = false;
        
        // Достаем моих соседей из плоского массива графа
        int start_idx = offsets[idx];
        int end_idx = offsets[idx + 1];

        // Проходим по всем контактам человека
        for (int i = start_idx; i < end_idx; i++) {
            int neighbor_id = neighbors[i];
            
            // Если контакт болен, он может меня заразить с вероятностью lambda
            if (state_in[neighbor_id] == INFECTED) {
                if (curand_uniform(&localState) < lambda) {
                    gets_infected = true;
                    break;
                }
            }
        }
        
        state_out[idx] = gets_infected ? INFECTED : SUSCEPTIBLE;
    }

    states[idx] = localState;
}

int main(int argc, char** argv) {
    // Параметры по заданию (Слайд 42)
    int N = 10000;         // Количество людей (узлов графа)
    int K = 4;             // Изначально каждый знает 4 соседей
    float beta = 0.1f;     // Вероятность дальних связей (Rewiring)
    float lambda = 0.3f;   // Вероятность заражения при контакте
    float gamma = 0.1f;    // Вероятность выздоровления за 1 день
    int max_days = 200;    // Максимальное время симуляции
    std::string out_file = "pandemic_history.csv"; // Имя файла

    if (argc >= 7) {
        N = std::atoi(argv[1]);
        beta = std::atof(argv[2]);
        lambda = std::atof(argv[3]);
        gamma = std::atof(argv[4]);
        max_days = std::atoi(argv[5]);
        out_file = argv[6];
    } else {
        std::cout << "Usage: ./pandemic_sim <N> <Beta> <Lambda> <Gamma> <Days> <OutFile.csv>\n";
        std::cout << "Using default parameters...\n\n";
    }

    std::cout << "--- PANDEMIC SPREAD (Small World Network) ---\n";
    std::cout << "Population: " << N << " | Beta (rewire): " << beta << "\n";
    std::cout << "Lambda (infect): " << lambda << " | Gamma (recover): " << gamma << "\n";

    // 1. Генерируем сеть Ваттса-Строгаца на CPU
    std::vector<std::vector<int>> adj_list(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<int> rand_node(0, N - 1);

    // Сначала кольцевая решетка (каждый соединен с K соседями)
    for (int i = 0; i < N; i++) {
        for (int j = 1; j <= K / 2; j++) {
            int right = (i + j) % N;
            int left = (i - j + N) % N;
            
            // Переподключение правой связи с вероятностью Beta
            if (dis(gen) < beta) {
                int new_target = rand_node(gen);
                while (new_target == i) new_target = rand_node(gen); // Нельзя дружить с самим собой
                adj_list[i].push_back(new_target);
                adj_list[new_target].push_back(i); // Неориентированный граф
            } else {
                adj_list[i].push_back(right);
                adj_list[right].push_back(i);
            }
        }
    }

    // Упаковываем граф (Flat arrays for GPU)
    std::vector<int> h_offsets(N + 1, 0);
    std::vector<int> h_neighbors;
    for (int i = 0; i < N; i++) {
        h_offsets[i] = h_neighbors.size();
        for (int neighbor : adj_list[i]) {
            h_neighbors.push_back(neighbor);
        }
    }
    h_offsets[N] = h_neighbors.size();

    // 2. Инициализируем состояния людей
    std::vector<int> h_state(N, SUSCEPTIBLE);
    // Нулевой пациент (заражаем 5 случайных человек для старта)
    for(int i=0; i<5; i++) h_state[rand_node(gen)] = INFECTED;

    // Память на GPU
    int *d_state_in, *d_state_out;
    int *d_offsets, *d_neighbors;
    curandState* d_states;

    cudaMalloc(&d_state_in, N * sizeof(int));
    cudaMalloc(&d_state_out, N * sizeof(int));
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighbors, h_neighbors.size() * sizeof(int));
    cudaMalloc(&d_states, N * sizeof(curandState));

    cudaMemcpy(d_state_in, h_state.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, h_neighbors.data(), h_neighbors.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    initCurand<<<numBlocks, blockSize>>>(d_states, 777, N);

    // Файл
    std::ofstream history(out_file);
    history << "Day,Susceptible,Infected,Recovered\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_gpu_time = 0;
    int actual_days = 0;

    // Главный цикл симуляции
    for (int day = 0; day < max_days; day++) {
        cudaEventRecord(start);
        
        epidemicStep<<<numBlocks, blockSize>>>(d_state_in, d_state_out, d_offsets, d_neighbors, N, lambda, gamma, d_states);
        
        int* temp = d_state_in; d_state_in = d_state_out; d_state_out = temp; // Swap
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_gpu_time += ms;

        cudaMemcpy(h_state.data(), d_state_in, N * sizeof(int), cudaMemcpyDeviceToHost);

        int count_s = 0, count_i = 0, count_r = 0;
        for (int i = 0; i < N; i++) {
            if (h_state[i] == SUSCEPTIBLE) count_s++;
            else if (h_state[i] == INFECTED) count_i++;
            else if (h_state[i] == RECOVERED) count_r++;
        }

        history << day << "," << count_s << "," << count_i << "," << count_r << "\n";
        actual_days++;

        // Если больных нет, эпидемия закончилась
        if (count_i == 0) {
            std::cout << "Epidemic ended on day " << day << ".\n";
            break;
        }
    }

    std::cout << "\n=== PANDEMIC BENCHMARK ===\n";
    std::cout << "Total GPU Time: " << total_gpu_time << " ms\n";
    std::cout << "Avg time per day: " << (total_gpu_time / actual_days) << " ms\n";

    cudaFree(d_state_in); cudaFree(d_state_out);
    cudaFree(d_offsets); cudaFree(d_neighbors); cudaFree(d_states);
    history.close();
    return 0;
}

// nvcc -O3 -arch=sm_86 pandemic.cu -o pandemic_sim -Xlinker /SUBSYSTEM:CONSOLE
// ./pandemic_sim.exe 10000 0.0 0.3 0.1 200 regular.csv
// ./pandemic_sim.exe 10000 0.1 0.3 0.1 200 smallworld.csv