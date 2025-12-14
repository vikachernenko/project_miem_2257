// compile: nvcc -O3 -arch=sm_86 -o random_walk_1d random_walk_1d.cu -lcurand
// run: ./random_walk_1d [num_walkers] [num_steps] [threads_per_block]
// example: ./random_walk_1d 5000000 1000 256 10

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>
#include <random>

#define CUDA_CHECK(call) { cudaError_t e = (call); if(e != cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }

// GPU Kernels
__global__ void init_rng(curandStatePhilox4_32_10_t *rng_states, unsigned long long seed, size_t n_walkers){
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=n_walkers) return;
    curand_init(seed, idx, 0, &rng_states[idx]);
}

__global__ void simulate_1d_gpu(
    curandStatePhilox4_32_10_t *rng_states,
    long long *hist,
    double *sum_pos,
    double *sum_pos_sq,
    int *sample_paths,
    size_t n_walkers,
    int n_steps,
    int N_sample)
{
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=n_walkers) return;

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int pos = 0;

    for(int step=0; step<n_steps; ++step){
        int move = (curand_uniform(&state)<=0.5f) ? -1 : +1;
        pos += move;
        atomicAdd(&sum_pos[step], (double)pos);
        atomicAdd(&sum_pos_sq[step], (double)pos*(double)pos);

        if(idx<N_sample){
            sample_paths[idx*n_steps + step] = pos;
        }
    }

    int hist_idx = pos + n_steps;
    atomicAdd((unsigned long long*)&hist[hist_idx], 1ULL);
    rng_states[idx] = state;
}

int main(int argc,char** argv){
    // Parameters
    size_t n_walkers = 5'000'000;
    int n_steps = 1000;
    int threads_per_block = 256;
    int N_sample = 10;

    if(argc>=2) n_walkers = strtoull(argv[1],nullptr,10);
    if(argc>=3) n_steps = atoi(argv[2]);
    if(argc>=4) threads_per_block = atoi(argv[3]);
    if(argc>=5) N_sample = atoi(argv[4]);

    printf("1D Random Walk — walkers=%zu, steps=%d, tpb=%d, sample=%d\n",
           n_walkers,n_steps,threads_per_block,N_sample);


    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<double> sum_cpu(n_steps,0.0);
    std::vector<double> sumsq_cpu(n_steps,0.0);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    for(size_t w=0;w<n_walkers;++w){
        int pos = 0;
        for(int step=0;step<n_steps;++step){
            int move = (dist(rng)<=0.5f)? -1 : +1;
            pos += move;
            sum_cpu[step] += pos;
            sumsq_cpu[step] += pos*pos;
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double,std::milli>(cpu_end-cpu_start).count();
    printf("CPU simulation time: %.3f ms\n", cpu_time_ms);


    curandStatePhilox4_32_10_t *d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng,n_walkers*sizeof(curandStatePhilox4_32_10_t)));
    long long *d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist,(2*n_steps+1)*sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_hist,0,(2*n_steps+1)*sizeof(long long)));

    double *d_sum, *d_sum_sq;
    CUDA_CHECK(cudaMalloc(&d_sum,n_steps*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq,n_steps*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum,0,n_steps*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_sq,0,n_steps*sizeof(double)));

    int *d_sample_paths=nullptr;
    if(N_sample>0) CUDA_CHECK(cudaMalloc(&d_sample_paths,N_sample*n_steps*sizeof(int)));

    size_t blocks = (n_walkers+threads_per_block-1)/threads_per_block;
    unsigned long long seed = (unsigned long long)time(NULL);
    init_rng<<<blocks,threads_per_block>>>(d_rng,seed,n_walkers);
    CUDA_CHECK(cudaDeviceSynchronize());


    auto gpu_start = std::chrono::high_resolution_clock::now();
    simulate_1d_gpu<<<blocks,threads_per_block>>>(d_rng,d_hist,d_sum,d_sum_sq,d_sample_paths,
                                                  n_walkers,n_steps,N_sample);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double,std::milli>(gpu_end-gpu_start).count();
    printf("GPU simulation time: %.3f ms\n", gpu_time_ms);


    std::vector<double> sum_gpu(n_steps);
    std::vector<double> sumsq_gpu(n_steps);
    std::vector<long long> hist(2*n_steps+1);

    CUDA_CHECK(cudaMemcpy(sum_gpu.data(),d_sum,n_steps*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sumsq_gpu.data(),d_sum_sq,n_steps*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hist.data(),d_hist,(2*n_steps+1)*sizeof(long long),cudaMemcpyDeviceToHost));

    std::vector<int> sample_paths;
    if(N_sample>0){
        sample_paths.resize(N_sample*n_steps);
        CUDA_CHECK(cudaMemcpy(sample_paths.data(),d_sample_paths,N_sample*n_steps*sizeof(int),cudaMemcpyDeviceToHost));
    }


    FILE *f_sigma = fopen("sigma_1d.csv","w");
    fprintf(f_sigma,"step,sigma_empirical,sigma_theory\n");
    for(int step=0;step<n_steps;++step){
        double mean = sum_gpu[step]/(double)n_walkers;
        double mean_sq = sumsq_gpu[step]/(double)n_walkers;
        double var = mean_sq - mean*mean;
        if(var<0 && var>-1e-12) var=0;
        double sigma = sqrt(var);
        fprintf(f_sigma,"%d,%.10g,%.10g\n", step+1,sigma,sqrt(step+1.0));
    }
    fclose(f_sigma);

    FILE *f_hist = fopen("histogram_1d.csv","w");
    fprintf(f_hist,"position,count\n");
    for(int i=0;i<2*n_steps+1;++i) fprintf(f_hist,"%d,%lld\n", i-n_steps, hist[i]);
    fclose(f_hist);

    if(N_sample>0){
        FILE *f_paths = fopen("sample_paths_1d.csv","w");
        fprintf(f_paths,"walker,step,position\n");
        for(int w=0;w<N_sample;++w)
            for(int s=0;s<n_steps;++s)
                fprintf(f_paths,"%d,%d,%d\n",w,s,sample_paths[w*n_steps+s]);
        fclose(f_paths);
    }


    cudaFree(d_rng); cudaFree(d_hist); cudaFree(d_sum); cudaFree(d_sum_sq);
    if(d_sample_paths) cudaFree(d_sample_paths);

    printf("Done.\n");
    return 0;
}