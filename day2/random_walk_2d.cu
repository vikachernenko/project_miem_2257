// compile: nvcc -O3 -arch=sm_86 -o random_walk_2d random_walk_2d.cu -lcurand
// run: ./random_walk_2d [num_walkers] [num_steps] [threads_per_block] [bins]
// example: ./random_walk_2d 1000000 1000 256 100 10


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

__global__ void simulate_2d_gpu(
    curandStatePhilox4_32_10_t *rng_states,
    unsigned long long *hist_r,
    double *sum_r2,
    int *x_paths, int *y_paths,
    size_t n_walkers,
    int n_steps,
    int bins,
    double max_r,
    int N_sample)
{
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=n_walkers) return;

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int x=0, y=0;

    for(int step=0;step<n_steps;++step){
        x += (curand_uniform(&state)<=0.5f)? -1 : +1;
        y += (curand_uniform(&state)<=0.5f)? -1 : +1;
        atomicAdd(&sum_r2[step], double(x*x + y*y));

        if(idx<N_sample){
            x_paths[idx*n_steps + step] = x;
            y_paths[idx*n_steps + step] = y;
        }
    }

    double r = sqrt(double(x*x + y*y));
    int bin = int(r/max_r * bins);
    if(bin<0) bin=0;
    if(bin>=bins) bin=bins-1;
    atomicAdd(&hist_r[bin],1ULL);

    rng_states[idx] = state;
}

int main(int argc,char** argv){
    size_t n_walkers=1'000'000;
    int n_steps=1000;
    int threads_per_block=256;
    int bins=100;
    int N_sample=10;

    if(argc>=2) n_walkers= strtoull(argv[1],nullptr,10);
    if(argc>=3) n_steps= atoi(argv[2]);
    if(argc>=4) threads_per_block= atoi(argv[3]);
    if(argc>=5) bins= atoi(argv[4]);
    if(argc>=6) N_sample= atoi(argv[5]);

    printf("2D Random Walk — walkers=%zu, steps=%d, tpb=%d, bins=%d, sample=%d\n",
           n_walkers,n_steps,threads_per_block,bins,N_sample);


    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<double> sum_r2_cpu(n_steps,0.0);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    for(size_t w=0; w<n_walkers; ++w){
        int x=0,y=0;
        for(int step=0; step<n_steps;++step){
            x += (dist(rng)<=0.5f)? -1 : +1;
            y += (dist(rng)<=0.5f)? -1 : +1;
            sum_r2_cpu[step] += double(x*x + y*y);
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double,std::milli>(cpu_end-cpu_start).count();
    printf("CPU simulation time: %.3f ms\n", cpu_time_ms);


    curandStatePhilox4_32_10_t *d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng,n_walkers*sizeof(curandStatePhilox4_32_10_t)));
    unsigned long long *d_hist;
    double *d_sum_r2;
    CUDA_CHECK(cudaMalloc(&d_hist,bins*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hist,0,bins*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_r2,n_steps*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_r2,0,n_steps*sizeof(double)));

    int *d_x_paths=nullptr,*d_y_paths=nullptr;
    if(N_sample>0){
        CUDA_CHECK(cudaMalloc(&d_x_paths,N_sample*n_steps*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_y_paths,N_sample*n_steps*sizeof(int)));
    }

    size_t blocks=(n_walkers+threads_per_block-1)/threads_per_block;
    unsigned long long seed = (unsigned long long)time(NULL);
    init_rng<<<blocks,threads_per_block>>>(d_rng,seed,n_walkers);
    CUDA_CHECK(cudaDeviceSynchronize());

    double max_r = sqrt(2.0*n_steps);


    auto gpu_start = std::chrono::high_resolution_clock::now();
    simulate_2d_gpu<<<blocks,threads_per_block>>>(d_rng,d_hist,d_sum_r2,d_x_paths,d_y_paths,
                                                  n_walkers,n_steps,bins,max_r,N_sample);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double,std::milli>(gpu_end-gpu_start).count();
    printf("GPU simulation time: %.3f ms\n", gpu_time_ms);


    std::vector<double> sum_r2(n_steps);
    std::vector<unsigned long long> hist(bins);
    CUDA_CHECK(cudaMemcpy(sum_r2.data(),d_sum_r2,n_steps*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hist.data(),d_hist,bins*sizeof(unsigned long long),cudaMemcpyDeviceToHost));

    std::vector<int> x_paths,y_paths;
    if(N_sample>0){
        x_paths.resize(N_sample*n_steps);
        y_paths.resize(N_sample*n_steps);
        CUDA_CHECK(cudaMemcpy(x_paths.data(),d_x_paths,N_sample*n_steps*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_paths.data(),d_y_paths,N_sample*n_steps*sizeof(int),cudaMemcpyDeviceToHost));
    }


    FILE *f_sigma=fopen("sigma_2d.csv","w");
    fprintf(f_sigma,"step,sigma_empirical,sigma_theory\n");
    for(int step=0;step<n_steps;++step){
        double mean_r2=sum_r2[step]/(double)n_walkers;
        fprintf(f_sigma,"%d,%.10g,%.10g\n",step+1,sqrt(mean_r2),sqrt(2.0*(step+1)));
    }
    fclose(f_sigma);

    FILE *f_hist=fopen("histogram_2d.csv","w");
    fprintf(f_hist,"bin,r_min,r_max,count\n");
    for(int b=0;b<bins;++b){
        double rmin = double(b)*max_r/bins;
        double rmax = double(b+1)*max_r/bins;
        fprintf(f_hist,"%d,%.5f,%.5f,%llu\n",b,rmin,rmax,hist[b]);
    }
    fclose(f_hist);

    if(N_sample>0){
        FILE *f_paths=fopen("sample_paths_2d.csv","w");
        fprintf(f_paths,"walker,step,x,y\n");
        for(int w=0;w<N_sample;++w)
            for(int s=0;s<n_steps;++s)
                fprintf(f_paths,"%d,%d,%d,%d\n",w,s,x_paths[w*n_steps+s],y_paths[w*n_steps+s]);
        fclose(f_paths);
    }


    cudaFree(d_rng); cudaFree(d_hist); cudaFree(d_sum_r2);
    if(d_x_paths) cudaFree(d_x_paths);
    if(d_y_paths) cudaFree(d_y_paths);

    printf("Done.\n");
    return 0;
}