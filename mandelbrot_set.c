#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 100
#define ESCAPE_RADIUS_SQ 4.0

int mandelbrot(double complex c) {
    double real = 0.0, imag = 0.0;
    double real_sq = 0.0, imag_sq = 0.0;
    int n;
    for (n = 0; n < MAX_ITER; n++) {
        imag = 2 * real * imag + cimag(c);
        real = real_sq - imag_sq + creal(c);
        real_sq = real * real;
        imag_sq = imag * imag;
        if (real_sq + imag_sq > ESCAPE_RADIUS_SQ) {
            return n;
        }
    }
    return MAX_ITER;
}

double* linspace(double start, double end, int num) {
    double* arr = (double*)malloc(num * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed");
        exit(1);
    }
    
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        arr[i] = start + i * step;
    }
    return arr;
}

int* mandelbrot_set(double xmin, double xmax, double ymin, double ymax) {
    double* x = linspace(xmin, xmax, WIDTH);
    double* y = linspace(ymin, ymax, HEIGHT);
    int* mset = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    
    if (!mset) {
        fprintf(stderr, "Memory allocation failed");
        free(x);
        free(y);
        exit(1);
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double complex c = x[j] + y[i] * I;
            mset[i * WIDTH + j] = mandelbrot(c);
        }
    }

    free(x);
    free(y);
    return mset;
}

int main() {
    clock_t start_time, end_time;
    double calc_time, save_time, total_time;
    double xmin = -2.0, xmax = 1.0, ymin = -1.5, ymax = 1.5;
    start_time = clock();
    int* mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax);
    end_time = clock();
    calc_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    start_time = clock();

    FILE* fp = fopen("mandelbrot.pgm", "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing");
        free(mandelbrot_image);
        exit(1);
    }

    fprintf(fp, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        unsigned char val = (unsigned char)(255 * mandelbrot_image[i] / MAX_ITER);
        fwrite(&val, 1, 1, fp);
    }
    fclose(fp);
    
    end_time = clock();
    save_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    total_time = calc_time + save_time;
    printf("Calculation time: %.3f seconds\n", calc_time);
    printf("Save time: %.3f seconds\n", save_time);
    printf("Total time: %.3f seconds\n", total_time);
    free(mandelbrot_image);
    return 0;
}