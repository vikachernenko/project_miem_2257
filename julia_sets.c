#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 1000
#define ESCAPE_RADIUS_SQ 4.0

double complex c = -0.7 + 0.27015 * I;

int julia(double complex z) {
    double real = creal(z);
    double imag = cimag(z);
    double real_sq = real * real;
    double imag_sq = imag * imag;
    int n;
    for (n = 0; n < MAX_ITER; n++) {
        if (real_sq + imag_sq > ESCAPE_RADIUS_SQ) {
            return n;
        }
        imag = 2 * real * imag + cimag(c);
        real = real_sq - imag_sq + creal(c);
        real_sq = real * real;
        imag_sq = imag * imag;
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

int* julia_set(double xmin, double xmax, double ymin, double ymax) {
    double* x = linspace(xmin, xmax, WIDTH);
    double* y = linspace(ymin, ymax, HEIGHT);
    
    int* jset = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    if (!jset) {
        fprintf(stderr, "Memory allocation failed");
        free(x);
        free(y);
        exit(1);
    }
    const double creal_c = creal(c);
    const double cimag_c = cimag(c);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double complex z = x[j] + y[i] * I;
            jset[i * WIDTH + j] = julia(z);
        }
    }

    free(x);
    free(y);
    return jset;
}

void get_color(int iter, unsigned char pixel[3]) {
    if (iter == MAX_ITER) {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
    } else {
        double t = (double)iter / MAX_ITER;
        pixel[0] = (unsigned char)(9 * (1-t) * t * t * t * 255);
        pixel[1] = (unsigned char)(15 * (1-t) * (1-t) * t * t * 255);
        pixel[2] = (unsigned char)(8.5 * (1-t) * (1-t) * (1-t) * t * 255);
    }
}

int main() {
    clock_t start_time, end_time;
    double calc_time, save_time, total_time, xmin = -1.5, xmax = 1.5, ymin = -1.5, ymax = 1.5;
    unsigned char* pixel_data = (unsigned char*)malloc(WIDTH * HEIGHT * 3);
    int* julia_image = julia_set(xmin, xmax, ymin, ymax);
    int i;
    start_time = clock();
    end_time = clock();
    calc_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    FILE* fp = fopen("julia_set.ppm", "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing");
        free(julia_image);
        exit(1);
    }
    
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    start_time = clock();
    if (!pixel_data) {
        fprintf(stderr, "Memory allocation failed for pixel data");
        fclose(fp);
        free(julia_image);
        exit(1);
    }
    
    #pragma omp parallel for
    for (i = 0; i < WIDTH * HEIGHT; i++) {
        unsigned char pixel[3];
        get_color(julia_image[i], pixel);
        pixel_data[i * 3] = pixel[0];
        pixel_data[i * 3 + 1] = pixel[1];
        pixel_data[i * 3 + 2] = pixel[2];
    }
    
    fwrite(pixel_data, 1, WIDTH * HEIGHT * 3, fp);
    end_time = clock();
    save_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    total_time = calc_time + save_time;
    printf("Calculation time: %.3f seconds\n", calc_time);
    printf("Save time: %.3f seconds\n", save_time);
    printf("Total time: %.3f seconds\n", total_time);
    fclose(fp);
    free(pixel_data);
    free(julia_image);
    return 0;
}