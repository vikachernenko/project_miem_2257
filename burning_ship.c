#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 100
#define ESCAPE_RADIUS_SQ 16.0

typedef struct {
    double xmin, xmax, ymin, ymax;
    int width, height, max_iter;
} FractalParams;

int burning_ship(double cx, double cy) {
    double x = 0.0, y = 0.0;
    double x2 = 0.0, y2 = 0.0;
    int iter = 0;
    
    while (iter < MAX_ITER) {
        y = fabs(2.0 * x * y) + cy;
        x = fabs(x2 - y2) + cx;
        x2 = x * x;
        y2 = y * y;
        
        if (x2 + y2 > ESCAPE_RADIUS_SQ) {
            break;
        }
        iter++;
    }
    return iter;
}

void get_burning_ship_color(int iter, unsigned char* r, unsigned char* g, unsigned char* b) {
    if (iter == MAX_ITER) {
        *r = 0; *g = 0; *b = 0;
    } else {
        double t = (double)iter / MAX_ITER;
        *r = (unsigned char)(255 * (t * 0.8 + 0.2));
        *g = (unsigned char)(255 * (t * t * 0.6));
        *b = (unsigned char)(255 * (sqrt(t) * 0.4));
    }
}

unsigned char* generate_burning_ship(FractalParams params) {
    double x_scale = (params.xmax - params.xmin) / params.width;
    double y_scale = (params.ymax - params.ymin) / params.height;
    unsigned char* image = (unsigned char*)malloc(params.width * params.height * 3);
    
    if (!image) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < params.height; y++) {
        double cy = params.ymin + y * y_scale;
        for (int x = 0; x < params.width; x++) {
            double cx = params.xmin + x * x_scale;
            int iter = burning_ship(cx, cy);
            unsigned char r, g, b;
            get_burning_ship_color(iter, &r, &g, &b);
            int idx = (y * params.width + x) * 3;
            image[idx] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
    return image;
}

void save_ppm(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image, 1, width * height * 3, fp);
    fclose(fp);
}

int main() {
    clock_t start_time, end_time;
    double calc_time, save_time;
    
    FractalParams params = {-1.8, -1.7, -0.09, 0.01, WIDTH, HEIGHT, MAX_ITER};
    
    start_time = clock();
    unsigned char* image = generate_burning_ship(params);
    end_time = clock();
    calc_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    start_time = clock();
    save_ppm("burning_ship.ppm", image, params.width, params.height);
    end_time = clock();
    save_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("Calculation: %.3f s\n", calc_time);
    printf("Save: %.3f s\n", save_time);
    printf("Total: %.3f s\n", calc_time + save_time);
    
    free(image);
    return 0;
}