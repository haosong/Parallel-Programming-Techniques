#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include "drand.h"
#include "timing.h"

int main(int argc, char const *argv[]) {
    dsrand(12345);
    double wcs, wce, ct;
    timing(&wcs, &ct);
    int N0 = 0, N1 = 0;
    #pragma omp parallel {
    #pragma omp for 
    for (int x = -2000; x < 500; x++) {
        for (int y = 0; y < 1250; y++) {
            double cr = (drand() + (double)x) * 0.001;
            double ci = (drand() + (double)y) * 0.001;
            double zr = cr;
            double zi = ci;
            int i = 0;
            for (; i < 20000; i++) {
                double new_zr = zr * zr - zi * zi + cr;
                double new_zi = 2 * zr * zi + ci;
                zr = new_zr;
                zi = new_zi;
                if (zr * zr + zi * zi > 4) break;
            }
            if (i > 20000) N1++;
            else N0++;
        }
    }
    }
    timing(&wce, &ct);
    double runtime = wce - wcs;
    double area = 2 * N1 * 3.125 / (N1 + N0);
    printf("area = %f\n", area);
    printf("runtime = %f\n", runtime);
    return 0;
}

