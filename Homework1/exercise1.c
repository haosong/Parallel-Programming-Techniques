#include <stdio.h>
#include "timing.h"

int main() {
    double wcs, wce, ct;
    int N = 1000000000;
    double dx = 1.0 / N;
    double xi = 1.0 / (2 * N);
    double pi = 0.0;
    timing(&wcs, &ct);
    for (int i = 0; i < N; i++) {
        pi += dx / (1.0 + xi * xi);
        xi += dx;
    }
    timing(&wce, &ct);
    double runtime = wce - wcs;
    printf("\nPi = %f\n", pi * 4);
    printf("Runtime = %f\n", runtime);
    printf("MFlops = %f\n\n", (double) 5 * N / (runtime * 1000000));
    return 0;
}

