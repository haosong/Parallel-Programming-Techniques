#include <stdio.h>
#include "timing.h"

int main() {
    double wcs, wce, ct;
    int N = 1000000000;
    double dx = 1.0 / N;
    double xi = 1.0 / (2 * N);
    double pi = 0.0;
    double a = 1.0; 
    double b = 1.0;
    timing(&wcs, &ct);
    for (int i = 0; i < N; i++) {
        pi += dx / (1.0 + xi * xi);
        xi += dx;
	//a /= b;
    }
    pi *= 4;
    timing(&wce, &ct);
    double runtime = wce - wcs;
    printf("%f\n", pi);
    printf("%f\t%f\n", runtime, (double) 5 * N / (runtime * 1000000));
    return 0;
}

