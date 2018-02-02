#include <stdio.h>
#include "timing.h"

int main() {
    double wcs, wce, ct;
    int N = 1000000000;
    double a = (double) rand() / RAND_MAX; 
    double b = (double) rand() / RAND_MAX;
    timing(&wcs, &ct);
    for (int i = 0; i < N; i++) {
        a = a / b;
        a = b / a;
    }
    timing(&wce, &ct);
    double runtime = wce - wcs;
    printf("\na = %f\nb = %f\n", a, b); 
    printf("Runtime = %f\n", runtime);
    printf("MFlops = %f\n\n", (double) 2 * N / (runtime * 1000000));
    return 0;
}

