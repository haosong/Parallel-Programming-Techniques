#include <stdio.h>
#include <math.h>
#include "timing.h"

int main() {
    double wcs, wce, ct;
    size_t N = (size_t) floor(pow(2.1, 25));
    double *a = calloc(N, sizeof(double));
    double *b = calloc(N, sizeof(double));
    double *c = calloc(N, sizeof(double));
    double *d = calloc(N, sizeof(double));
    for (int k = 3; k <= 25; k++) {
        N = (size_t) floor(pow(2.1, k));
        for (size_t i = 0; i < N; i++) {
            a[i] = 100.0 / (double) RAND_MAX * (double) rand();
            b[i] = 100.0 / (double) RAND_MAX * (double) rand();
            c[i] = 100.0 / (double) RAND_MAX * (double) rand();
            d[i] = 100.0 / (double) RAND_MAX * (double) rand();
        }
        int repeat = 1;
        double runtime = 0.0;
        while (runtime < 1.0) {
            timing(&wcs, &ct);
            for (int r = 0; r < repeat; ++r) {
                /* PUT THE KERNEL BENCHMARK LOOP HERE */
                for (size_t i = 0; i < N; i++) {
                    a[i] = b[i] + c[i] * d[i];
                }
                if (a[N >> 1] < 0) r += 0; // fools the compiler
            }
            timing(&wce, &ct);
            runtime = wce - wcs;
            repeat *= 2;
        }
	repeat /= 2;
        printf("%zu\t%i\t%f\t%f\n", N, repeat, runtime, (double) 2 * N * repeat / (runtime * 1000000));
    }
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

