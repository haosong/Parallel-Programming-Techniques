#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


static uint64_t seed;

// set seed as thread private to make program thread safe
#pragma omp threadprivate(seed)

void dsrand(unsigned s);
void dsrand_parallel(unsigned s);
double drand(void);
