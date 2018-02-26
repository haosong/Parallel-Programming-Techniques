#include "drand.h"

void dsrand(unsigned s)
{
    seed = s-1;
    // printf("Seed = %lu. RAND_MAX = %d.\n",seed,RAND_MAX);
}

void dsrand_parallel(unsigned s)
{
    seed = s-1;
    // assign an offset for each threads based on their thread num
    for (int i = 0; i < (omp_get_thread_num() * (2500 * 1250 / omp_get_num_threads())); i++)
        seed = 6364136223846793005ULL*seed + 1;
        // printf("Seed = %lu. RAND_MAX = %d.\n",seed,RAND_MAX);
}

double drand(void)
{
    seed = 6364136223846793005ULL*seed + 1;
    return((double)(seed>>33)/(double)RAND_MAX);
}
