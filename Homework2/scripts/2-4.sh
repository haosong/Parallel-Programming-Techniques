#!/bin/bash
#SBATCH --partition=cpsc424
#SBATCH --job-name=ass2_task2
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=10
#SBATCH --mem-per-cpu=6100 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

module load Langs/Intel/2015_update2

echo -e "schedule(static,1), collapse(2)\n"
export OMP_SCHEDULE="static,1"
./task2_4 2
./task2_4 4
./task2_4 8
./task2_4 10

echo -e "schedule(dynamic), collapse(2)\n"
export OMP_SCHEDULE="dynamic"
./task2_4 2
./task2_4 4
./task2_4 8
./task2_4 10

echo -e "schedule(guided), collapse(2)\n"
export OMP_SCHEDULE="guided"
./task2_4 2
./task2_4 4
./task2_4 8
./task2_4 10
