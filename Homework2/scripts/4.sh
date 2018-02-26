#!/bin/bash
#SBATCH --partition=cpsc424
#SBATCH --job-name=ass2_task2
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=10
#SBATCH --mem-per-cpu=6100 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

module load Langs/Intel/2015_update2

export OMP_SCHEDULE="dynamic"
./task4 10
