#!/bin/bash
#SBATCH --partition=cpsc424
#SBATCH --job-name=ass2
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=10
#SBATCH --mem-per-cpu=6100 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

echo -e "=============== Task 1 ==============="

echo -e "\n---- Run Serial Program ----"
./scripts/1.sh

echo -e "\n=============== Task 2 ==============="

echo -e "\n---- Run Program for Threads 1-10 ----"
./scripts/2-2.sh

echo -e "\n---- Use Schedule Option ----"
./scripts/2-3.sh

echo -e "\n---- Add Collapse Clause ----"
./scripts/2-4.sh

echo -e "\n=============== Task 3 ==============="

echo -e "\n---- Each Cell Constitutes A Task ----"
./scripts/3-1.sh

echo -e "\n---- Each Row Constitutes A Task ----"
./scripts/3-2.sh

echo -e "\n---- Task Creation Shared by All Threads ----"
./scripts/3-3.sh

echo -e "\n=============== Task 4 ==============="
echo -e "\n---- Parallel Random Number Generation ----"
./scripts/4.sh
