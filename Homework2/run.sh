#!/bin/bash
#SBATCH --partition=cpsc424
#SBATCH --job-name=ass2
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=10
#SBATCH --mem-per-cpu=6100 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

echo -e "=============== Task 1 ==============="

echo -e "\nRun Serial Program"
./scripts/1.sh

echo -e "\n=============== Task 2 ==============="

echo -e "\nRun Program for Threads 1-10"
./scripts/2-2.sh

echo -e "\nUse Schedule Option"
./scripts/2-3.sh

echo -e "\nAdd Collapse Clause"
./scripts/2-4.sh

echo -e "\n=============== Task 3 ==============="

echo -e "\nEach Cell Constitutes A Task"
./scripts/3-1.sh

echo -e "\nEach Row Constitutes A Task"
./scripts/3-2.sh

echo -e "\nTask Creation Shared by All Threads"
./scripts/3-3.sh

echo -e "\n=============== Task 4 ==============="
echo -e "\nParallel Random Number Generation"
./scripts/4.sh
