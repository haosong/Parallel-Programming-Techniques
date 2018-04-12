#!/bin/bash
#SBATCH --partition=cpsc424
# set total number of MPI processes
#SBATCH --ntasks=8
# set number of MPI processes per node
# (number of nodes is calculated by Slurm)
#SBATCH --ntasks-per-node=8
# set number of cpus per MPI process
#SBATCH --cpus-per-task=1
# set memory per cpu
#SBATCH --mem-per-cpu=6100mb
#SBATCH --job-name=MPI_RUN
#SBATCH --time=15:00

module load Langs/Intel/15 MPI/OpenMPI/2.1.1-intel15
pwd
echo $SLURM_JOB_NODELIST
echo $SLURM_NTASKS_PER_NODE
make clean
make task2
time mpirun -n 8 ./task2
