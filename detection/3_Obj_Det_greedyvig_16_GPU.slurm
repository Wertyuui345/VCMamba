#!/bin/bash

#SBATCH -J mmunir                                                   # Job name
#SBATCH -o out_%j.txt                                               # Name of stdout output file
#SBATCH -e error_%j.txt                                             # Name of stderr error file
#SBATCH -p gpu-a100                                                 # Queue (partition) name
#SBATCH -N 6                                                        # Total # of nodes (must be 1 for serial)
#SBATCH -n 6                                                        # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 04:00:00                                                  # Run time (hh:mm:ss)
#SBATCH --mail-type=all                                             # Send email at begin and end of job
#SBATCH --mail-user='mmunir@utexas.edu'

# Any other commands must follow all #SBATCH directives...
module load "intel/19.1.1" "python3/3.9.7"
#module load "cuda/11.3" "cudnn/8.2.1" "nccl/2.9.9"
source $SCRATCH/Documents/Virt_Envs/TACC_MobileViGv2/bin/activate

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

ibrun sh $SCRATCH/Documents/Git_Repos/GreedyViG/detection/3_Obj_Det_greedyvig_16_GPU.sh $master_addr



