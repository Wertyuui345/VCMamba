#!/bin/bash

#SBATCH -A CCR23002        #<project_name from user portal>
#SBATCH -J VCMambaEff           # Job name
#SBATCH -o VCMambaEff.o%j       # Name of stdout output file
#SBATCH -e VCMambaEff.e%j       # Name of stderr error file
#SBATCH -p gpu-a100        # Queue (partition) name
#SBATCH -N 6               # Total # of nodes (must be 1 for serial)
#SBATCH -n 6               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
# #SBATCH --mail-type=all    # Send email at begin and end of job

# Any other commands must follow all #SBATCH directives...
# module load cuda/11.3 cudnn nccl
# nvcc --version

##  module load "cuda/11.8" "python=3.10.13"

source $WORK/env/VCMamba/bin/activate

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

date
ibrun sh $WORK/VCMamba/VCMamba/model/scripts/vcmamba-eff-l-multi.sh $master_addr
date
