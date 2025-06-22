#!/bin/bash

#SBATCH -A CCR23002        #<project_name from user portal>
#SBATCH -J VCMambaEffSeg           # Job name
#SBATCH -o VCMambaEffSeg.o%j       # Name of stdout output file
#SBATCH -e VCMambaEffSeg.e%j       # Name of stderr error file
#SBATCH -p gpu-a100      # Queue (partition) name
#SBATCH -N 3               # Total # of nodes (must be 1 for serial)
#SBATCH -n 3               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 6:00:00        # Run time (hh:mm:ss)
# #SBATCH --mail-type=all    # Send email at begin and end of job
# #SBATCH --mail-user=alex.zhang@utexas.edu

# Any other commands must follow all #SBATCH directives...
# module load cuda/11.3 cudnn nccl
# nvcc --version

ml use /work/02064/ajs2987/ls6/apps/modulefiles
ml cuda/11.8

source $WORK/env/VCMamba/bin/activate

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

date
ibrun sh $WORK/VCMamba/VCMamba/segmentation/MambaCNNRun.sh $master_addr
date

