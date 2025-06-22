#!/bin/bash

#SBATCH -J mmunir                                                   # Job name
#SBATCH -o out_%j.txt                                               # Name of stdout output file
#SBATCH -e error_%j.txt                                             # Name of stderr error file
#SBATCH -G 8                                                        # GPUs
#SBATCH --cpus-per-gpu=16                                            # CPUs per GPU
#SBATCH -t 04:00:00                                                 # Run time (hh:mm:ss)

python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 main.py configs/mask_rcnn_greedyvig_s_depth_fpn_1x_coco.py --greedyvig_model greedyvig_s --resume-from detection_results/epoch_11_42_74_39_21.pth --work-dir detection_results/ --launcher pytorch > detection_results/greedyvig_s_depth_run.txt 


