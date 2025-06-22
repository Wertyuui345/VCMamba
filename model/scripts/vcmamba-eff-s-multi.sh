#!/bin/bash
#conda init ~/bin/bash
#conda activate VCMamba
cd /scratch/09816/wertyuui345/ls6/VCMamba/VCMamba/model;

MASTER_ADDR=$1 LOCAL_ADDR=$(hostname | cut -d '.' -f 1) #Master ADDR change based on argument passed in order.

echo first_local_addr_value @ $LOCAL_ADDR...
echo first master_addr_value @ $MASTER_ADDR...

if [ $MASTER_ADDR = $LOCAL_ADDR ]
then
    echo Launching master @ $MASTER_ADDR...
    RANK=0 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=0 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
elif [ $(scontrol show hostnames | sed -n '2p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=3 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=1 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
elif [ $(scontrol show hostnames | sed -n '3p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=6 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=2 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
elif [ $(scontrol show hostnames | sed -n '4p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=9 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=3 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
elif [ $(scontrol show hostnames | sed -n '5p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=12 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=4 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
elif [ $(scontrol show hostnames | sed -n '6p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=15 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --master_addr=$MASTER_ADDR --node_rank=5 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.2 --weight-decay 0.05 --num_workers 25 --model VCMamba_EfficientFormer_S --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test/hybrid_s --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
else
    echo No matching addr...
fi