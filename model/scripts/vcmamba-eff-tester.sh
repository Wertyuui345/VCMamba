#!/bin/bash
#conda init ~/bin/bash
#conda activate VCMamba
cd /scratch/09816/wertyuui345/ls6/VCMamba/VCMamba/model;

MASTER_ADDR=$1 LOCAL_ADDR=$(hostname | cut -d '.' -f 1) #Master ADDR change based on argument passed in order.

echo first_local_addr_value @ $LOCAL_ADDR...
echo first master_addr_value @ $MASTER_ADDR...

if [ $MASTER_ADDR = $LOCAL_ADDR ]
then
    RANK=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_addr=$MASTER_ADDR --node_rank=0 --master_port=3456 --use_env main.py --batch-size 128 --input-size 224 --lr 1e-3 --drop-path 0.05 --weight-decay 0.05 --num_workers 25 --model efficientformerv2_s2 --data-path $SCRATCH/ls6/VCMamba/data/ --data-set 'IMNET' --output_dir ./output/test --epochs 300 --no_amp --distillation-type 'soft' --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
else
    echo No matching addr...
fi