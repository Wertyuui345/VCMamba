#!/bin/bash
#conda init ~/bin/bash
#conda activate VCMamba 
cd /work/09816/wertyuui345/ls6/VCMamba/VCMamba/segmentation;

MASTER_ADDR=$1 LOCAL_ADDR=$(hostname | cut -d '.' -f 1) #Master ADDR change based on argument passed in order.

echo first_local_addr_value @ $LOCAL_ADDR...
echo first master_addr_value @ $MASTER_ADDR...

if [ $MASTER_ADDR = $LOCAL_ADDR ]
then
    echo Launching master @ $MASTER_ADDR...
    NCCL_DEBUG=INFO RANK=0 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 3 --master_addr=$MASTER_ADDR --node_rank 0 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_b_ade20k_40k.py --model VCMamba_EfficientFormer_B --work-dir semantic_results/MambaCNN_B/ --launcher pytorch > semantic_results/MambaCNN_B_run_semantic.txt
elif [ $(scontrol show hostnames | sed -n '2p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=3 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 3 --master_addr=$MASTER_ADDR --node_rank 1 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_b_ade20k_40k.py --model VCMamba_EfficientFormer_B --work-dir semantic_results/MambaCNN_B/ --launcher pytorch > semantic_results/MambaCNN_B_run_semantic.txt
elif [ $(scontrol show hostnames | sed -n '3p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=6 python -m torch.distributed.launch --nproc_per_node 2 --nnodes 3  --master_addr=$MASTER_ADDR --node_rank 2 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_b_ade20k_40k.py --model VCMamba_EfficientFormer_B --work-dir semantic_results/MambaCNN_B/ --launcher pytorch > semantic_results/MambaCNN_B_run_semantic.txt
# elif [ $(scontrol show hostnames | sed -n '4p') = $LOCAL_ADDR ]
# then
#     echo Launching master @ $LOCAL_ADDR...
#     RANK=9 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 8 --master_addr=$MASTER_ADDR --node_rank 3 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_s_ade20k_40k.py --model VCMamba_EfficientFormer_S --work-dir semantic_results/MambaCNN_S/ --launcher pytorch > semantic_results/MambaCNN_S_run_semantic.txt
# elif [ $(scontrol show hostnames | sed -n '5p') = $LOCAL_ADDR ]
# then
#     echo Launching master @ $MASTER_ADDR...
#     RANK=12 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 8 --master_addr=$MASTER_ADDR --node_rank 4 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_s_ade20k_40k.py --model VCMamba_EfficientFormer_S --work-dir semantic_results/MambaCNN_S/ --launcher pytorch > semantic_results/MambaCNN_S_run_semantic.txt
# elif [ $(scontrol show hostnames | sed -n '6p') = $LOCAL_ADDR ]
# then
#     echo Launching master @ $LOCAL_ADDR...
#     RANK=15 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 8 --master_addr=$MASTER_ADDR --node_rank 5 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_s_ade20k_40k.py --model VCMamba_EfficientFormer_S --work-dir semantic_results/MambaCNN_S/ --launcher pytorch > semantic_results/MambaCNN_S_run_semantic.txt
# elif [ $(scontrol show hostnames | sed -n '7p') = $LOCAL_ADDR ]
# then
#     echo Launching master @ $LOCAL_ADDR...
#     RANK=18 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 8 --master_addr=$MASTER_ADDR --node_rank 6 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_s_ade20k_40k.py --model VCMamba_EfficientFormer_S --work-dir semantic_results/MambaCNN_S/ --launcher pytorch > semantic_results/MambaCNN_S_run_semantic.txt
# elif [ $(scontrol show hostnames | sed -n '8p') = $LOCAL_ADDR ]
# then
#     echo Launching master @ $LOCAL_ADDR...
#     RANK=21 python -m torch.distributed.launch --nproc_per_node 3 --nnodes 8 --master_addr=$MASTER_ADDR --node_rank 7 --master_port=3456 --use_env train.py configs/sem_fpn/fpn_mcnn_s_ade20k_40k.py --model VCMamba_EfficientFormer_S --work-dir semantic_results/MambaCNN_S/ --launcher pytorch > semantic_results/MambaCNN_S_run_semantic.txt
else
    echo No matching addr...
fi