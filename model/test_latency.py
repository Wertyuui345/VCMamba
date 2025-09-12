import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
import time

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from util import *
import mambacnn

# Lambda Nvidia RTX A6000
# Office Computer Nvidia GeForce RTX 3090 Ti
# Do all latency experiments on Office Computer

def get_args_parser():
    parser = argparse.ArgumentParser(
        'MobileViG training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-batches', default=1000, type=int)

    # Model parameters
    parser.add_argument('--model', default='VCMamba_EfficientFormer_M', type=str, metavar='MODEL', # pvihg_s_224_gelu, GreedyViG_S, mobilevig_m
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    
    return parser

parser = argparse.ArgumentParser(
    'MobileViG throughput script', parents=[get_args_parser()])
args = parser.parse_args()
dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size)

model = create_model(args.model)
model.to(args.device)

# switch to evaluation mode
model.eval()

average = 0
trials = 50
with open (args.model + 'InferenceCPU.txt', 'w') as out:
    for j in range(trials):
        total_time = 0
        for i in range(args.num_batches):
            images = dummy_input.to(args.device)

            # compute output
            with torch.cuda.amp.autocast():
                start = time.time()
                output = model(images)
                total_time += (time.time() - start)
        average = average + float(((total_time) / args.num_batches)) * 1000
        out.write(f'Total Inference Time: {total_time:.4f}\n')
        out.write(f'Latency: {float(((total_time) / args.num_batches)) * 1000} ms\n')    
    out.write(f'Average: {float(average / trials)} ms\n')

# total_time = 0
# for i in range(args.num_batches):
#     images = dummy_input.to(args.device)

#     # compute output
#     with torch.cuda.amp.autocast():
#         start = time.time()
#         output = model(images)
#         total_time += (time.time() - start)

# print(f'Total Inference Time: {total_time:.4f}')
# print(f'Latency: {float(((total_time) / args.num_batches)) * 1000} ms')
# print(f'Throughput: {int((args.batch_size * args.num_batches) / (total_time))}')