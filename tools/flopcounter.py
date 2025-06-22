from fvcore.nn import FlopCountAnalysis
import torch

import sys
sys.path.insert(0, '/scratch/09816/wertyuui345/ls6/VCMamba/VCMamba/model/')
import efficientformer

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
    'l3+': [4, 4, 12, 8],
}


    
image = torch.rand(128, 3, 224, 224).to('cuda:0')
model = efficientformer.EfficientFormer(
        layers=EfficientFormer_depth['l3'],
        embed_dims=EfficientFormer_width['l3'],
        downsamples=[True, True, True, True],
        vit_num=4,
        #drop_path_rate=0.1,
        #num_classes = 200,
        resolution = 224,
        #stride = 4,
        rms_norm=True, 
        residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
        if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", 
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, if_bidirectional=True)

model.to('cuda:0')

print("hello")
flops = FlopCountAnalysis(model, image)

print(flops.total())
print(flops.by_operator())
print(flops.by_module())