"""
EfficientFormer
"""
import os
import copy
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict #VMamba

import math

from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_,  lecun_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import plain_mamba_layer
from utils import interpolate_pos_embed

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

#from rope import *

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

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
        
def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class FFN(nn.Module): #Replacement for Meta4D
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x



class Meta4D(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:

            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, drop_rate=.0, 
                drop_path_rate=0., use_layer_scale=True, 
                layer_scale_init_value=1e-5, vit_num=1, init_layer_scale = None
                ):
    blocks = []         
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(plain_mamba_layer.PlainMambaLayer(
                    embed_dims = dim,
                    use_rms_norm=False,
                    drop_path_rate=block_dpr,
                    use_post_head = True,
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate = drop_rate,
                    init_layer_scale = init_layer_scale,
                    #stride = 2
                ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))                  
    blocks = nn.ModuleList([*blocks])
    return blocks

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class MambaCNN(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.2,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution = 224,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 device=None,
                 dtype=None,
                 if_abs_pos_embed=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
                
        #self.num_tokens = 1 if if_cls_token else 0
        self.vit_num = vit_num
        
        self.d_model = self.num_features = self.embed_dim = embed_dims  # num_features for consistency with other models
        
        self.patch_embed = stem(3, embed_dims[0], act_layer)
        self.layers = layers #Save layers for forward pass readout
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.embed_dim = embed_dims[3] #Mamba Embed Dims
        self.if_abs_pos_embed = if_abs_pos_embed
        self.downsamples = downsamples
        
        self.num_tokens = 0
        print(resolution)
        self.num_patches = int(resolution // math.pow(2, sum(downsamples) + 1))
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.num_patches + self.num_tokens, self.num_patches + self.num_tokens))
            self.pos_drop = nn.Dropout(p=drop_rate)

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                vit_num=vit_num,
                                )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            # for i_emb, i_layer in enumerate(self.out_indices):
            #     if i_emb == 0 and os.environ.get('FORK_LAST3', None):
            #         layer = nn.Identity()
            #     else:
            #         layer = norm_layer(embed_dims[i_emb])
            #     layer_name = f'norm{i_layer}'
            #     self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)
        
        # original init
        self.patch_embed.apply(segm_init_weights)
        if not self.fork_feat:
            self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        # if self.fork_feat and (
        #         self.init_cfg is not None or pretrained is not None):
        #     self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        print(self.init_cfg)
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            
            # import ipdb; ipdb.set_trace()
            # state_dict_model = state_dict["model"]
            # state_dict.pop("head.weight")
            # state_dict.pop("head.bias")
            print(self.init_cfg['checkpoint'])
            # Do I need to disconnect batchnorm? I don't think so right
            # if self.patch_embed.patch_size[-1] != state_dict["model"]["patch_embed.proj.weight"].shape[-1]:
            #     state_dict_model.pop("patch_embed.proj.weight")
            #     state_dict_model.pop("patch_embed.proj.bias")
            interpolate_pos_embed(self, state_dict, self.init_cfg["resolution"])
            
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
                
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    def forward_tokens(self, x):
        outs = []
        embedCount = 0
        for idx, block in enumerate(self.network):
            if isinstance(block, Embedding):
                x = block(x)
                embedCount += 1
            else:
                for subidx, layer in enumerate(block):
                    ### FOR PLAIN MAMBA (INDEP FLATTEN)
                    if isinstance(layer, plain_mamba_layer.PlainMambaLayer):
                        if self.if_abs_pos_embed:
                            # print(self.pos_embed.shape)
                            x = x + self.pos_embed
                            x = self.pos_drop(x)   
                            
                        for i in range(self.layers[idx - embedCount] - self.vit_num, self.layers[idx - embedCount]):
                            x = block[i](x)
                        break
                    else:
                        x = layer(x)
            if self.fork_feat and idx in self.out_indices:
                # x = x.flatten(2).transpose(1, 2)
                # # norm_layer = getattr(self, f'norm{idx}')
                # # x_out = norm_layer(x[0])
                outs.append(x)
            
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        else:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            
        if self.dist:
            cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        # for image classification
        return cls_out


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


# @register_model
# def efficientformer_l1(pretrained=False, **kwargs):
#     model = EfficientFormer(
#         layers=EfficientFormer_depth['l1'],
#         embed_dims=EfficientFormer_width['l1'],
#         downsamples=[True, True, True, True],
#         vit_num=1,
#         **kwargs)
#     model.default_cfg = _cfg(crop_pct=0.9)
#     return model


# @register_model
# def efficientformer_l3(pretrained=False, **kwargs):
#     model = EfficientFormer(
#         layers=EfficientFormer_depth['l3'],
#         embed_dims=EfficientFormer_width['l3'],
#         downsamples=[True, True, True, True],
#         vit_num=4,
#         **kwargs)
#     model.default_cfg = _cfg(crop_pct=0.9)
#     return model


# @register_model
# def efficientformer_l7(pretrained=False, **kwargs):
#     model = EfficientFormer(
#         layers=EfficientFormer_depth['l7'],
#         embed_dims=EfficientFormer_width['l7'],
#         downsamples=[True, True, True, True],
#         vit_num=8,
#         **kwargs)
#     model.default_cfg = _cfg(crop_pct=0.9)
#     return model
 
 
# VIM    
# S0, S0, Vit 2 
# S1, S1 Vit 4
# S2, S2, Vit 8 
# L,  L, Vit 10
# l1, l1, Vit 2
# l3, l3, Vit 4
# l7, l7, Vit 8

# PlainMambaHybrid
# l3, l3, Vit 4
# l3, l3, Vit 6
# l3+, l3, Vit 6

#VMambaHybrid
# l3, l3, Vit 4

#EffPlainMambaHybrid
# l3,  l3 , Vit 4
#Resolution changed for MMSEG
#Note, fixed _functions in MMCV to work with this
if has_mmdet:
    @det_BACKBONES.register_module()
    def VCMamba_EfficientFormer_B(pretrained=False, **kwargs):
        model = MambaCNN(
            layers=EfficientFormer_depth['l3'],
            embed_dims=EfficientFormer_width['l3'],
            fork_feat = True,
            downsamples=[True, True, True, True],
            vit_num=4,
            drop_path_rate=0.2,
            resolution = 512,
            if_abs_pos_embed=True,
            pretrained = pretrained,
            **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9)

        return model

    # l1,  l1 , Vit 1
    @det_BACKBONES.register_module()
    def VCMamba_EfficientFormer_M(pretrained=False, **kwargs):
        model = MambaCNN(
            layers=EfficientFormer_depth['l3'],
            embed_dims=EfficientFormer_width['l1'],
            fork_feat = True,
            downsamples=[True, True, True, True],
            vit_num=4,
            drop_path_rate=0.2,
            resolution = 512,
            if_abs_pos_embed=True,
            pretrained = pretrained,
            **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9)
        return model

        # S@,  S2 , Vit 4 
    @det_BACKBONES.register_module()
    def VCMamba_EfficientFormer_S(pretrained=False, **kwargs):
        model = MambaCNN(
            layers=EfficientFormer_depth['S2'],
            embed_dims=EfficientFormer_width['S2'],
            fork_feat = True,
            downsamples=[True, True, True, True],
            vit_num=4,
            drop_path_rate=0.2,
            resolution = 512,
            if_abs_pos_embed=True, 
            pretrained = pretrained,
            **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9)

        return model