'''
Author: Chenhongyi Yang
'''

import math
from einops import repeat

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm


#Alternative PlainMambaModule without Silu + Linear multi branch
class PlainMamba2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
        default_hw_shape=None,
        stride = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4
        if default_hw_shape is not None:
            orders, inverse_orders, directions = self.get_permute_order(default_hw_shape)
            (
                self.default_permute_order,
                self.default_permute_order_inverse,
                self.default_direction
            ) = orders, inverse_orders, directions

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
        
        if stride is not None:
            self.resolution = math.ceil(d_model / stride)
            self.stride_conv = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=3, stride=stride, padding=1, groups=d_model),
                                             nn.BatchNorm2d(d_model), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = d_model
            self.stride_conv = None
            self.upsample = None

        # self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        
        self.in_conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_inner, 1 ,bias=conv_bias), 
                                    nn.BatchNorm2d(self.d_inner),         
                                    )

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner) #Tinker with this
        
        #self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.out_conv = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(self.d_inner, self.d_model, 1 ,bias=conv_bias), 
                                    nn.BatchNorm2d(self.d_model),         
                                    )
        

        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions+1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)


    def get_permute_order(self, hw_shape):
        if self.default_permute_order is not None:
            if hw_shape[0] == self.default_hw_shape[0] and hw_shape[1] == self.default_hw_shape[1]:
                return self.default_permute_order, self.default_permute_order_inverse, self.default_direction
        H, W = hw_shape
        L = H * W

        # [start, right, left, up, down] [0, 1, 2, 3, 4]

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"

            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        o2 = []
        d2 = []
        o2_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H-1, W-1
            j_d = "left"
        else:
            i, j = H-1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d2.append(1)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d2.append(2)
                else:
                    i = i - 1
                    d2.append(3)
                    j_d = "right"
        d2 = [0] + d2[:-1]

        o3 = []
        d3 = []
        o3_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o3_inverse[idx] = len(o3)
            o3.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d3.append(4)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d3.append(3)
                else:
                    j = j + 1
                    d3.append(1)
                    i_d = "down"
        d3 = [0] + d3[:-1]

        o4 = []
        d4 = []
        o4_inverse = [-1 for _ in range(L)]

        if W % 2 == 1:
            i, j = H - 1, W - 1
            i_d = "up"
        else:
            i, j = 0, W - 1
            i_d = "down"
        while j > -1:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o4_inverse[idx] = len(o4)
            o4.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d4.append(4)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d4.append(3)
                else:
                    j = j - 1
                    d4.append(2)
                    i_d = "down"
        d4 = [0] + d4[:-1]

        o1 = tuple(o1)
        d1 = tuple(d1)
        o1_inverse = tuple(o1_inverse)

        o2 = tuple(o2)
        d2 = tuple(d2)
        o2_inverse = tuple(o2_inverse)

        o3 = tuple(o3)
        d3 = tuple(d3)
        o3_inverse = tuple(o3_inverse)

        o4 = tuple(o4)
        d4 = tuple(d4)
        o4_inverse = tuple(o4_inverse)

        return (o1, o2, o3, o4), (o1_inverse, o2_inverse, o3_inverse, o4_inverse), (d1, d2, d3, d4)
    
    def forward(self, x):
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            
        # print(x.shape)
        _, _, H, W = x.shape
        
        x = self.in_conv(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        batch_size, L, _ = x.shape
        E = self.d_inner

        conv_state, ssm_state = None, None
        
        # x = self.in_proj(x)  # [B, L, E]
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.get_permute_order([H,W])
        direction_Bs = [self.direction_Bs[d, :] for d in directions]  # each [L, d_state]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in direction_Bs]
        ys = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_o, :]
            for o, inv_o, dB in zip(orders, inverse_orders, direction_Bs)
        ]
        y = sum(ys)
            
        out = self.norm(y)
        
        out = out.transpose(1, 2).unflatten(2, (H, W))

        #out = self.out_proj(out)
        
        if self.upsample is not None:
            out = self.upsample(out)
        
        # print(x.shape)
        out = self.out_conv(out)
        # print('done')
        
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out

#Taken from EfficientFormer
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

class PlainMambaLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        use_rms_norm,
        use_post_head,
        drop_path_rate,
        mlp_drop_rate,
        init_layer_scale,
        mlp_ratio,
        bias = True,
        stride = None
        # conv_bias=True,
        # init_layer_scale=None,
        # default_hw_shape=None,
    ):
        super(PlainMambaLayer, self).__init__()
        #mamba_cfg.update({'d_model': embed_dims})
        
        # print(embed_dims)
        # if stride is not None:
        #     self.resolution = math.ceil(embed_dims / stride)
        #     self.stride_conv = nn.Sequential(nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=stride, padding=1, groups=embed_dims),
        #                                      nn.BatchNorm2d(embed_dims), )
        #     self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        # else:
        #     self.resolution = embed_dims
        #     self.stride_conv = None
        #     self.upsample = None
            
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
            self.norm2 = RMSNorm(embed_dims)
        else:
            self.norm = nn.BatchNorm2d(embed_dims)
            self.norm2 = nn.BatchNorm2d(embed_dims)

        self.mamba = PlainMamba2D(embed_dims, init_layer_scale=init_layer_scale, bias = bias, stride = None)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        #self.mlp = LinearMlp(embed_dims, mlp_hidden_dim, act_layer=nn.GELU, drop=mlp_drop_rate)
        self.mlp = Mlp(embed_dims, mlp_hidden_dim, act_layer=nn.GELU, drop=mlp_drop_rate, mid_conv=True)
    
    def forward(self, x):
        # if self.stride_conv is not None:
        #     x = self.stride_conv(x)
        
        #hw_shape = (x.shape[2], x.shape[3])
        # x = self.flatten(x)
        x = x + self.drop_path(self.mamba(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('fin')
        # x = self.unflatten(x, hw_shape)
        
        # if self.upsample is not None:
        #     x = self.upsample(x)
            
        return x