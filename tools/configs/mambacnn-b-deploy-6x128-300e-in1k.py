_base_ = [
    '../configs/_base_/datasets/imagenet_bs64_swin_224_lmdb.py',
    '../configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PlainMamba',
        layers=[4, 4, 12, 6],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=4,
        downsamples=[True, True, True, True],
        pool_size=3,
        norm_layer=dict(type='LayerNorm', eps=1e-6),
        act_layer=dict(type='GELU'),
        num_classes=1000,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.2,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        resolution=224,
        fork_feat=False,
        init_cfg=dict(type='Pretrained', checkpoint='../VCMamba/trained/PlainMambaHybrid334Base.pth'),
        vit_num=4,
        distillation=True,
        if_abs_pos_embed=False
    ),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=dict(
        type='ClsHead',
        num_classes=1000,
        in_channels=768,
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0
        ),
        topk=(1, 5)
    )
)

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1
)

# Testing settings
test_cfg = dict(
    type='TestLoop',
    test_interval=1
)

# # Data settings
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=4,
#     train=dict(
#         type='ImageNet',
#         data_prefix='data/imagenet',
#         ann_file='data/imagenet/train_list.txt'
#     ),
#     val=dict(
#         type='ImageNet',
#         data_prefix='data/imagenet',
#         ann_file='data/imagenet/val_list.txt'
#     ),
#     test=dict(
#         type='ImageNet',
#         data_prefix='data/imagenet',
#         ann_file='data/imagenet/val_list.txt'
#     )
# )

# Optimizer settings
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule settings
lr_config = dict(
    policy='step',
    step=[30, 60],
    gamma=0.1
)

# Runtime settings
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Misc settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/efficientformer'
load_from = None
resume_from = None
auto_resume = False
gpu_ids = range(0, 1)

# Evaluation settings
evaluation = dict(
    interval=1,
    metric='accuracy'
)
