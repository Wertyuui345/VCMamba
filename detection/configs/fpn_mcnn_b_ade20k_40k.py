_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
crop_size = (512, 512)
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VCMamba_EfficientFormer_B',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/work/09816/wertyuui345/ls6/VCMamba/VCMamba/trained/PlainMambaHybrid334Base.pth',
            resolution = 512,
            ),
        ),
    neck=dict(in_channels=[64, 128, 320, 512],),
    decode_head=dict(num_classes=150,),
    #Try this, pulled from vim
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
    )


gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
