_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
crop_size = (512, 512)
model = dict(
    backbone=dict(
        type='VCMamba_EfficientFormer_B',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/work/09816/wertyuui345/ls6/VCMamba/VCMamba/trained/PlainMambaHybrid334Base.pth',
            resolution = 512,
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341)))


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

