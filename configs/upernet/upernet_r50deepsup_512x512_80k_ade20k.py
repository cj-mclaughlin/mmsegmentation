_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(num_classes=150), 
    auxiliary_head=dict(
        type='JointPrediction',
        in_features=[1024],  # either [1024] or [256, 512, 1024, 2048]
        in_index=[2],   # either use [2] for single head or [0, 1, 2, 3] for all
        channels=256,   # num channels for intermediate conv
        num_classes=150,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='JointLoss'))
    )
