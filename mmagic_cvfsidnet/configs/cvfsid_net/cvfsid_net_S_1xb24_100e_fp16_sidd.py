_base_ = '../_base_/default_runtime_epoch.py'

max_epochs = 100

batch_size = 24

experiment_name = f'cvfsid_1xb{batch_size}_{max_epochs}e_sidd'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='CVFSIDNet',
        color_channels=3,
        mid_channels=64,
        num_layers=(17, (10, 4, 4)),
        act_cfg=dict(type='ReLU', inplace=True),
        restructure_in_test=True),
    pixel_loss=dict(type='CVFSIDLoss',
                    loss_weight=1.0,
                    re_pad=10,
                    gamma=1.0,
                    reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.])
)

# data preprocessing
train_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PatchStd', key='img', std_kernel_size=6),
    dict(type='Crop', keys=['img'], crop_size=(40, 40), random_crop=True),
    dict(type='Flip', keys=['img'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img'], transpose_ratio=0.5),
    dict(type='NumpyPad', keys=['img'], padding=((10, 10), (10, 10), (0, 0)), mode='reflect'),
    dict(type='PackInputs', meta_keys=['patch_std'])
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackInputs')
]


# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=8,
    batch_size=batch_size,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='my_sidd', task_name='my_denoising'),
        data_root='./data/SIDD_Val_and_GT_Benchamrk/train',
        data_prefix=dict(img='noisy'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    num_workers=8,
    batch_size=batch_size,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='my_sidd', task_name='my_denoising'),
        data_root='./data/SIDD_Val_and_GT_Benchamrk/val/',
        data_prefix=dict(img='noisy', gt='gt'),
        pipeline=val_pipeline)
)

test_dataloader = val_dataloader


# metrics setting
val_evaluator = [
    dict(type='MAE'),
    dict(type='PSNR'),
    dict(type='SSIM')
]
test_evaluator = val_evaluator


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='AmpOptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-9, amsgrad=True)
)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=24),
    dict(type='StepLR', begin=0, end=max_epochs, by_epoch=True, step_size=10, gamma=0.99)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        out_dir=save_dir,
        save_best='PSNR',
        rule='greater')
)

# python tools/train.py configs/cvfsid_net/cvfsid_net_S_1xb24_100e_fp16_sidd.py

# python tools/test.py configs/cvfsid_net/cvfsid_net_S_1xb24_100e_fp16_sidd.py work_dirs/cvfsid_1xb24_100e_sidd/best_PSNR_epoch_87.pth

# python tools/analysis_tools/get_flops.py configs/cvfsid_net/cvfsid_net_S_1xb24_100e_fp16_sidd.py --shape 3 256 256
