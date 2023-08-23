# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init
from mmagic.registry import MODELS


class GenClean(BaseModule):
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 kernel_size=3,
                 num_layers=17,
                 act_cfg=dict(type='ReLU', inplace=True)
                 ):
        super(GenClean, self).__init__(init_cfg=None)

        layers = []
        for i in range(num_layers - 1):
            layers.append(
                ConvModule(
                    in_channels=color_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    act_cfg=act_cfg)
            )

        layers.append(
            ConvModule(
                in_channels=mid_channels,
                out_channels=color_channels,
                kernel_size=1,
                act_cfg=None)
        )

        self.gen_clean = nn.Sequential(*layers)

    def forward(self, x):
        x = self.gen_clean(x)
        return x


class GenNoise(BaseModule):
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 kernel_size=3,
                 num_layers=(10, 4, 4),
                 act_cfg=dict(type='ReLU', inplace=True)
                 ):
        super(GenNoise, self).__init__(init_cfg=None)

        # body branch
        m = []
        for i in range(num_layers[0]):
            m.append(
                ConvModule(
                    in_channels=color_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    act_cfg=act_cfg)
            )

        self.gen_noise_body = nn.Sequential(*m)

        # noise_w parts
        gen_noise_w = []

        for i in range(num_layers[1]):
            gen_noise_w.append(
                ConvModule(in_channels=mid_channels,
                           out_channels=mid_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2,
                           act_cfg=act_cfg)
            )

        gen_noise_w.append(
            ConvModule(in_channels=mid_channels,
                       out_channels=color_channels,
                       kernel_size=1,
                       act_cfg=None)
        )

        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        # noise_b parts
        gen_noise_b = []

        for i in range(num_layers[2]):
            gen_noise_b.append(
                ConvModule(in_channels=mid_channels,
                           out_channels=mid_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2,
                           act_cfg=act_cfg)
            )

        gen_noise_b.append(
            ConvModule(in_channels=mid_channels,
                       out_channels=color_channels,
                       kernel_size=1,
                       act_cfg=None)
        )

        self.gen_noise_b = nn.Sequential(*gen_noise_b)

    def forward(self, x):
        x = self.gen_noise_body(x)
        noise_w = self.gen_noise_w(x)
        noise_b = self.gen_noise_b(x)

        m_w = torch.mean(torch.mean(noise_w, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_w = noise_w - m_w
        m_b = torch.mean(torch.mean(noise_b, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_b = noise_b - m_b

        return noise_w, noise_b


@MODELS.register_module()
class CVFSIDNet(BaseModule):
    """
    CVF-SID Net: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image.

    Refs:
        official paper: https://arxiv.org/abs/2203.13009
        official code: https://github.com/Reyhanehne/CVF-SID_PyTorch

    Args:
        color_channels (int): The channels of input images.
        mid_channels (int): The channels of 'gen_clean' and 'gen_noise'.
        num_layers (tuple(int, tuple)): Config for network architecture, which can be regarded as
            (config of gen_clean, config of gen_noise)
        act_cfg (dict, optional): Default: dict(type='ReLU', inplace=True).
        restructure_in_test (bool): To tell network in training or in val/test. This func is applied for some network which it
            architecture relays on phase, such as CVF-SID Net, RepVGG, RMNet...
            For example, In CVF-SID Net, it removes the 'gen_noise' in val/test. But in training, it keeps 'gen_noise'.
            So we should set 'restructure_in_test=True' to tell it which phase it is.
            This func can be applied in RepVGG or some RepVGG-like network, too.
    """
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 num_layers=(17, (10, 4, 4)),
                 act_cfg=dict(type='ReLU', inplace=True),
                 restructure_in_test=True
                 ):
        super(CVFSIDNet, self).__init__(init_cfg=None)
        self.gen_clean = GenClean(color_channels=color_channels,
                                  mid_channels=mid_channels,
                                  kernel_size=3,
                                  num_layers=num_layers[0],
                                  act_cfg=act_cfg)
        self.gen_noise = GenNoise(color_channels=color_channels,
                                  mid_channels=mid_channels,
                                  kernel_size=3,
                                  num_layers=num_layers[1],
                                  act_cfg=act_cfg)

    def forward(self, x, mode='loss'):
        clean = self.gen_clean(x)
        noise_w, noise_b = self.gen_noise(x - clean)

        if mode == 'loss':
            return noise_w, noise_b, clean, x
        else:
            return clean

    def init_weights(self):
        super(CVFSIDNet, self).init_weights()

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')