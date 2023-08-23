# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

import torch
import torchvision.transforms as trans
from mmcv.transforms import BaseTransform

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PatchStd(BaseTransform):
    def __init__(self,
                 key: str,
                 std_kernel_size: int = 6):
        self.key = key
        self.std_kernel_size = std_kernel_size

    def patch_std_cal(self, tensor: torch.Tensor) -> torch.Tensor:
        c, _, _ = tensor.shape
        unfold = torch.nn.Unfold(kernel_size=self.std_kernel_size, stride=1)
        tensor = unfold(tensor)
        tensor = tensor.reshape(c, tensor.shape[0] // c, tensor.shape[1])
        tensor = torch.std(tensor, 1, unbiased=False)
        tensor = torch.mean(tensor, 1)
        return tensor

    def transform(self, results: dict) -> dict:
        tensor = trans.ToTensor()(copy.deepcopy(results[self.key]))
        patch_std = self.patch_std_cal(tensor)
        patch_std = torch.unsqueeze(patch_std, 0)
        results['patch_std'] = patch_std
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'std_kernel_size={self.std_kernel_size}')
        return repr_str
