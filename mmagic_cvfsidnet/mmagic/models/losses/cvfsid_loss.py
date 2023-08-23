# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS
from mmagic.structures import DataSample

def re_pad_func(tensor: torch.Tensor, re_pad: int = 20) -> torch.Tensor:
    return F.pad(input=tensor[..., re_pad: -re_pad, re_pad: -re_pad],
                 pad=(re_pad, re_pad, re_pad, re_pad),
                 mode='reflect')


def mse_loss(pred: torch.Tensor,
             target: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    """mse_loss loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        reduction: Nothing else, just reduction.

    Returns:
        Tensor: Calculated mse_loss loss.
    """
    return F.mse_loss(pred, target, reduction=reduction)


def loss_aug(clean: torch.Tensor, clean1: torch.Tensor,
             noise_w: torch.Tensor, noise_w1: torch.Tensor,
             noise_b: torch.Tensor, noise_b1: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    loss1 = mse_loss(clean1, clean, reduction)
    loss2 = mse_loss(noise_w1, noise_w, reduction)
    loss3 = mse_loss(noise_b1, noise_b, reduction)
    loss = loss1 + loss2 + loss3
    return loss


def loss_main(input_noise: torch.Tensor,
              input_noise_pred: torch.Tensor,
              clean: torch.Tensor,
              clean1: torch.Tensor,
              clean2: torch.Tensor,
              clean3: torch.Tensor,
              noise_b: torch.Tensor,
              noise_b1: torch.Tensor,
              noise_b2: torch.Tensor,
              noise_b3: torch.Tensor,
              noise_w: torch.Tensor,
              noise_w1: torch.Tensor,
              noise_w2: torch.Tensor,
              std: torch.Tensor,
              gamma: float = 1.0,
              reduction: str = 'mean') -> torch.Tensor:
    loss1 = mse_loss(input_noise_pred, input_noise)

    loss2 = mse_loss(clean1, clean, reduction)
    loss3 = mse_loss(noise_b3, noise_b, reduction)
    loss4 = mse_loss(noise_w2, noise_w, reduction)
    loss5 = mse_loss(clean2, clean, reduction)

    loss6 = mse_loss(clean3, torch.zeros_like(clean3), reduction)
    loss7 = mse_loss(noise_w1, torch.zeros_like(noise_w1), reduction)
    loss8 = mse_loss(noise_b1, torch.zeros_like(noise_b1), reduction)
    loss9 = mse_loss(noise_b2, torch.zeros_like(noise_b2), reduction)

    sigma_b = torch.std(noise_b.reshape([noise_b.shape[0], noise_b.shape[1], -1]), -1)
    sigma_w = torch.std(noise_w.reshape([noise_w.shape[0], noise_w.shape[1], -1]), -1)
    blur_clean = F.avg_pool2d(clean, kernel_size=6, stride=1, padding=3)
    clean_mean = torch.mean(torch.square(torch.pow(blur_clean, gamma).reshape([clean.shape[0], clean.shape[1], -1])),
                            -1)  # .detach()
    sigma_wb = torch.sqrt(clean_mean * torch.square(sigma_w) + torch.square(sigma_b))

    loss10 = mse_loss(sigma_wb, std, reduction)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10
    return loss


@MODELS.register_module()
class CVFSIDLoss(nn.Module):
    """
        The loss func of CVF-SID Net.

        Refs:
            official paper: https://arxiv.org/abs/2203.13009
            official code: https://github.com/Reyhanehne/CVF-SID_PyTorch/blob/main/src/model/loss.py
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 re_pad: int = 10,
                 gamma: float = 1.0,
                 reduction: str = 'mean'
                 ) -> None:
        super().__init__()

        self.loss_weight = loss_weight
        self.re_pad = re_pad
        self.gamma = gamma
        self.reduction = reduction

        _reduction_modes = ['none', 'mean', 'sum']
        if self.reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self,
                pred: torch.Tensor,
                edit_model_self: nn.Module,
                data_samples: Optional[List[DataSample]]) -> torch.Tensor:

        noise_w, noise_b, clean, noise_im = pred

        noise_w1, noise_b1, clean1, _ = edit_model_self.generator(re_pad_func(clean, self.re_pad))
        noise_w2, noise_b2, clean2, _ = edit_model_self.generator(re_pad_func(clean + torch.pow(clean, self.gamma) * noise_w, self.re_pad))  # 1
        noise_w3, noise_b3, clean3, _ = edit_model_self.generator(re_pad_func(noise_b, self.re_pad))

        noise_w4, noise_b4, clean4, _ = edit_model_self.generator(
            re_pad_func(clean + torch.pow(clean, self.gamma) * noise_w - noise_b, self.re_pad))  # 2
        noise_w5, noise_b5, clean5, _ = edit_model_self.generator(
            re_pad_func(clean - torch.pow(clean, self.gamma) * noise_w + noise_b, self.re_pad))  # 3
        noise_w6, noise_b6, clean6, _ = edit_model_self.generator(
            re_pad_func(clean - torch.pow(clean, self.gamma) * noise_w - noise_b, self.re_pad))  # 4
        noise_w10, noise_b10, clean10, _ = edit_model_self.generator(
            re_pad_func(clean + torch.pow(clean, self.gamma) * noise_w + noise_b, self.re_pad))  # 5

        noise_w7, noise_b7, clean7, _ = edit_model_self.generator(re_pad_func(clean + noise_b, self.re_pad))  # 6
        noise_w8, noise_b8, clean8, _ = edit_model_self.generator(re_pad_func(clean - noise_b, self.re_pad))  # 7
        noise_w9, noise_b9, clean9, _ = edit_model_self.generator(re_pad_func(clean - torch.pow(clean, self.gamma) * noise_w, self.re_pad))  # 8

        noise_pred = clean + torch.pow(clean, self.gamma) * noise_w + noise_b

        std = torch.cat(data_samples.patch_std, 0).cuda()
        loss = loss_main(noise_im, noise_pred,
                         clean, clean1, clean2, clean3,
                         noise_b, noise_b1, noise_b2, noise_b3,
                         noise_w, noise_w1, noise_w2,
                         std, self.gamma)

        loss_neg1 = loss_aug(clean, clean4, noise_w, noise_w4, noise_b, -noise_b4)
        loss_neg2 = loss_aug(clean, clean5, noise_w, -noise_w5, noise_b, noise_b5)
        loss_neg3 = loss_aug(clean, clean6, noise_w, -noise_w6, noise_b, -noise_b6)

        loss_neg4 = loss_aug(clean, clean7, torch.zeros_like(noise_w), noise_w7, noise_b, noise_b7)
        loss_neg5 = loss_aug(clean, clean8, torch.zeros_like(noise_w), noise_w8, noise_b, -noise_b8)
        loss_neg6 = loss_aug(clean, clean9, -noise_w, noise_w9, torch.zeros_like(noise_b), noise_b9)
        loss_neg7 = loss_aug(clean, clean10, noise_w, noise_w10, noise_b, noise_b10)

        loss_total = loss + .1 * (loss_neg1 + loss_neg2 + loss_neg3 + loss_neg4 + loss_neg5 + loss_neg6 + loss_neg7)

        return self.loss_weight * loss_total


