# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS, build_loss
from ..losses import accuracy
from .decode_head import BaseDecodeHead
from .psp_head import PPM



@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, 
                pool_scales=(1, 2, 3, 6),
                loss_decode=dict(
                     type="JointLoss",
                     loss_weight = 1.0
                ),
                class_loss_weight=0.25,
                **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select',
            **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            self.conv_seg
        ).to("cuda")

        # segmentation head
        self.segmentation_head = nn.Sequential(
            self.conv_seg
        ).to("cuda")

        self.class_loss_weight = class_loss_weight
        self.loss = build_loss(loss_decode)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        segmentation, classification = self.segmentation_head(output), self.classification_head(output)
        return segmentation, classification

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        segmentation, classification = self.forward(inputs)
        segmentation = resize(segmentation, size=gt_semantic_seg.size()[2:], mode="bilinear")
        seg_loss, class_loss = self.loss((segmentation, classification), gt_semantic_seg)
        # get l2 norm of segmentation loss
        seg_loss.backward(retain_graph=True)
        seg_norm = torch.norm(self.conv_seg.weight.grad, p=2)
        self.zero_grad()

        assert self.conv_seg.weight.grad is None or torch.allclose(self.conv_seg.weight.grad, torch.zeros_like(self.conv_seg.weight.grad))
        # class_loss.backward(retain_graph=True)
        # class_norm = torch.norm(self.conv_seg.weight.grad, p=2)
        # self.zero_grad()
        
        # class_weight = (seg_norm / class_norm)
        # class_loss = class_loss * class_weight

        loss = self.class_loss_weight * class_loss + (1 - self.class_loss_weight) * seg_loss
        loss.backward(retain_graph = True)
        joint_norm = torch.norm(self.conv_seg.weight.grad, p=2)
        loss = loss * (seg_norm.detach() / joint_norm.detach())
        self.zero_grad()

        loss_dict = dict()
        
        loss_dict['acc_seg'] = accuracy(
            segmentation, gt_semantic_seg.squeeze(1), ignore_index=self.ignore_index)
        loss_dict["loss_joint"] = loss
        return loss_dict

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, return only segmentation"""
        return self.forward(inputs)[0]