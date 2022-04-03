# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.ops import Encoding, resize
from ..builder import HEADS, build_loss
from ..losses import accuracy
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class JointPredictionHead(BaseDecodeHead):
    def __init__(self,
                 in_channels=[1024],
                 in_index=[2],
                 channels=256,
                 num_classes=150,
                 norm_cfg = dict(type='SyncBN', requires_grad=True),
                 loss_decode=dict(
                     type="JointLoss",
                     loss_weight=0.4,
                 ),
                 class_loss_weight=0.25,
                 **kwargs):
        super(JointPredictionHead, self).__init__(
            input_transform='multiple_select', 
            in_channels=in_channels,
            in_index=in_index,
            channels=channels, 
            num_classes=num_classes,
            **kwargs
            )

        self.in_index = in_index
        self.class_loss_weight = class_loss_weight

        self.bottleneck = ConvModule(in_channels[0], channels, kernel_size=1, norm_cfg=norm_cfg)

        # classification head
        self.classification_head = nn.Sequential(
                            self.bottleneck,
                            self.dropout,
                            nn.AdaptiveMaxPool2d((1,1)),
                            self.conv_seg
                        ).to("cuda")

        # segmentation head
        self.segmentation_head = nn.Sequential(
            self.bottleneck,
            self.dropout,
            self.conv_seg
        ).to("cuda")

        self.loss = build_loss(loss_decode)

    def forward(self, inputs):
        """Forward function."""
        """Compute both classification and segmentation with same weights."""
        inputs = self._transform_inputs(inputs)

        segmentation = self.segmentation_head(inputs[0])
        classification = self.classification_head(inputs[0])
    
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

        # class_loss.backward(retain_graph=True)
        # class_norm = torch.norm(self.conv_seg.weight.grad, p=2)
        # self.zero_grad()
        
        # class_weight = (seg_norm / class_norm) # * self.loss_weight
        # class_loss = class_loss * class_weight

        loss = self.class_loss_weight * class_loss + (1 - self.class_loss_weight) * seg_loss
        loss.backward(retain_graph = True)
        joint_norm = torch.norm(self.conv_seg.weight.grad, p=2)
        loss = loss * (seg_norm.detach() / joint_norm.detach())
        self.zero_grad()

        loss_dict = dict()
        loss_dict["loss_joint"] = loss
        loss_dict['acc_seg'] = accuracy(
            segmentation, gt_semantic_seg.squeeze(1), ignore_index=self.ignore_index)
        return loss_dict

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, return only segmentation"""
        return self.forward(inputs)[0]