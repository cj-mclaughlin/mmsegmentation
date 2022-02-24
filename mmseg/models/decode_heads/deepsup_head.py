# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.ops import Encoding, resize
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class DeepSupHead(BaseDecodeHead):
    def __init__(self,
                 in_features=[256, 512, 1024, 2048],
                 in_index=[0, 1, 2, 3],
                 num_classes=150,
                 dropout=0.1,
                 loss=dict(
                     type="ClassificationLoss",
                     loss_weight=0.4
                 ),
                 **kwargs):
        super(DeepSupHead, self).__init__(
            input_transform='multiple_select', 
            in_channels=in_features,
            in_index=in_index,
            channels=1024, 
            num_classes=num_classes,
            **kwargs
            )
        
        self.cls_heads = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout2d(p=dropout, inplace=True),
                nn.Conv2d(in_channels=in_features[i], out_channels=num_classes, kernel_size=1),
                nn.Sigmoid()
            ).to("cuda")
            for i in range(len(in_features))
        ]

        self.cls_loss = build_loss(loss)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        cls_predictions = [
            self.cls_heads[i](inputs[i]) for i in range(len(inputs))
        ]

        return cls_predictions

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
        cls_predictions = self.forward(inputs)
        losses = self.cls_loss(cls_predictions, gt_semantic_seg)
        ret_dict = dict()
        ret_dict["class_loss"] = losses
        return ret_dict

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, ignore se_loss."""
        return self.forward(inputs)