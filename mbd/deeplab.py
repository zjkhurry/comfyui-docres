"""DeepLab-V3+ segmentation used by MBD to extract the document mask.

Forward-only copy from the original repo (data/MBD/model/deep_lab_model).
Training helpers and the distributed sync-batchnorm path were removed: DocRes
inference instantiates this with `sync_bn=None`, so `nn.BatchNorm2d` is always
used and no gradients ever flow through it.
"""

import torch.nn as nn
import torch.nn.functional as F

from .aspp import build_aspp
from .decoder import build_decoder
from .resnet import ResNet101


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d
        self.backbone = ResNet101(output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        return F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
