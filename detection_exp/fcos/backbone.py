from collections import OrderedDict
from torch import nn
from . import fpn as fpn_module
from . import resnet
from .fcos_core_layers import conv_with_kaiming_uniform


# Constants copied over from FCOS config
RES2_OUT_CHANNELS = 256
BACKBONE_OUT_CHANNELS = 256
USE_GN = False
USE_RELU = False


def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = RES2_OUT_CHANNELS
    out_channels = BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            USE_GN, USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model