"""
Variant of the resnet module used by FCOS.
Also contains pasted code for initializing this from the Model Zoo.
"""

from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import nn
from .fcos_core_layers import FrozenBatchNorm2d, Conv2d, group_norm
import pickle
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

NUM_GROUPS = 1
WIDTH_PER_GROUP = 64
STEM_OUT_CHANNELS = 64
RES2_OUT_CHANNELS = 256
FREEZE_CONV_BODY_AT = 2
STRIDE_IN_1X1 = True

# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)


# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)


ResNet4SemanticInputs = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True))
)


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config=None
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            raise NotImplementedError
        else:
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BasicBlock(nn.Module):
    # Hacking ResNet34 block back into FCOS' implementation
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func=group_norm,
        dcn_config=[]
    ):
        super(BasicBlock, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
                
        self.conv2 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)

        for l in [self.conv1, self.conv2,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu_(out)
        
        return out


class BaseStem(nn.Module):
    def __init__(self, norm_func):
        super(BaseStem, self).__init__()

        out_channels = STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config=None
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self):
        super(StemWithFixedBatchNorm, self).__init__(
            norm_func=FrozenBatchNorm2d
        )


class ResNet(nn.Module):
    def __init__(self, stem_module=StemWithFixedBatchNorm, transformation_module=BottleneckWithFixedBatchNorm, stage_specs=ResNet50FPNStagesTo5, in_channels=STEM_OUT_CHANNELS, bottleneck_dims=True):
        super(ResNet, self).__init__()

        # Construct the stem module
        self.stem = stem_module()

        # Constuct the specified ResNet stages
        num_groups = NUM_GROUPS
        width_per_group = WIDTH_PER_GROUP
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            if not bottleneck_dims:
                out_channels = out_channels // 4
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={}
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


# Initizlization related code
def _align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))

    logger.debug("Loading pretrained ResNet weights.")
    logger.debug(f"current_keys: {current_keys}")
    logger.debug(f"loaded_keys: {loaded_keys}")
    logger.debug(f"loaded {len(loaded_keys)} keys out of {len(current_keys)}")

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.debug(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def _weights_find_replace(weight_dict, fromstr, tostr):
    new_weight_dict = {}
    for key, val in weight_dict.items():
        new_weight_dict[key.replace(fromstr, tostr)] = val
    return new_weight_dict


def _rename_basic_resnet_weights(weight_dict):
    weight_dict = _weights_find_replace(weight_dict, "_", ".")
    weight_dict = _weights_find_replace(weight_dict, ".w", ".weight")
    
    weight_dict = _weights_find_replace(weight_dict, ".bn", "_bn")
    weight_dict = _weights_find_replace(weight_dict, ".b", ".bias")
    weight_dict = _weights_find_replace(weight_dict, "_bn.s", "_bn.scale")
    weight_dict = _weights_find_replace(weight_dict, ".biasranch", ".branch")
    weight_dict = _weights_find_replace(weight_dict, ".bbox.pred", "bbox_pred")
    weight_dict = _weights_find_replace(weight_dict, "cls.score", "cls_score")
    weight_dict = _weights_find_replace(weight_dict, "res.conv1_", "conv1_")
    
    weight_dict = _weights_find_replace(weight_dict, ".biasbox", ".bbox")
    weight_dict = _weights_find_replace(weight_dict, "conv.rpn", "rpn.conv")
    weight_dict = _weights_find_replace(weight_dict, "rpn.bbox.pred", "rpn.bbox_pred")
    weight_dict = _weights_find_replace(weight_dict, "rpn.cls.logits", "rpn.cls_logits")

    weight_dict = _weights_find_replace(weight_dict, "_bn.scale", "_bn.weight")
    weight_dict = _weights_find_replace(weight_dict, "conv1_bn.", "bn1.")

    weight_dict = _weights_find_replace(weight_dict, "res2.", "layer1.")
    weight_dict = _weights_find_replace(weight_dict, "res3.", "layer2.")
    weight_dict = _weights_find_replace(weight_dict, "res4.", "layer3.")
    weight_dict = _weights_find_replace(weight_dict, "res5.", "layer4.")

    weight_dict = _weights_find_replace(weight_dict, ".branch2a.", ".conv1.")
    weight_dict = _weights_find_replace(weight_dict, ".branch2a_bn.", ".bn1.")
    weight_dict = _weights_find_replace(weight_dict, ".branch2b.", ".conv2.")
    weight_dict = _weights_find_replace(weight_dict, ".branch2b_bn..", ".bn2.")
    weight_dict = _weights_find_replace(weight_dict, ".branch2c.", ".conv3.")
    weight_dict = _weights_find_replace(weight_dict, ".branch2c_bn.", ".bn3.")
    
    weight_dict = _weights_find_replace(weight_dict, ".branch1.", ".downsample.0.")
    weight_dict = _weights_find_replace(weight_dict, ".branch1_bn.", ".downsample.1.")

    weight_dict = _weights_find_replace(weight_dict, "conv1.gn.s", "bn1.weight")
    weight_dict = _weights_find_replace(weight_dict, "conv1.gn.bias", "bn1.bias")
    weight_dict = _weights_find_replace(weight_dict, "conv2.gn.s", "bn2.weight")
    weight_dict = _weights_find_replace(weight_dict, "conv2.gn.bias", "bn2.bias")
    weight_dict = _weights_find_replace(weight_dict, "conv3.gn.s", "bn3.weight")
    weight_dict = _weights_find_replace(weight_dict, "conv3.gn.bias", "bn3.bias")

    weight_dict = _weights_find_replace(weight_dict, "downsample.0.gn.s", "downsample.1.weight")
    weight_dict = _weights_find_replace(weight_dict, "downsample.0.gn.bias", "downsample.1.bias")
    
    return weight_dict


def initialize_pretrained_resnet(model, path2pth):
    """
    Initialize the ResNet backbone of a model--
    made from the "ResNet" above--with pretrained
    Imagenet weights as per FCOS.
    """
    with open(path2pth, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    loaded_keys = sorted(list(data.keys()))
    logger.debug(f"loaded_keys before weight renaming: {loaded_keys}")
    data = _rename_basic_resnet_weights(data)
    model_state_dict = model.state_dict()
    _align_and_update_state_dicts(model_state_dict, data)
    # added
    model_state_dict = {k: torch.tensor(v) for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
