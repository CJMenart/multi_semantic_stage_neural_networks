# Packages to keep
import math
import torch
import torch.nn.functional as F
from torch import nn
from .fcos_core_layers import Scale

# Packages we want to delete
#from .inference import make_fcos_postprocessor
NUM_CONVS = 4
PRIOR_PROB = 0.01 # initialized output bias
# we are not doing the fancy post-paper stuff
#USE_DCN_IN_TOWER = False


class FCOSHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels=None, classpred=False):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        if hid_channels is None:
            hid_channels = in_channels
        num_classes = out_channels

        cls_tower = []
        for i in range(NUM_CONVS):
           
            conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels if i==0 else hid_channels,
                    hid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, hid_channels))
            cls_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        self.cls_logits = nn.Conv2d(
            hid_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        if classpred:
            prior_prob = PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        #self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
    
        #print(f"x.shape start of FCOSHead: {x.shape}")
        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        #print(f"x.shape end of FCOSHead: {logits.shape}")    

        # scale-depending param moved outside this Module
        #bbox_pred = self.scales[l](self.bbox_pred(box_tower))    
        
        return logits

