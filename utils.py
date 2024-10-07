"""
Miscellaneous utilities such as evaluation metrics.
"""
import torch
import collections
from typing import Callable
from typing import List, Dict, Any
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os

from torchvision.transforms.functional import resize, InterpolationMode, normalize, hflip

class ImToTensor(Callable):

    def __init__(self, keyname='Image'):
        self.keyname = keyname
        self.fromPIL = transforms.PILToTensor()

    def __call__(self, datadict):
        datadict = datadict.copy()
        datadict[self.keyname] = self.fromPIL(datadict[self.keyname]).float()
        return datadict


def dict_to_gpu(dict, rank=None):
    return {key: dict[key].cuda(rank) for key in dict}


def mse(gt, pred):
    return torch.mean(torch.pow(gt - pred, 2))


# copied from later torch
def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


# DistributedDataParallel utils
# ================================
def setup_distributed(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()


def launch_distributed_experiment(fn, *args):
    world_size = torch.cuda.device_count()
    print(f'Pytorch sees {world_size} GPUs!')
    mp.spawn(fn,
             args=(world_size, *args),
             nprocs=world_size,
             join=True)
# ================================


class DictToTensors:

    def __init__(self) -> None:
        return

    def __call__(self, dct):        
        return {key: F.to_tensor(val) for key, val in dct.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MVSplit(torch.nn.Module):
    """
    To make it easier writing predictors that predict GaussianVariables.
    Automatically splits the channel dimension (dimension 1) into 
    "mean" and "variance" channels so that you can write your
    predictors otherwise to compute a regular 4D output tensor
    instead of a 5D tensor or somesuch.
    """
    def __init__(self, stdbias = None):
        super().__init__()
        if stdbias is not None:
            self.stdbias = torch.nn.parameter.Parameter(data=torch.tensor([stdbias], dtype=torch.float32), requires_grad=True)
        else:
            self.stdbias = None
 
    def forward(self, logits):
        mu, pre_sigma = torch.chunk(logits, 2, dim=1)
        if self.stdbias is not None:
            pre_sigma = pre_sigma + self.stdbias
        return torch.stack([mu, pre_sigma], dim=1)        


class MultiInputSequential(torch.nn.Sequential):
    def forward(self, *x):
        first = True
        for module in self._modules.values():
            if first:
                x = module(*x)
                first = False
            else: 
                x = module(x)
        return x

