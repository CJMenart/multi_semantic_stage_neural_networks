import torch
from torch.distributions.mixture_same_family import MixtureSameFamily


class MixtureSameFamilyFixed(MixtureSameFamily):
    """
    This bugfix will hopefully be in PyTorch proper someday, but for now, I need 
    my code to work!
    """

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = len(self.batch_shape)
        cat_batch_ndims = len(self.mixture_distribution.batch_shape)
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x
