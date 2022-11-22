"""
Various bespoke/exotic/uncommon subclasses of RandomVariable, put in here to avoid cluttering random_variable.py
"""
import torch
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import List
from torch.nn.parameter import Parameter
from random_variable import *


class ProbSpaceBooleanVariable(BooleanVariable):
    """
    Just like ProbSpaceCategoricalVariable, ProbSpaceBooleanVariable is a BooleanVariable which 
    is parameterized by feeding it probabilities, not logits or log-probabilities. 
    This is usually so that you can predict it using hard rules based on external knowledge that 
    takes the form of explicit probabilities such as 1 or 0.
    WARNING: There is no temperature scaling! 
    """
    def __init__(self, **kwargs):
        super(BooleanVariable, self).__init__(**kwargs)
        
    def _calibration_parameters(self):
        return []
        
    def _dist_params_to_model(self, dist_params):
        """
        probs emerge in format 'batch, other dimensions' 
        """
        try:
            model = torch.distributions.Bernoulli(probs=dist_params, validate_args=True)
        except ValueError as e:
            print(f"{self.name} recieved value error: {e}")
            print(f"The invalid probs: {dist_params}")
            raise ValueError
        return model
        
    def estimate_distribution_from_dists(self, dist_params, weights): 
        return self._estimate_distribution_from_probs(dist_params, weights)    