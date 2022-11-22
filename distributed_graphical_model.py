from torch.nn.parallel import DistributedDataParallel
#from .graphical_model import NeuralGraphicalModel
from typing import List, Union, Optional
from random_variable import RandomVariable
from graphical_model import NeuralGraphicalModel


class DDPNGM(DistributedDataParallel):
    #Subclasses DDP in order to expose NGM attributes after wrapping it
    # I have checked that none of these clash with DDP members

    def __init__(self, module, *args, **kwargs):
        assert isinstance(module, NeuralGraphicalModel)
        super().__init__(module, *args, find_unused_parameters=True, **kwargs)
        module.forward_sample = self.forward  # little cursed but we'll live

    def addvar(self, variable):
        self.module.addvar(variable)

    def __len__(self):
        return len(self.module)

    def __iter__(self):
        return iter(self.module.random_variables)

    def __getitem__(self, key):
        return self.module[key]

    def __contains__(self, variable):
        return variable in self.module

    def calibration_parameters(self):
        return self.module.calibration_parameters()

    def reset_calibration(self):
        return self.module.reset_calibration()

    def cuda(self, device=None):
        return self.module.cuda(device)

    def predict_marginals(self, gt, to_predict_marginal: Union[List[RandomVariable], List[str]], \
                            force_predicted_input: Union[List[RandomVariable], List[str]] = [], \
                            samples_per_pass: int = 1, num_passes=1):
        return self.module.predict_marginals(gt, to_predict_marginal, force_predicted_input, samples_per_pass, num_passes)

    def loss(self, gt, keep_unsupervised_loss=True, samples_in_pass=1, force_predicted_input=[], summary_writer=None, **summary_writer_kwargs):
        return self.module.loss(gt, keep_unsupervised_loss, samples_in_pass, force_predicted_input, summary_writer, **summary_writer_kwargs)

    def validate(self):
        return self.module.validate()

