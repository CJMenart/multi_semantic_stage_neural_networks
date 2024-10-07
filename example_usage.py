"""
An attempt to show all the important things about using this codebase in the form of a toy example.
"""
from .graphical_model import NeuralGraphicalModel
from .random_variable import CategoricalVariable
import torch
import numpy as np
from .utils import dict_to_gpu
from torch.utils.tensorboard import SummaryWriter
# For distributing training
from .distributed_graphical_model import DDPNGM
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def example_usage():
    world_size = torch.cuda.device_count()
    print(f'Pytorch sees {world_size} GPUs!')
    mp.spawn(example_usage_thread,
             args=(world_size,),
             nprocs=world_size,
             join=True)


# DistributedDataParallel boilerplate
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def example_usage_thread(rank, world_size):
    NCAT = 3  # all our vars in this example will have same number of categories
    TRAIN_ITER = 1000
    TEST_ITER = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    CAL_ITER = 10

    setup_distributed(rank, world_size)

    # set up model--the NeuralGraphicalModel is your actual NGM
    graph = NeuralGraphicalModel()
    """
    You can construct RandomVariables, such as CategoricalRandomVariable
    or GaussianRandomVariable, and add them to the graph. In traditional deep-learning
    speak, you can think of each variable as a "task". Except that any inputs to 
    the model are also RandomVariables--there is no fundamental difference
    between inputs/outputs. 
    
    The variable type encodes an 
    assumption about the conditional distribution of each task's values. So
    for instance, CategoricalRandomVariable assumes that you are trying to 
    perform e.g. a classification task, where there is always one correct discrete output. 
    
    In the RandomVariable constructor, predictor_fns is a list of all of the predictors, 
    or in DL-speak, networks, or network heads, trying to perform that task. 
    predictor_fns and per_prediction_parents are 
    lists because you can have multiple predictors, each with its own set of parents. Their
    outputs will be ensembled/fused together at inference time. 
    
    In this example, though, each variable only has one predictor. 
    Check out random_variable.py for more RandomVariables
    """
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', 
        predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', 
        predictor_fns=[torch.nn.Linear(NCAT, NCAT)], 
        per_prediction_parents=[[graph['A']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', 
        predictor_fns=[torch.nn.Linear(NCAT, NCAT)], 
        per_prediction_parents=[[graph['B']]]))
    """
    The graph we just made corresponds to predicting B from A and then 
    predicting C from B:
    A  ->  B  ->  C 
    The structure does not have to be so simple. Technically you can make 
    pretty much any arbitrary directed graph, although in the fully-general case,
    any arbitrary graph won't necessarily work well or even be able to compute 
    probabilities.
    RandomVariable.add_predictor() can be used to create cycles (you can add predictors 
    to a RandomVariable you already constructed.) But you SHOULD finish constructing 
    the graph before you run it or wrap it in DDPNGM.
    
    One more note about advanced usage: If you share weights by passing the same predictor
    to multiple Variables, it will be run multiple times, potentially performing 
    redundant computation and wasting memory. If you want to share features (such as e.g.
    having one network backbone with multiple heads, a common pattern in this repo) it is
    better to declare the shared feature as a "variable" using DeterministicContinuousVariable.
    This class represents a node in your directed graph which isn't *really* a random variable,
    just a Tensor that you want to share. It will be computed only once in a forward 
    pass, and then can be consumed by multiple different predictors by making it 
    their parent.
    """

    """
    In the likely event that you're running a big model, you may want to 
    distribute training across multiple GPUs. Because of how I wrote 
    NeuralGraphicalModel(), you can't really just use Pytorch's DataParallel objects 
    directly, but I made a custom wrapper so you can do this: 
    """
    graph = graph.cuda(rank)
    graph = DDPNGM(graph, device_ids=[rank])

    """
    As input ('gt' in function signatures) the NeuralGraphicalModel() expects a dictionary
    where keys are RandomVariables (or just their string names) and the values are tensors
    representing thier observed value. (During training, both your inputs and outputs/targets
    will be passed in this way.) You do not have to provide observations for
    every variable all the time! If a variable is unobserved/you don't have ground
    truth, you can leave that key out of the dictionary or include it with value None.
    The model will throw an error if you haven't provided enough observations to run
    the model at all (most common example: you have a DAG and you didn't provide an 
    observation for the root.)
    """
    """
    We will use function gt() to sample values for all  variables each iteration.
    In the real world, you probably use a DataLoader--data loaders can  be written 
    to return dictionaries with any number of values, not just inputs and outputs.
    """
    def gt(batch_size):
        A = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        B = A.copy()
        C = B.copy()
        return {graph['A']: torch.tensor(A), 
                graph['B']: torch.tensor(B), 
                graph['C']: torch.tensor(C)}
        
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    # cal_optimizer will only update parameters which should be calibrated
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
    
    # You can optionally pass in a SummaryWriter to log some info about the NGM. 
    # But you probably only want one thread doing that.
    summary_writer = SummaryWriter() if rank == 0 else None
    
    # training loop
    graph.train()
    for iter in range(TRAIN_ITER):
        sample = gt(BATCH_SIZE)
        sample = dict_to_gpu(sample, rank)
        """
        When training using default (non-EM) settings, 
        samples_in_pass would most commonly be set to 1 to maximize updates/time
        """
        loss = graph.loss(sample, samples_in_pass=1, summary_writer = summary_writer)
        """
        Use zero_grad(True) instead of zero_grad() so thatt updates 
        (momentum, optimizer statistics) are not done
        to parameters (such as combinatino weights or whole predictors) that were not 
        involved in the forward pass!!
        """
        optimizer.zero_grad(True)
        loss.backward()
        optimizer.step()
        
    """
    Standard practice should be to calibrate your model before you deploy or evaluate, 
    using held-out data. Ideally, it should even be different held-out data from any 
    you're using for e.g. validation performance checks throughout training. 
    You need to make these 2 calls before performing calibration--validation mode 
    is slightly different from either train or eval mode.
    """
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt(BATCH_SIZE)
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # Evaluation only needs to be performed by one GPU
    if rank != 0 :
        return
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample[graph['B']] = None
            """
            Use to_predict_marginal to predict the value/distribution of 
            any variable(s) of interest. Use more samples/passes to get 
            more accurate Monte Carlo estimates. In many cases you probably 
            don't need many though, especially if there's a right answer 
            and your model has probably learned it.
            """
            marginals = graph.predict_marginals(sample, 
                to_predict_marginal=['B'], 
                samples_per_pass=SAMPLES_PER_PASS, 
                num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted distribution of B: {marginals[graph['B']].probs}")

    dist.destroy_process_group()


if __name__ == '__main__':
    example_usage()
