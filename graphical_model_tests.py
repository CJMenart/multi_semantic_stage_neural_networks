"""
(Essentially) unit tests for random_variable and graphical_model.
"""

from graphical_model import NeuralGraphicalModel
from random_variable import *
import torch
import numpy as np


def predict_marginal_logic_test():
    """
    This test copies some of the logic inside NeuralGraphicalModel.predict_marginals(), in order to test its correctness.
    It copies the logic, rather than trying to call the function directly, not only so that we can access the intermediate
    results we are interested in, but so that we can make a special tensor packed with ascending integers, and skip over the
    part where the model transforms inputs to outputs, letting us more verify that what's happening is correct.
    The logic in question is complicated because we pack both 'multiple data items' and 'multiple samples of each'
    along the batch dimension of input tensors, and then have to unpack them--and the corresponding losses--after.
    The whole model will be wrong if loss entries to not correspond to the correct distribution parameters!
    """
    
    original_gt = torch.tensor([[1,2],[3,4],[5,6]])
    original_gt = torch.tensor([[[[1,2],[3,4]]],[[[5,6],[7,8]]],[[[9,10],[11,12]]]])
    print(f"original_gt.shape: {original_gt.shape}")
    
    samples_per_pass = 4
    tiled_gt = torch.repeat_interleave(original_gt, samples_per_pass, dim=0)
    print(f"tiled_gt: {tiled_gt}")
        
    result = tiled_gt  # we just pretend its an identity transform for test purposes
    # each entry of loss goes with data-item-A, sample-B, hence the numbers
    loss = torch.tensor([11,12,13,14,21,22,23,24,31,32,33,34])
    print(f"loss: {loss}")
    
    transformed_loss = -loss.view(-1, samples_per_pass)
    print("Transformed loss (this should correspond with transformed result:")
    print(f"{transformed_loss}")
        
    result = torch.split(result,samples_per_pass,dim=0)
    result = torch.stack(result)
    result = result.permute(0, *list(range(2, result.dim())), 1)
    print(f"transformed result: {result}")
    
    # in 'real computation' there may also be multiple passes which are stacked. Let's see if that's correct:
    stacked_losses = torch.cat([transformed_loss, transformed_loss, transformed_loss], dim=-1)
    print(f"Stacked losses: {stacked_losses}")
    samples_or_dists = torch.cat([result, result, result], dim=-1)
    print(f"stacked result: {samples_or_dists}")


def categorical_loss_test_A():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    dist_params = torch.tensor([[0.0,0.0,0.0]])
    gt = torch.tensor([1])
    
    loss = variable.loss(dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")
    
    
def categorical_loss_test_B():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])
    gt = torch.tensor([[1,0]])
    
    loss = variable.loss(dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")
    
    
def categorical_loss_test_B():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])
    gt = torch.tensor([[1,0]])
    
    # this test should result in a loss with two entries 
    loss = variable.loss(dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")
    

def categorical_loss_test_B2():
        
    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    # second dimension is 'category'--this should represent a batch of two instances
    # of one uniform distribution--and result in a loss tensor w/two entries
    dist_params = torch.tensor([[[0.0],[0.0],[0.0]], [[1.0],[1.0],[1.0]]])
    gt = torch.tensor([[1],[0]])
    
    # this test should result in a loss with two entries 
    loss = variable.loss(dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")


def categorical_loss_test_C():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])
    gt = None
    sampled_value = torch.tensor([[1,0]])
    total_downstream_loss = 1.0
    
    loss = variable.loss(dist_params, gt, sampled_value, total_downstream_loss)
    
    print(f"dist_params: {dist_params}")
    print(f"sampled value: {sampled_value} and downstream loss {total_downstream_loss}")
    print(f"loss: {loss}")    


def gaussian_loss_test_A():
    
    variable = GaussianVariable(name='test_var', predictor_fn=None)
    
    # second dimension is 'mean, then stddev' and should always have size 2.
    # a sigma 'logit' of 0 corresponds to a standard deviation of 1.
    dist_params = torch.tensor([[[1.0,2.0,3.0],[0.0,0.0,0.0]]])
    
    gt = torch.tensor([[1.0,2.0,3.0]])  # every value was at the predicted mean 
    
    loss = variable.loss(dist_params, gt, None, None)

    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")    


def gaussian_loss_test_A2():
    
    variable = GaussianVariable(name='test_var', predictor_fn=None)
    
    dist_params = torch.tensor([[[1.0],[0.0]],[[1.0],[0.0]],[[1.0],[0.0]]])
    
    gt = torch.tensor([[1.0],[2.0],[3.0]])  # every value was at the predicted mean 
    
    loss = variable.loss(dist_params, gt, None, None)

    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")    
    

def categorical_estimate_distribution_test_A():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    dist_params = torch.tensor([[[1.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    result_dist = variable.estimate_distribution_from_dists(dist_params, weights)
    print(f"result_dist: {result_dist.probs}") 
    
    
def categorical_estimate_distribution_test_B():
    
    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fn=None)
    
    dist_params = torch.tensor([[[1.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    result_dist = variable.estimate_distribution_from_samples(dist_params, weights)
    print(f"result_dist: {result_dist.probs}") 
    
    
def gaussian_estimate_distribution_test_A():

    variable = GaussianVariable(name='test_var', predictor_fn=None)
    
    dist_params = torch.tensor([[0.0,1.0,2.0,3.0],[1.0,1.0,1.0,1.0]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    result_dist = variable.estimate_distribution_from_samples(dist_params, weights)
    print(f"result_dist: {result_dist.mean}, {result_dist.variance}") 
    

def gaussian_estimate_distribution_test_B():

    variable = GaussianVariable(name='test_var', predictor_fn=None)
    
    samples = torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0,1.0,0.0]])
    
    result_dist = variable.estimate_distribution_from_samples(samples, weights)
    print(f"result_dist: {result_dist.mean}, {result_dist.variance}") 
    

def gaussian_estimate_distribution_test_C():

    variable = GaussianVariable(name='test_var', predictor_fn=None)
    
    # 2 batch items, 6 samples each
    dist_params = torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0],[-100.0,-100.0,-100.0,-100.0,-100.0,-100.0]],\
                                [[0.0,1.0,2.0,3.0,4.0,5.0],[2.0,2.0,2.0,2.0,2.0,2.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0,1.0,0.0]])
    
    result_dist = variable.estimate_distribution_from_dists(dist_params, weights)
    print(f"result_dist: {result_dist.mean}, {result_dist.variance}") 
    print(f"{result_dist.component_distribution}, {result_dist.mixture_distribution}") 
    

def gaussian_model_test():
    """
    Test an extremely simple model with a chain of three Gaussian variables to see if distribution prediction based 
    on both upstream and downstream variables seems to work.
    """

    DIM = 3  # all vars will have same dim for simplicity
    TRAIN_ITER = 5000
    TEST_ITER = 5
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    
    # set up model
    class SigmaLinear(torch.nn.Module):
    
        def __init__(self):
            super(SigmaLinear, self).__init__()
            self.linear = torch.nn.Linear(DIM, DIM*2, bias=True)
            
        def forward(self, x):
            x = self.linear(x)
            x = x.view(-1, 2, DIM)
            return x
    
    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='A', predictor_fn=None))
    graph.addvar(GaussianVariable(name='B', predictor_fn=SigmaLinear(), parents=[graph['A']]))
    graph.addvar(GaussianVariable(name='C', predictor_fn=SigmaLinear(), parents=[graph['B']]))
    
    # set up 'ground truth model'--this is so simple the model should match it exactly or something is wrong
    # especially given that the predictive structure will match the causality
    # the model should reduce uncertainty low, close to 0!
    def gt(batch_size):
        A = np.random.rand(batch_size,DIM).astype(np.float32)
        B = A*2
        C = np.fliplr(B).copy()  # copy fixes 'strides' of matrix. Weird torch thing
        return {graph['A']: torch.tensor(A), graph['B']: torch.tensor(B), graph['C']: torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    print(list(graph.parameters()))
    graph.train()
    
    for iter in range(TRAIN_ITER):
        sample = gt(BATCH_SIZE)
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample[graph['B']] = None
            sample[graph['C']] = None
            marginals = graph.predict_marginals(sample, to_predict_marginal=[graph['B'], graph['C']], samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B: {marginals[graph['B']].mean}, variance: {marginals[graph['B']].variance}")
            print(f"Predicted C: {marginals[graph['C']].mean}, variance: {marginals[graph['C']].variance}")
            

def categorical_model_test():
    """
    Test an extremely simple model with a chain of three Categorical variables to see if distribution prediction based 
    on both upstream and downstream variables seems to work.
    This is the limit of the unit tests. For more complicated tests/examples, you have to look at toy_experiments
    """

    NCAT = 3  # all vars will have same number of categories
    TRAIN_ITER = 500
    TEST_ITER = 5
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    
    # set up model    
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(NCAT, name='A', predictor_fn=None))
    graph.addvar(CategoricalVariable(NCAT, name='B', predictor_fn=torch.nn.Linear(NCAT, NCAT), parents=[graph['A']]))
    graph.addvar(CategoricalVariable(NCAT, name='C', predictor_fn=torch.nn.Linear(NCAT, NCAT), parents=[graph['B']]))
    
    # set up 'ground truth model'--this is slightly more complex than the one in gaussian_model_test.
    # This time, our causality will be wrong--B will be a function of C, and they will be unaffected by A.
    # so the 'answers' we learn should basically be a uniform distribution for B and C--unless we can observe 
    # C, in which case we should get a definite answer for B.
    def gt(batch_size):
        A = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        C = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        B = C.copy()
        return {graph['A']: torch.tensor(A), graph['B']: torch.tensor(B), graph['C']: torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    print(list(graph.parameters()))
    graph.train()
    
    for iter in range(TRAIN_ITER):
        sample = gt(BATCH_SIZE)
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample[graph['B']] = None
            marginals = graph.predict_marginals(sample, to_predict_marginal=['B'], samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B: {marginals[graph['B']].probs}")
