"""
(Essentially) unit tests for random_variable and graphical_model.
"""

from .graphical_model import NeuralGraphicalModel
from .random_variable import *
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
    
    #original_gt = torch.tensor([[1,2],[3,4],[5,6]])
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
    print("First dim should be batch, last dim should be sample.")
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

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[], per_prediction_parents=[])
    
    dist_params = [torch.tensor([[0.0,0.0,0.0]])]
    gt = torch.tensor([1])
    
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    # should both be -natural log of 1/3
    print(f"loss: {loss}")
    print(f"nlogp: {nlogp}")

    
def categorical_loss_test_B():

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[], per_prediction_parents=[])
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = [torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])]
    gt = torch.tensor([[1,0]])
    
    # this test should result in a loss with one entry--both losses averaged together,
    # hence same result as test_A above
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")
    print(f"nlogp: {nlogp}")


def categorical_loss_test_C():
        
    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[], per_prediction_parents=[])
    
    # second dimension is 'category'--this should represent a batch of two instances
    # of one uniform distribution--and result in a loss tensor w/two entries
    dist_params = [torch.tensor([[[0.0],[0.0],[0.0]], [[1.0],[1.0],[1.0]]])]
    gt = torch.tensor([[1],[0]])
    
    # this test should result in a loss with two entries 
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, None, None)
    
    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")
    print(f"nlogp: {nlogp}")


def categorical_loss_test_D():
    # Test unsupervised loss with gumbel

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[], per_prediction_parents=[], gradient_estimator='reparameterize')
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = [torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])]
    gt = None
    sampled_value = torch.tensor([[1,0]])
    total_downstream_loss = torch.tensor(1.0)
    
    # loss and nlogp should be nothing, as we simply pass through gumbel!
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, sampled_value, total_downstream_loss)
    
    print(f"dist_params: {dist_params}")
    print(f"sampled value: {sampled_value} and downstream loss {total_downstream_loss}")
    print(f"loss: {loss}")    
    print(f"nlogp: {nlogp}")


def categorical_loss_test_E():
    # Test unsupervised loss with REINFORCE

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[], per_prediction_parents=[], gradient_estimator='REINFORCE')
    
    # second dimension is 'category'--this should represent two uniform distributions!
    dist_params = [torch.tensor([[[0.0,1.0],[0.0,1.0],[0.0,1.0]]])]
    gt = None
    sampled_value = torch.tensor([[1,0]])
    total_downstream_loss = torch.tensor(1.0)
    
    # loss should be 1x the log-prob--or average of 1.098 again--with REINFORCE!
    # nlogp should still be None
    print(variable.training)
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, sampled_value, total_downstream_loss)
    
    print(f"dist_params: {dist_params}")
    print(f"sampled value: {sampled_value} and downstream loss {total_downstream_loss}")
    print(f"loss: {loss}")    
    print(f"nlogp: {nlogp}")


def gaussian_loss_test_A():
    
    variable = GaussianVariable(name='test_var', predictor_fns=[], per_prediction_parents=[])
    
    # second dimension is 'mean, then stddev' and should always have size 2.
    # a sigma 'logit' of 0 corresponds to a standard deviation of 1.
    dist_params = [torch.tensor([[[1.0,2.0,3.0],[0.0,0.0,0.0]]])]
    
    gt = torch.tensor([[1.0,2.0,3.0]])  # every value was at the predicted mean 
    
    # loss = log_prob, which should be -log(1/sqrt(2*pi))
    print(variable.training)
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, None, None)

    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")    
    print(f"nlogp: {nlogp}")


def gaussian_loss_test_B():
    # Another test of gaussian loss. this one should result in a loss with 
    # three entries since this has a 'batch size' of 3. First entry should match loss of test A.
    # proceeding losses should be larger.
    
    variable = GaussianVariable(name='test_var', predictor_fns=[], per_prediction_parents=[])
    
    dist_params = [torch.tensor([[[1.0],[0.0]],[[1.0],[0.0]],[[1.0],[0.0]]])]
    
    gt = torch.tensor([[1.0],[2.0],[3.0]])  # first value was at the predicted mean 
    
    loss, nlogp = variable.loss_and_nlogp(dist_params, dist_params, gt, None, None)

    print(f"dist_params: {dist_params}")
    print(f"gt: {gt}")
    print(f"loss: {loss}")    
    print(f"nlogp: {nlogp}")
   

def categorical_estimate_distribution_test_A():
    # should combine all probabilities--more energy in first bin than in second or third.
    # should be about 1/3rd higher since softmax e^2/(e^2 + e^1 + e^1)

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[None], per_prediction_parents=[[]])
    dist_params = [torch.tensor([[[1.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0]]])]
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    
    variable.eval()
    result_dist = variable.estimate_distribution_from_dists(dist_params, weights)
    print(f"result_dist: {result_dist.probs}") 
    
    
def categorical_estimate_distribution_test_B():
    # Now we are combining results from two predictors...but they are the same as before with half confidence
    # (and the combination weights are at their default of 1.0)
    # , so we should get the same result as before
    
    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[None, None], per_prediction_parents=[[],[]])    
    dist_params = [torch.tensor([[[0.5,0.0,0.0,0.5],[0.0,0.0,0.5,0.0],[0.0,0.5,0.0,0.0]]]), torch.tensor([[[0.5,0.0,0.0,0.5],[0.0,0.0,0.5,0.0],[0.0,0.5,0.0,0.0]]])]
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    
    variable.eval()
    result_dist = variable.estimate_distribution_from_dists(dist_params, weights)
    print(f"result_dist: {result_dist.probs}") 


def categorical_estimate_distribution_test_C():
    # now combine samples instead of distributions
    # should get 0.5, 0.25, 0.25

    variable = CategoricalVariable(num_categories=3, name='test_var', predictor_fns=[None], per_prediction_parents=[[]])
    samples = torch.tensor([[[1.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    
    variable.eval()
    result_dist = variable.estimate_distribution_from_samples(samples, weights)
    print(f"result_dist: {result_dist.probs}") 
   
    
def gaussian_estimate_distribution_test_A():
    # first one--4 values with means averaging mean of 1.5
    # second dim--means averaging 1, and since they're the same sigma should be << 1

    variable = GaussianVariable(name='test_var', predictor_fns=[None], per_prediction_parents=[[]])
    
    samples = torch.tensor([[0.0,1.0,2.0,3.0],[1.0,1.0,1.0,1.0]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0]])
    result_dist = variable.estimate_distribution_from_samples(samples, weights)
    print(f"result_dist: mean {result_dist.mean}, variance {result_dist.variance}") 
    

def gaussian_estimate_distribution_test_B():
    # an empty spatial dimension and only one entry. Mean should be 2.0 because weight of last sample is 0!!

    variable = GaussianVariable(name='test_var', predictor_fns=[None], per_prediction_parents=[[]])
    
    samples = torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0]]])
    weights = torch.tensor([[1.0,1.0,1.0,1.0,1.0,0.0]])
    
    result_dist = variable.estimate_distribution_from_samples(samples, weights)
    print(f"result_dist: mean {result_dist.mean}, variance {result_dist.variance}") 
    

def gaussian_estimate_distribution_test_C():
    # This has a batch size of 2. The mean in both cases should be 2, but the second one...should 
    # have a higher std. dev because the components do as well? I think so. Gaussians can be strange.

    variable = GaussianVariable(name='test_var', predictor_fns=[None], per_prediction_parents=[[]])
    
    # 2 batch items, 6 samples each
    dist_params = [torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0],[-100.0,-100.0,-100.0,-100.0,-100.0,-100.0]],\
                                [[0.0,1.0,2.0,3.0,4.0,5.0],[2.0,2.0,2.0,2.0,2.0,2.0]]])]
    weights = torch.tensor([[1.0,1.0,1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0,1.0,0.0]])
    
    result_dist = variable.estimate_distribution_from_dists(dist_params, weights)
    print(f"result_dist: {result_dist.mean}, {result_dist.variance}") 
    print(f"{result_dist.component_distribution}, {result_dist.mixture_distribution}") 
    

def gaussian_estimate_distribution_test_D():
    # combine multiple predictors. THis is like test C but we use 2 predictors!
    # Like before, having 2 predictors with combination weights of 1 means we should become more confident, right?
    # Ah, I see. The first gaussian's std. dev. doesn't change because it's std. dev. starts out as basically zero...
    # And we combine the predictors before we combine samples.

    variable = GaussianVariable(name='test_var', predictor_fns=[None, None], per_prediction_parents=[[],[]])
    
    # 2 batch items, 6 samples each
    dist_params = [ torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0],[-100.0,-100.0,-100.0,-100.0,-100.0,-100.0]],\
                                [[0.0,1.0,2.0,3.0,4.0,5.0],[2.0,2.0,2.0,2.0,2.0,2.0]]]),\
                    torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0],[-100.0,-100.0,-100.0,-100.0,-100.0,-100.0]],\
                                [[0.0,1.0,2.0,3.0,4.0,5.0],[2.0,2.0,2.0,2.0,2.0,2.0]]])]
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
    TRAIN_ITER = 2000
    TEST_ITER = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    CAL_ITER = 1
    
    # set up model
    class SigmaLinear(torch.nn.Module):
    
        def __init__(self):
            super(SigmaLinear, self).__init__()
            # output is twice as large because we must predict mean and std. dev.
            self.linear = torch.nn.Linear(DIM, DIM*2, bias=True)
            
        def forward(self, x):
            x = self.linear(x)
            x = x.view(-1, 2, DIM)
            return x
    
    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[SigmaLinear()], per_prediction_parents=[[graph['A']]]))
    graph.addvar(GaussianVariable(name='C', predictor_fns=[SigmaLinear()], per_prediction_parents=[[graph['B']]]))
    
    # set up 'ground truth model'--this is so simple the model should match it exactly or something is wrong
    # especially given that the predictive structure will match the causality
    # the model should reduce uncertainty low, close to 0!
    def gt(batch_size):
        A = np.random.rand(batch_size,DIM).astype(np.float32)
        B = A*2
        C = np.fliplr(B).copy()  # copy fixes 'strides' of matrix. Weird torch thing. Torch doesn't want strided np stuff
        return {graph['A']: torch.tensor(A), graph['B']: torch.tensor(B), graph['C']: torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
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
        
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt(BATCH_SIZE)
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
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
    TRAIN_ITER = 1000
    TEST_ITER = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    CAL_ITER = 1
    
    
    # set up model    
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['A']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['B']]]))
    
    # set up 'ground truth model'--this is slightly more complex than the one in gaussian_model_test.
    # This time, our causality will be wrong--B will be a function of C, and they will be unaffected by A.
    # so the 'answers' we learn should basically be a uniform distribution for B and C--unless we can observe 
    # C, in which case we should get a definite answer for B. Given fully supervised training, anyway.
    """
    We could not get correct predictions of B were partially supervised, but in a testament to the power of directed 
    cycles: An extra predictor going from C to B would fix that.
    """
    def gt(batch_size):
        A = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        C = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        B = C.copy()
        return {graph['A']: torch.tensor(A), graph['B']: torch.tensor(B), graph['C']: torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
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
        
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt(BATCH_SIZE)
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample[graph['B']] = None
            marginals = graph.predict_marginals(sample, to_predict_marginal=['B'], samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B: {marginals[graph['B']].probs}")


def combination_weight_tree_test():
    
    
    """
        5 combination weights in the member multiply_predicted_var.combination_weights
        should be organized in a binary tree like this:
    
                            (3,0)
                           /     \
                          /       \
                         /         \
                      (2,0)       (2,1)
                       / \         / \
                      /   \       /   \
                     /     \     /     \
                  (1,0)  (1,1) (1,2)  (1,3)
                   / \    / \  /  \    / \
                  p1 p2  p3 p4 p5
                                     
         With each predictor's final weight a la get_combination_weights() being the product of the weight on each 
         branch from the root to the associated leaf. Works as of time of writing!
    """
    
    NCAT = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 1
    NPRED = 5
    TRAIN_ITER = 100
    
    class RandomLogits(torch.nn.Module):
        def forward(self):
            return torch.rand([BATCH_SIZE*SAMPLES_PER_PASS, NCAT], dtype=torch.float32)
    
    graph = NeuralGraphicalModel()
    multiply_predicted_var = CategoricalVariable(name='A', num_categories=4, predictor_fns=[RandomLogits(), RandomLogits(), RandomLogits(), RandomLogits(), RandomLogits()],\
            per_prediction_parents=[[]]*NPRED)
    graph.addvar(multiply_predicted_var)
    graph.train()
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.01)
    
    sample = {'A': torch.tensor([1])}
    for iter in range(TRAIN_ITER):
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # You are not supposed to access these members directly, but it's for a test. We want to see that they are multiplied correctly
    print(f"Trained combination weight vector: {[(key, multiply_predicted_var.combination_weights[key]) for key in multiply_predicted_var.combination_weights]}")
    # The calibration is zero, we don't need to examine _combination_weight_calibration
    print(f"Trained combination weights by predictor: {multiply_predicted_var.get_combination_weights([1]*NPRED)}")
    

def toy_experiment_template(graph, gt_func, to_predict_marginal: List[str], test_observed: List[str], drop_probs={}):
    """
    Template for a toy experiment with made-up variables. Will be used to run a number of tests.
    """
    
    TRAIN_ITER = 5000
    CAL_ITER = 0  # 500
    PASSES_PER_BATCH = 8
    weight_decay = 0.001
    lr = 0.01
    eps = 1e-2
    SAMPLES_PER_PASS = 2 if graph.SGD_rather_than_EM else 8
    BATCH_SIZE = 2
    TEST_ITER = 5
    
    # add l2 loss this time
    optimizer = torch.optim.Adam(graph.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    summary_writer = SummaryWriter()

    # DEBUG
    checkup_gt = gt_func(1)
    print(f'example output of gt_func: {checkup_gt}')
    
    # train
    graph.train()    
    for iter in range(TRAIN_ITER):
        sample = gt_func(BATCH_SIZE)
        for droppable in drop_probs.keys():
            if np.random.random() < drop_probs[droppable]:
                sample[droppable] = None
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS, summary_writer=summary_writer if iter % 100 == 1 else None, global_step=iter)
        if iter % 100 == 1:
            print(loss)
            summary_writer.add_scalar('trainloss', loss, iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Calibrating!')

    # validate
    force_predicted_input = [key for key in checkup_gt if key not in test_observed]
    graph.validate()
    graph.reset_calibration()
    for iter in range(CAL_ITER):
        sample = gt_func(BATCH_SIZE)
        #print(f'Ground Truth C: {sample["C"]}')
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS, force_predicted_input=force_predicted_input, summary_writer=summary_writer if iter % 100 == 1 else None, global_step=iter+TRAIN_ITER)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt_func(1)
            observations = {name: sample[name] for name in test_observed}

            marginals = graph.predict_marginals(observations, to_predict_marginal=to_predict_marginal, samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)

            for var_name in to_predict_marginal:
                if isinstance(graph[var_name], GaussianVariable):
                    print(f"ground truth {var_name}: {sample[var_name]}")
                    print(f"predicted {var_name}: {marginals[graph[var_name]].mean}, variance: {marginals[graph[var_name]].variance}")
                elif isinstance(graph[var_name], CategoricalVariable):
                    print(f"ground truth {var_name}: {sample[var_name]}")
                    print(f"predicted {var_name}: {marginals[graph[var_name]].probs}")                
                else:
                    raise ValueError("Unrecognized RandomVariable type in to_predict_marginal.")


def gt_simple_world(batch_size):
    GAUS_DIM=1
    B = np.random.normal(loc=0.0, scale=1.0, size=(batch_size,GAUS_DIM)).astype(np.float32)
    D = np.random.choice([-1.5, 0.181, 5.5], size=(batch_size,1), replace=True, p=[0.25,0.5,0.25]).astype(np.float32)
    
    A = np.squeeze(np.greater(D, 0)*1 + np.greater(D, 5)*1, axis=-1)
    E = A.copy()
    C = A.copy()*np.squeeze(np.greater(B,0),axis=-1) + ((A+1)%NCAT)*np.squeeze(np.less_equal(B,0),axis=-1)
    
    F = B + D 
    G = np.greater(np.squeeze(F,axis=-1), E-1)*1 + np.greater(np.squeeze(F,axis=-1), E)*1
    
    return {'A': torch.tensor(A), 'B': torch.tensor(B), 'C': torch.tensor(C), \
            'D': torch.tensor(D), 'E': torch.tensor(E), 'F': torch.tensor(F), 'G': torch.tensor(G)}


HIDDEN_DIM = 25
NCAT = 3
GAUS_DIM = 1


class CatLinear(torch.nn.Module):

    def __init__(self, parent_dims):
        super(CatLinear, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_dim, HIDDEN_DIM) for in_dim in parent_dims])
        self.final_linear = torch.nn.Linear(HIDDEN_DIM, NCAT)

    def forward(self, *parent_vals):
        hidden_features = []
        for i, val in enumerate(parent_vals):
            hidden_features.append(self.linears[i](val))
        x = torch.sum(torch.stack(hidden_features,dim=0),dim=0)
        x = torch.nn.functional.leaky_relu(x)
        x = self.final_linear(x)
        return x
        

class GausLinear(torch.nn.Module):
    
    def __init__(self, parent_dims):
        super(GausLinear, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_dim, HIDDEN_DIM) for in_dim in parent_dims])
        self.final_linear = torch.nn.Linear(HIDDEN_DIM, GAUS_DIM*2)
        
    def forward(self, *parent_vals):
        hidden_features = []
        for i, val in enumerate(parent_vals):
            hidden_features.append(self.linears[i](val))
        x = torch.sum(torch.stack(hidden_features,dim=0),dim=0)
        x = torch.nn.functional.leaky_relu(x)
        x = self.final_linear(x)
        x = x.view(-1, 2, GAUS_DIM)
        return x
        

class CatFeedForward(torch.nn.Module):

    def __init__(self, parent_dims):
        super(CatFeedForward, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(in_dim, HIDDEN_DIM) for in_dim in parent_dims])
        self.hid_linearA = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.hid_linearB = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.final_linear = torch.nn.Linear(HIDDEN_DIM, NCAT)

    def forward(self, *parent_vals):
        hidden_features = []
        for i, val in enumerate(parent_vals):
            hidden_features.append(self.linears[i](val))
        x = torch.sum(torch.stack(hidden_features,dim=0),dim=0)
        x = torch.nn.functional.leaky_relu(x)
        
        x = torch.nn.functional.leaky_relu(self.hid_linearA(x))
        x = torch.nn.functional.leaky_relu(self.hid_linearB(x))
        
        x = self.final_linear(x)
        return x


def toy_experiment_A():
    """
    A simple model strung together from various simple variables. Fully supervised. Should get everything right
    except C, which cannot be predicted from the variables it's connected to.
    Now, when D is 5.5, we still get this issue where the predicted variance of F is high. What's happening is that earlier in training, 
    F learns to output a high variance value, since when D is 5.5 (25% of the time) we get values of F that are far away from the other values.
    It eventually learns to get the mean relatively close--but learning to bring the variance back down after that appears to be quite a slow
    process.
    """
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT, NCAT])], per_prediction_parents=[[graph['A'],graph['C']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'])


def partially_supervised_gaus_test():
    """
    The same as toy experiment A, but with F supervised only a small amount of the time. Will it still be predicted with correect mean 
    and variance? It should.
    """
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT, NCAT])], per_prediction_parents=[[graph['A'],graph['C']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'], drop_probs={'F': 0.8})
        
        
def toy_experiment_correct_C_prediction():
    """
    If we have a B+C->E predictor--which is capable of representing the exact relationship between these three variables--then 
    we can predict C correctly (using consistency with downstream variables) even though C's parents don't have enough info
    to fully specify its value.
    """
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']], [graph['C'],graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'])
    

def toy_experiment_C_partially_supervised():
    """
    However, if C is partially supervised, the graph above breaks and doesn't work anymore! This is a known issue with partial
    supervision and SGD.
    """
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']], [graph['C'],graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'], drop_probs={'C': 0.5})
    
    
def toy_experiment_C_EM():
    """
    Expectation Maximization, while slower, is one way to fix the shortcoming above.
    """
    # set up model    
    graph = NeuralGraphicalModel(SGD_rather_than_EM=False)
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']], [graph['C'],graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'], drop_probs={'C': 0.5})


def toy_experiment_C_EM_supervised():
    # set up model    
    graph = NeuralGraphicalModel(SGD_rather_than_EM=False)
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']], [graph['C'],graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))
                                                    
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'], drop_probs={})    
    
    
def toy_experiment_C_fixed_with_loops():
    """
    toy_experiment_C_partially_supervised fixed by adding a directed cycle from E to C.
    """
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[], per_prediction_parents=[]))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[CatLinear([GAUS_DIM])], per_prediction_parents=[[graph['B']]]))
    graph.addvar(GaussianVariable(name='D', predictor_fns=[GausLinear([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A'], graph['B']]]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']], [graph['C'],graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fns=[GausLinear([GAUS_DIM, GAUS_DIM])], per_prediction_parents=[[graph['B'], graph['D']]]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM])],\
                                                    per_prediction_parents=[[graph['C'], graph['D'], graph['E'], graph['F']]]))                                       
    graph['C'].add_predictor(parents=[graph['E'], graph['B']], fn=CatLinear([NCAT, GAUS_DIM]))
                                       
    toy_experiment_template(graph, gt_simple_world, to_predict_marginal=['C','D','E','F','G'], test_observed=['A','B'], drop_probs={'C': 0.5})

    
def bayesian_reasoning_required_cat_gt(batch_size):
    A = np.random.choice([0, 1, 2], size=(batch_size,), replace=True, p=[0.25,0.5,0.25]).astype(np.int64)
    B = []
    for dig in A:
        if dig == 0:
            B.append(np.random.choice([0, 1, 2], size=(1), replace=True, p=[0.0,0.75,0.25]).astype(np.int64))    
        elif dig == 1:
            B.append(np.random.choice([0, 1, 2], size=(1), replace=True, p=[0.25,0.0,0.75]).astype(np.int64))
        elif dig == 2:
            B.append(np.random.choice([0, 1, 2], size=(1), replace=True, p=[0.75,0.25,0.0]).astype(np.int64))
    return {'A': torch.tensor(A), 'B': torch.tensor(np.squeeze(B, axis=1))}


def bayesian_reasoning_test():
    """
    We can't perfectly predict the values of a variable, but there is a 'correct' distribution given the upstream info we have. (Much easier than 
    trying to get the right answer given downstream info...)
    """
    NCAT = 3
    # set up model    
    graph = NeuralGraphicalModel()
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[CatLinear([NCAT])], per_prediction_parents=[[graph['A']]]))                                                    
    toy_experiment_template(graph, bayesian_reasoning_required_cat_gt, to_predict_marginal=['B'], test_observed=['A'], drop_probs={})


# TODO then test multiple predictions with missing observed inputs
# TODO this one also appears to be giving wrong answers due to the issues
def missing_observation_test():
    """
    As we add the capability for directed cycles, we also added the related ability to run the graph
    forward with observation-only variables unobserved--as long as there are still valid paths 
    through the graph, everything should be fine.
    """
    
    NCAT = 3
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='obvA', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='obvB', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[torch.nn.Linear(NCAT, NCAT), torch.nn.Linear(NCAT, NCAT)], \
                per_prediction_parents=[[graph['obvA']],[graph['obvB']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['B']]]))
    
    def missing_obv_gt(batch_size):
        obvA = np.random.choice([0, 1, 2], size=(batch_size,), replace=True, p=[0.3,0.4,0.3]).astype(np.int64)
        obvB = np.random.choice([0, 1, 2], size=(batch_size,), replace=True, p=[0.3,0.4,0.3]).astype(np.int64)
        B = []
        for digA, digB in zip(obvA, obvB):
            p = [0.3, 0.4, 0.3]
            for ind in range(NCAT):
                if digA == ind or digB == ind:
                    p[ind] = 0
            B.append(np.random.choice([0, 1, 2], size=(1), replace=True, p=p/np.sum(p)).astype(np.int64)) 
        B = np.squeeze(B, axis=1)
        C = B.copy()
        return {'obvA': torch.tensor(obvA), 'obvB': torch.tensor(obvB), 'B': torch.tensor(B), 'C': torch.tensor(C)}
        
    toy_experiment_template(graph, missing_obv_gt, to_predict_marginal=['B', 'C'], test_observed=['obvA'], drop_probs={'obvB': 0.5, 'B': 0.0})
    

def simple_cycle_test():
    """
    Just make sure the most basic kind of directed cycle functions
    """
        
    TRAIN_ITER = 2000
    SAMPLES_PER_PASS = 1
    PASSES_PER_BATCH = 1
    CAL_ITER = 1
    TEST_ITER = 5
    
    # set up model    
    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(GaussianVariable(name='B', predictor_fns=[GausLinear([1])], per_prediction_parents=[[graph['A']]]))
    graph['A'].add_predictor(parents=[graph['B']], fn=GausLinear([1]))
        
    def gt(train=True):
        sample = {}
        
        while len(sample) == 0:
            sample = {}
            val = np.random.normal()
            if not train or np.random.random() < 0.5:
                sample['A'] = torch.tensor([[val]])
            if not train or np.random.random() < 0.5:
                sample['B'] = torch.tensor([[val]])
        
        print(f'sample: {sample}')        
        return sample
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
    print(list(graph.parameters()))
    graph.train()
    
    for iter in range(TRAIN_ITER):
        sample = gt()
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt()
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # evaluate
    graph.eval()
    for iter in range(TEST_ITER):
        with torch.no_grad():
            sample = gt(False)
            marginals = graph.predict_marginals({'A': sample['A']}, to_predict_marginal=['B'], samples_per_pass=3, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B from A: {marginals[graph['B']].mean}, variance {marginals[graph['B']].variance}")
            marginals = graph.predict_marginals({'B': sample['B']}, to_predict_marginal=['A'], samples_per_pass=3, num_passes=PASSES_PER_BATCH)
            print(f"Predicted A from B: {marginals[graph['A']].mean}, variance {marginals[graph['A']].variance}")
        

def categorical_model_failure():
    """
    Showing how this categorical model fails to learn correctly if B and C are partially supervised, because its parent has insufficient information.
    """

    NCAT = 3  # all vars will have same number of categories
    TRAIN_ITER = 1000
    TEST_ITER = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    CAL_ITER = 1
    
    
    # set up model    
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['A']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['B']]]))
    
    # set up 'ground truth model'--this is slightly more complex than the one in gaussian_model_test.
    # This time, our causality will be wrong--B will be a function of C, and they will be unaffected by A.
    # so the 'answers' we learn should basically be a uniform distribution for B and C--unless we can observe 
    # C, in which case we should get a definite answer for B. Given fully supervised training, anyway.
    """
    We could not get correct predictions of B were partially supervised, but in a testament to the power of directed 
    cycles: An extra predictor going from C to B would fix that.
    """
    def gt(batch_size):
        A = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        C = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        B = C.copy()
        return {'A': torch.tensor(A), 'B': torch.tensor(B), 'C': torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
    print(list(graph.parameters()))
    graph.train()
    
    for iter in range(TRAIN_ITER):
        sample = gt(BATCH_SIZE)
        if np.random.random() < 0.5:
            sample['C'] = None
        if np.random.random() < 0.5 and sample['C'] is not None:
            sample['B'] = None
        print(f'sample: {sample}')
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt(BATCH_SIZE)
        sample[graph['B']] = None
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample['B'] = None
            marginals = graph.predict_marginals(sample, to_predict_marginal=['B'], samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B: {marginals[graph['B']].probs}")
            
            
def categorical_model_fixed_with_loops():
    """
    Just like categorical_model_failure, except with an extra connection from C to B. Without changing the supervision, this allows us to get
    accurate predictions!
    """

    NCAT = 3  # all vars will have same number of categories
    TRAIN_ITER = 5000
    TEST_ITER = 4
    BATCH_SIZE = 2
    SAMPLES_PER_PASS = 4
    PASSES_PER_BATCH = 2
    CAL_ITER = 1
    
    # set up model    
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['A']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['B']]]))
    graph['B'].add_predictor(parents=[graph['C']], fn=torch.nn.Linear(NCAT, NCAT))
    
    def gt(batch_size):
        A = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        C = np.random.randint(3,size=(BATCH_SIZE)).astype(np.int64)
        B = C.copy()
        return {'A': torch.tensor(A), 'B': torch.tensor(B), 'C': torch.tensor(C)}
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005)
    cal_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=0.005)
    print(list(graph.parameters()))
    graph.train()
    
    for iter in range(TRAIN_ITER):
        sample = gt(BATCH_SIZE)
        if np.random.random() < 0.4:
            sample['C'] = None
        if np.random.random() < 0.5 and sample['C'] is not None:
            sample['B'] = None
        print(f'sample: {sample}')
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    graph.validate()
    graph.reset_calibration()
        
    for iter in range(CAL_ITER):
        sample = gt(BATCH_SIZE)
        sample[graph['B']] = None
        loss = graph.loss(sample, samples_in_pass=SAMPLES_PER_PASS)
        if iter % 10 == 1:
            print(loss)
        cal_optimizer.zero_grad()
        loss.backward()
        cal_optimizer.step()
        
    # evaluate
    graph.eval()
    with torch.no_grad():
        for iter in range(TEST_ITER):
            sample = gt(1)
            sample['B'] = None
            marginals = graph.predict_marginals(sample, to_predict_marginal=['B'], samples_per_pass=SAMPLES_PER_PASS, num_passes=PASSES_PER_BATCH)
            print(f"ground truth: {sample}")
            print(f"Predicted B: {marginals[graph['B']].probs}")
            

def loop_de_loop_test():
    """
    Test a model with multiple complex and overlapping directed cycles.
    """
    NCAT = 3
    graph = NeuralGraphicalModel()
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='A', predictor_fns=[], per_prediction_parents=[]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='F', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['A']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='D', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['F']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='G', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['F']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='E', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['G']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='B', predictor_fns=[torch.nn.Linear(NCAT, NCAT)], per_prediction_parents=[[graph['G']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='H', predictor_fns=[torch.nn.Linear(NCAT, NCAT), torch.nn.Linear(NCAT, NCAT), torch.nn.Linear(NCAT, NCAT)], \
                    per_prediction_parents=[[graph['G']], [graph['E']], [graph['D']]]))
    graph.addvar(CategoricalVariable(num_categories=NCAT, name='C', predictor_fns=[torch.nn.Linear(NCAT, NCAT), torch.nn.Linear(NCAT, NCAT)], \
                    per_prediction_parents=[[graph['B']], [graph['H']]]))
    graph['A'].add_predictor(parents=[graph['C']], fn=torch.nn.Linear(NCAT, NCAT))
    graph['F'].add_predictor(parents=[graph['E']], fn=torch.nn.Linear(NCAT, NCAT))
    
    def loop_de_loop_gt(batch_size):
        val = np.random.randint(3,size=(batch_size)).astype(np.int64)
        return {'A': torch.tensor(val), 'B': torch.tensor(val), 'C': torch.tensor(val), 'D': torch.tensor(val), 'E': torch.tensor(val), \
                'F': torch.tensor(val), 'G': torch.tensor(val), 'H': torch.tensor(val) }
    
    toy_experiment_template(graph, loop_de_loop_gt, to_predict_marginal=['B', 'C', 'D', 'E', 'F', 'G', 'H'], test_observed=['A'], drop_probs={letter: 0.5 for letter in ['B','C','D','E','F','G','H']})
    

# TODO in the Tasknomy file, a variational autoencoding test--when using Cross-Task Consistency, can you get accurate, crisp dense predictions? Use Taskonomy data for that.
