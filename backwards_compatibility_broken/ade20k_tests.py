"""
Unit/bug tests for functionality that was added to support ADE20K experiments.
"""
from graphical_model import NeuralGraphicalModel
from random_variable import CategoricalVariable, GaussianVariable, MultiplyPredictedCategoricalVariable
from advanced_random_variables import CategoricalVariableWSmoothing
import torch
import numpy as np
from toy_experiments import CatLinear, CatFeedForward, GausLinear, toy_experiment, gt_fixed, NCAT, GAUS_DIM
import pickle
from pathlib import Path
from ade20k_exp.ade20k_hierarchy import ADEWordnetHierarchy
from torch.nn.parameter import Parameter

# test soft EM
# This could be done by repeating a 'toy experiment'
def em_test_A():
    
    # set up model    
    graph = NeuralGraphicalModel(SGD_rather_than_EM=False)
    # two observation variables, one continuous, one discrete
    graph.addvar(CategoricalVariable(NCAT, name='A', predictor_fn=None))
    graph.addvar(GaussianVariable(name='B', predictor_fn=None))
    # 'second layer' variables
    graph.addvar(CategoricalVariable(NCAT, name='C', predictor_fn=CatFeedForward([GAUS_DIM]), parents=[graph['B']]))
    graph.addvar(GaussianVariable(name='D', predictor_fn=GausLinear([NCAT, GAUS_DIM]), parents=[graph['A'], graph['B']]))
    # 'third layer', at which point inputs can begin coming from different past layers 
    graph.addvar(MultiplyPredictedCategoricalVariable(NCAT, name='E', predictor_fns=[CatLinear([NCAT]), CatFeedForward([NCAT, GAUS_DIM])], per_prediction_parents=[[graph['A']],[graph['C'], graph['B']]]))
    graph.addvar(GaussianVariable(name='F', predictor_fn=GausLinear([GAUS_DIM, GAUS_DIM]), parents=[graph['B'], graph['D']]))
    # a 'final output'--potentially (but not necessarily), this could be the variable we are ultimately interested in
    graph.addvar(CategoricalVariable(NCAT, name='G', predictor_fn=CatLinear([NCAT, GAUS_DIM, NCAT, GAUS_DIM]),\
                                                    parents=[graph['C'], graph['D'], graph['E'], graph['F']]))

    toy_experiment(graph, 100000, drop_probs={'C': 0.25}, PASSES_PER_BATCH=2, weight_decay=0.0001, gt_func=gt_fixed, lr=0.0005, eps=0.1)                                                    
    
    
def em_test_B(SGD_rather_than_EM=False):

    # simpler experiment with no real-valued variables 
    BATCH_SIZE = 2
    SAMPLES_IN_PASS = 8
    def gt(batch_size):
        A = np.random.random_integers(0, NCAT-1, size=(batch_size,))
        B = A.copy()
        C = A.copy()
        
        return {'A': torch.tensor(A), 'B': torch.tensor(B), 'C': torch.tensor(C)}
        
    class Guess(torch.nn.Module):

        def __init__(self):
            super(Guess, self).__init__()
            self.params = Parameter(torch.ones((BATCH_SIZE*SAMPLES_IN_PASS, NCAT), dtype=torch.float32, requires_grad=True))
                
        def forward(self):
            return self.params
    
    # set up model--we only observe A
    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(CategoricalVariable(NCAT, name='A', predictor_fn=None))
    graph.addvar(CategoricalVariable(NCAT, name='B', predictor_fn=Guess()))
    graph.addvar(MultiplyPredictedCategoricalVariable(NCAT, name='C', predictor_fns=[CatLinear([NCAT]) for i in range(2)], per_prediction_parents=[[graph['A']], [graph['B']]]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.001, eps=1.0)
    #optimizer = torch.optim.SGD(graph.parameters(), lr=0.0005)
    
    # train, seeing variable B (what we'll try to predict) only half the time.
    graph.train()    
    for iter in range(500000):
        sample = gt(BATCH_SIZE)
        if iter % 2 == 1:
            observations = {'A': sample['A'], 'C': sample['C']}
        else:
            observations = sample
        loss = graph.loss(observations, samples_in_pass=SAMPLES_IN_PASS)
        if iter % 100 in [1, 2]:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(5):
            sample = gt(BATCH_SIZE)
            observations = {'A': sample['A'], 'C': sample['C']}

            marginals = graph.predict_marginals(observations, to_predict_marginal=['B'], samples_per_pass=SAMPLES_IN_PASS, num_passes=1)
                
            print(f"ground truth B: {sample['B']}")
            print(f"predicted B: {marginals[graph['B']].probs}")
    print(f"C prediction weights: {[par for par in graph['C'].parameters()]}")
    
    
def em_test_C(SGD_rather_than_EM=False):
    """
    The same as em_test_B, except that the variable of interest (B) can in fact be predicted given 
    its parents. This trivially causes the model--with either EM or SGD--to converge to the correct 
    solution.
    """

    BATCH_SIZE = 4
    SAMPLES_IN_PASS = 8
    def gt(batch_size):
        A = np.random.random_integers(0, NCAT-1, size=(batch_size,))
        B = A.copy()
        C = A.copy()
        preB = A.copy()
        
        return {'A': torch.tensor(A), 'B': torch.tensor(B), 'C': torch.tensor(C), 'preB': torch.tensor(preB)}
            
    # set up model--we only observe A
    graph = NeuralGraphicalModel(SGD_rather_than_EM=SGD_rather_than_EM)
    graph.addvar(CategoricalVariable(NCAT, name='A', predictor_fn=None))
    graph.addvar(CategoricalVariable(NCAT, name='preB', predictor_fn=None))
    graph.addvar(CategoricalVariable(NCAT, name='B', predictor_fn=CatLinear([NCAT]), parents=[graph['preB']]))
    graph.addvar(MultiplyPredictedCategoricalVariable(NCAT, name='C', predictor_fns=[CatLinear([NCAT]) for i in range(2)], per_prediction_parents=[[graph['A']], [graph['B']]]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.001, eps=1.0)
    
    # train, seeing variable B (what we'll try to predict) only half the time.
    graph.train()    
    for iter in range(2500):
        sample = gt(BATCH_SIZE)
        if iter % 10 == 1:
            observations = {'A': sample['A'], 'C': sample['C'], 'preB': sample['preB']}
        else:
            observations = sample
        loss = graph.loss(observations, samples_in_pass=SAMPLES_IN_PASS)
        if iter % 100 == 1:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(10):
            sample = gt(1)
            observations = {'A': sample['A'], 'preB': sample['preB']}

            marginals = graph.predict_marginals(observations, to_predict_marginal=['B'], samples_per_pass=SAMPLES_IN_PASS, num_passes=1)
                
            print(f"ground truth B: {sample['B']}")
            print(f"predicted B: {marginals[graph['B']].probs}")
        
        
# test variable w/cycle nonsense
def smoothing_cycle_test_A():

    graph = NeuralGraphicalModel()
    NCLASS = 2
    pred = torch.nn.Conv2d(2, NCLASS, kernel_size=1, bias=True)
        
    graph.addvar(CategoricalVariable(2, name='Input', predictor_fn=None))
    graph.addvar(CategoricalVariableWSmoothing(NCLASS, max_spatial_dimensions=2, name='Output', predictor_fn=pred, parents=[graph['Input']]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005, weight_decay=0.0)
    
    for iter in range(1000):
        input = torch.tensor([[[1,1],[0,0]]])
        output = torch.tensor([[[1,1],[0,0]]])
        sample = {'Input': input, 'Output': output}
        
        loss = graph.loss(sample, samples_in_pass=1, summary_writer=None, global_step=iter)
        
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("learned smoothing weights:")
    print(graph['Output'].smoothing_weights)  # should be 50-50
    
    
# test smoothing more deeply--test the sampling part as well!
def smoothing_cycle_test_B():
    
    graph = NeuralGraphicalModel()
    NCLASS = 2
    pred = torch.nn.Conv2d(3, NCLASS, kernel_size=1, bias=True)
        
    graph.addvar(CategoricalVariable(3, name='Input', predictor_fn=None))
    graph.addvar(CategoricalVariableWSmoothing(NCLASS, max_spatial_dimensions=2, name='Output', predictor_fn=pred, parents=[graph['Input']]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005, weight_decay=0.0)
    
    for iter in range(1000):
        input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
        output = torch.tensor([[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]]])
        sample = {'Input': input, 'Output': output}
        
        loss = graph.loss(sample, samples_in_pass=1, summary_writer=None, global_step=iter)
        
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("learned smoothing weights:")
    print(graph['Output'].smoothing_weights)  # should be tilted towards same-class adjacency
    
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(10):
            input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
            observations = {'Input': input}

            marginals = graph.predict_marginals(observations, to_predict_marginal=['Output'], samples_per_pass=32, num_passes=1)

            # we expect 3 rows to be one class and 3 rows the other in each sample.
            print(f"ground truth input: {observations[graph['Input']]}")
            print(f"predicted output: {marginals[graph['Output']]}")


# test having the smoothed variable upstream of another variable
def smoothing_cycle_test_C():
    
    graph = NeuralGraphicalModel()
    NCLASS = 2
    pred = torch.nn.Conv2d(3, NCLASS, kernel_size=1, bias=True)
    other_pred = torch.nn.Conv2d(NCLASS, NCLASS, kernel_size=1, bias=True)
        
    graph.addvar(CategoricalVariable(3, name='Input', predictor_fn=None))
    graph.addvar(CategoricalVariableWSmoothing(NCLASS, max_spatial_dimensions=2, name='Output', predictor_fn=pred, parents=[graph['Input']]))
    graph.addvar(CategoricalVariable(NCLASS, name='Downstream', predictor_fn = other_pred, parents=[graph['Output']]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005, weight_decay=0.0)
    
    for iter in range(1000):
        input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
        output = torch.tensor([[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]]])
        downstream = torch.tensor([[[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
        sample = {'Input': input, 'Output': output, 'Downstream': downstream}
        
        loss = graph.loss(sample, samples_in_pass=1, summary_writer=None, global_step=iter)
        
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("learned smoothing weights:")
    print(graph['Output'].smoothing_weights)  # should be tilted towards same-class adjacency
    
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(10):
            input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
            downstream = torch.tensor([[[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
            observations = {'Input': input, 'Downstream': downstream}

            marginals = graph.predict_marginals(observations, to_predict_marginal=['Output'], samples_per_pass=32, num_passes=1)

            # we expect 3 rows to be one class and 3 rows the other in each sample.
            print(f"ground truth input: {observations[graph['Input']]}")
            print(f"predicted output: {marginals[graph['Output']]}")
            

# test having the smoothed variable upstream of another variable
# But this time partially supervised!!
def smoothing_cycle_test_D():

    torch.autograd.set_detect_anomaly(True)
    graph = NeuralGraphicalModel()
    NCLASS = 2
    pred = torch.nn.Conv2d(3, NCLASS, kernel_size=1, bias=True)
    other_pred = torch.nn.Conv2d(NCLASS, NCLASS, kernel_size=1, bias=True)
        
    graph.addvar(CategoricalVariable(3, name='Input', predictor_fn=None))
    graph.addvar(CategoricalVariableWSmoothing(NCLASS, max_spatial_dimensions=2, name='Output', predictor_fn=pred, parents=[graph['Input']]))
    graph.addvar(CategoricalVariable(NCLASS, name='Downstream', predictor_fn = other_pred, parents=[graph['Output']]))
    
    optimizer = torch.optim.Adam(graph.parameters(), lr=0.005, weight_decay=0.0)
    
    for iter in range(1000):
        input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
        output = torch.tensor([[[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]]])
        downstream = torch.tensor([[[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
        sample = {'Input': input, 'Output': output, 'Downstream': downstream}
        
        if iter % 2 == 0:
            sample['Output'] = None
        
        loss = graph.loss(sample, samples_in_pass=1, summary_writer=None, global_step=iter)
        
        print(f"loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("learned smoothing weights:")
    print(graph['Output'].smoothing_weights)  # should be tilted towards same-class adjacency
    
    # evaluate non-observations from observations
    graph.eval()
    with torch.no_grad():
        for iter in range(10):
            input = torch.tensor([[[1,1,1,1],[2,2,2,2],[2,2,2,2],[0,0,0,0]]])
            downstream = torch.tensor([[[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
            observations = {'Input': input, 'Downstream': downstream}

            marginals = graph.predict_marginals(observations, to_predict_marginal=['Output'], samples_per_pass=32, num_passes=1)

            # we expect 3 rows to be one class and 3 rows the other in each sample.
            print(f"ground truth input: {observations[graph['Input']]}")
            print(f"predicted output: {marginals[graph['Output']]}")


def index_mapping_test(datapath: Path, csv_file: Path, index_file: Path):
    hierarchy = ADEWordnetHierarchy(csv_file, datapath, index_file)
    with open(index_file, "rb") as f:
        index = pickle.load(f)
        objectnames = index['objectnames']
    for synset_idx in hierarchy.valid_indices()[1]:
        print(f"ADE children of {hierarchy.ind2synset[synset_idx]}:")
        print([objectnames[hierarchy.valid_indices()[0][child_idx]] for child_idx in hierarchy.valid_ade_children_mapped(synset_idx)])