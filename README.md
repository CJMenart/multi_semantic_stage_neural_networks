### Welcome

This repository implements the Multi-Semantic-Stage Neural Network: a form of deep neural network/belief network for combining data-driven learning flexibly with human-programmed structure and knowledge. It also enables the construction of models which flexibly exploit heterogenous and incomplete sources of of data/labels. For more information on the MSSNN, see "Performant Learning and Qualified
Confidence with Multi-Semantic-Stage Neural Networks".

The classes which implement the MSSNN are kept at the top level. The bulk of the logic is found in graphical_model.py and random_variable.py.
If you want to get started writing your own MSSNN, example_usage.py is a decent, not-too-complicated example of how to use these classes.

This contains (or at one point in time contained) neural architecture code adapted from, reverse-engineered from, or inspired by these works:
- Renê Ranftl and Alexey Bochkovskiy, "Vision Transformers for Dense Prediction", ArXiv / Renê Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2020
- Ze Liu et al, "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows", ICCV 2021
- Nakano et al, "Cross-Task Consistency Learning Framework for Multi-Task Learning", CVPR 2020
- Tian et al, "FCOS: Fully-Convolutional One-Stage Object Detection", ICCV 2019
- And of course a copy of ResNet from the PyTorch model zoo.


### Installation

Theoretically, create_environment.sh shows how you can create an Anaconda environment with the same dependencies used by the author of this code. But package management is hard. Send an email or issue if you
 have trouble with this. Remember that in order for conda to install the GPU version of PyTorch, you should be running it on a machine which can see a GPU.


### Sub-Directories

Different subfolders primarily contain differnet experiments from the associated dissertation:
- The "preliminary shape experiment" which was used to demonstrate initial feasibility lives under backwards_compatibility_broken/shapes/ . As the directory structure suggests, it has not been updated to work with the latest version of the core engine many compatibility-breaking changes ago. Send an Issue/email if you need help with this
- All experiments on the MIT ADE20K dataset live under "ade20k_exp/". This includes the initial performance experiment, under "ade20k_experiments", and the explainability experiments.
	- To replicate the results from the first paper, use ade_experiments.py, and run experiments over five trials (using the random seeds 1-5 for the partial supervision experiments).
- Self-Driving Object Detection experiments are located under detection_exp/

The "Anchors" folder contains a pulled copy of the publicly-released code for the Anchors XAI method (). It is called by some of the explainability experiments.

NOTE: This branch was published in advance of the dissertation. To run different experiments with the corresponding version of the code, you may simply want to check out the appropriate branch from this repository.

### Other Notes

- (Most of) This repository was written before the title of this work was changed to "Multi-Semantic-Stage Neural Networks" from "Neural Graphical Models". Most of the code, and many of the comments, will still say "NGM", but they mean MSSNN.

- This code goes beyond the work seen in "Performant Learning and Qualified Confidence". It contains fully-general handling for models containing directed cycles of variables (which is why NeuralGraphicalModel() is such a long class). If you create directed cycles, the DFS will decide which predictors to run in each forward pass based the pattern of observation and pseudo-random heuristics. You might want to turn on Debug outputs for NeuralGraphicalModel()/look at Tensorboard logs on occassion so you are not surprised by your model's behavior.

- Distributed Data Parallel training is supported, but only using the custom "DDPNGM" in this repository. Other parallel training methods are not supported, and as far as I can tell not easily achievable.
