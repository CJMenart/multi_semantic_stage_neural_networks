This repository implements the Neural Graphical Model: a form of neural network/belief network for combining data-driven learning and human-programmed knowledge.

The classes which implement the NGM are kept at the top level. The bulk of the logic is found in graphical_model.py and random_variable.py
If you want to get started writing your own NGM, example_usage.py is a decent, not-too-complicated example of how to use these classes.

This contains (or at one point in time contained :P) neural architecture code adapted from the repositories:
-Renê Ranftl and Alexey Bochkovskiy, "Vision Transformers for Dense Prediction", ArXiv / Renê Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", TPAMI 2020
-Ze Liu et al, "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows", ICCV 2021
-Nakano et al, "Cross-Task Consistency Learning Framework for Multi-Task Learning", CVPR 2020
-And of course a copy of ResNet from the PyTorch model zoo.

To replicate the results from the first paper, use ade_experiments.py, and run experiments over five trials (using the random seeds 1-5 for the partial supervision experiments).
The first-experiment code has been updated to use the new core interface and hasn't been tested since then, so full warning, it's not the *exact* same code.

