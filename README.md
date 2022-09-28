This repository implements the Neural Graphical Model: a form of neural network/belief network for combining data-driven learning and human-programmed knowledge.

The classes which implement the NGM are kept at the top level. The bulk of the logic is found in graphical_model.py and random_variable.py
If you want to get started writing your own NGM, the tests near the bottom of graphical_model_tests.py are some simple examples of the class in action, and shapes/shape_experiments.py is a not-too-complicated example of a full-fledge experiment.

To replicate the results from the paper, use ade_experiments.py, and run experiments over five trials (using the random seeds 1-5 for the partial supervision experiments).

We use MIT's ADE20K dataset, which can be obtained here: <https://groups.csail.mit.edu/vision/datasets/ADE20K/>

We use the partitions of this data created by "A Framework for Explainable Deep Neural Models Using External Knowledge Graphs". We have included three csv's under ade20k_exp/ which show which images are in the train, validation, and test partitions if you want to run using the exact same splits.

As noted in ade20k_exp/, the code expects these files to be laid out as follows:
ade20k_common_scenes_v1.00
|
+-- Images
    |
    +--test
        |
        +-- Contains 10% of data
    +--train
        |  
        +-- Contains 85% of data
    +--val
        |
        +-- Contains 5% of data
    Seg
    |
    +--test
        |
        +-- Contains 10% of data
    +--train
        |  
        +-- Contains 85% of data
    +--val
        |
        +-- Contains 5% of data
        
This class also takes advantage of index_ade20k.pkl (which can be downlaoded from MIT's
ADE20K website) and ade20k_wordnet_object_hierarchy.csv (which is again courtsey of "External Knowledge Graphs").