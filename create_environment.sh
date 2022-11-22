conda create -n pytorch18 python=3.8 pytorch=1.8.0 torchvision cudatoolkit=10.2 numpy scikit-learn tensorboard seaborn -c pytorch
conda activate pytorch18
conda install timm dtreeviz lime -c conda-forge
