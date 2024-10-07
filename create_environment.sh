conda deactivate
conda create -n pytorch13 pytorch=1.13 torchvision=0.14 pytorch-cuda=11.6 numpy scikit-learn tensorboard seaborn -c pytorch -c nvidia
conda activate pytorch13
conda update numpy=1.23
conda install timm dtreeviz==1.4.0 lime pycocotools -c conda-forge