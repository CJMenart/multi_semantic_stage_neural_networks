"""
Experiments with a moderately-complex NGM applied to toy data consisting of colored geometric shapes.
"""
from random_variable import CategoricalVariable, GaussianVariable, MultiplyPredictedCategoricalVariable, ProbSpaceCategoricalVariable
from graphical_model import NeuralGraphicalModel
import torch
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from my_resnet import ResNet, BasicBlock
from torch.utils.data.dataset import Dataset
from shapes.shape_enums import *
import torch.nn as nn
import os
from torchvision.io import read_image
from torchvision import transforms
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parameter import Parameter
from torch.utils.data.sampler import Sampler
import random

NUM_GPU = 1
NUM_WORKER = 4*NUM_GPU
IMAGE_INPUT_SZ = 256
SHAPE_SEG_SZ = 32
TESTSET_SZ = 1024
WEIGHT_DECAY = 0.005
SAMPLES_PER_PASS = 1
ELLIPSE_OR_POL_DIM = 3
IS_PARALLELOGRAM_DIM = 2
UNSUPERVISED_DOWNWEIGHT_SCHEDULE = 2**13
DEF_BATCH_SIZE = 64
DEF_TRAIN_ITER = 2**16
DEF_CALIBRATE_ITER = 2**6
DEF_EPOCH_PER_CHECK = 8
DEF_PATIENCE_EPOCHS = 128
DEF_LR = 0.0005
DEF_WEIGHT_DECAY = 0.0005

class ShapeImageDataset(Dataset):
    def __init__(self, img_dir: Path, img_size: int, seg_size: int, num_imgs: int, is_partially_supervised=False, drop_type='orig'):
        super(ShapeImageDataset, self).__init__()
        if num_imgs % NUM_IM_CLASSES != 0:
            raise ValueError("Number of images must be multiple of number of image classes.")
        self.num_imgs = num_imgs
        self.img_dir = img_dir
        self.imresize = torchvision.transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST)
        self.segresize = torchvision.transforms.Resize(seg_size, interpolation=InterpolationMode.NEAREST)
        self.is_partially_supervised = is_partially_supervised
        self.drop_type = drop_type

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{idx}.png"
        seg_path = self.img_dir / f"{idx}_seg.png"
        image = read_image(str(img_path))
        shapeseg = read_image(str(seg_path))
        label = idx % NUM_IM_CLASSES
        image = self.imresize(image)/255
        shapeseg = torch.squeeze(self.segresize(shapeseg),dim=0)
        sample = {"Image": image, "ShapeSeg": shapeseg, "ImageClass": label}
        
        if self.is_partially_supervised:
            stype = (idx // NUM_IM_CLASSES) % 4
            if self.drop_type == 'orig':
                if stype == 0:
                    pass
                elif stype == 1:
                    del sample['ShapeSeg']
                elif stype == 2:
                    del sample['ImageClass']
                elif stype == 3:
                    del sample['ShapeSeg']
                    del sample['ImageClass']
            elif self.drop_type == 'drop_shapes':
                if stype in {0}:
                    pass
                else:
                    del sample['ShapeSeg']
            elif self.drop_type == 'drop_class':
                if stype in {0, 1}:
                    pass
                else:
                    del sample['ImageClass']
            else:
                raise ValueError
            
        return sample


class PartialSupervisionSampler(Sampler):

    def __init__(self, num_items, batch_size):
        assert num_items % (NUM_IM_CLASSES * 4) == 0  # must be multiple of 16--good enough for the one experiment we run here.
        super(PartialSupervisionSampler, self)
        self.num_items = num_items
        self.num_batches = self.num_items//batch_size
        self.batch_size = batch_size
        
    def __iter__(self):
        self._build_batches()
        return iter(self.batches)
    
    def __len__(self):
        return self.num_batches
        
    def _build_batches(self):
        print("Building legal batches...")
        self.indices = [[] for i in range(4)]
        for idx in range(self.num_items):
            stype = (idx // NUM_IM_CLASSES) % NUM_IM_CLASSES
            self.indices[stype].append(idx)       
        self.batches = []
        for stype in range(4):
            random.shuffle(self.indices[stype])
        for b in range(self.num_batches):
            self.batches.append([self.indices[b % 4].pop() for i in range(self.batch_size)])
        random.shuffle(self.batches)


class IsParallelogram(nn.Module):
    def __init__(self):
        super(IsParallelogram, self).__init__()
        self.register_buffer('ONE', torch.tensor([1.0],dtype=torch.float32,requires_grad=False), persistent=False)
        
    def forward(self, shapeseg):
        pgrams = torch.sum(shapeseg[:,PARALLELOGRAMS,:,:], dim=1, keepdim=True)
        return torch.cat([pgrams, self.ONE-pgrams], dim=1)
        
        
class EllipseOrPol(nn.Module):
    def __init__(self):
        super(EllipseOrPol, self).__init__()
        self.register_buffer('ONE', torch.tensor([1.0],dtype=torch.float32,requires_grad=False), persistent=False)
        
    def forward(self, shapeseg):
        ellipses = torch.sum(shapeseg[:,ELLIPSES,:,:], dim=1, keepdim=True)
        polygons = torch.sum(shapeseg[:,POLYGONS,:,:], dim=1, keepdim=True)
        return torch.cat([ellipses, polygons, self.ONE-ellipses-polygons], dim=1)


def dict_to_gpu(dict):
    return {key: dict[key].cuda() for key in dict}


def model():

    shape_pred = ResNet(layers=[2,2,2,2], block= BasicBlock, num_classes=NUM_SHAPES+1, active_stages=[0,1,2])
    shape2class = ResNet(layers=[2,2,2,2], block= BasicBlock, num_classes=NUM_IM_CLASSES, active_stages=[3,4,5], input_dim=NUM_SHAPES+1+ELLIPSE_OR_POL_DIM+IS_PARALLELOGRAM_DIM)
    # basically ResNet18
    direct_class_pred = ResNet(layers=[2,2,2,2], block= BasicBlock, num_classes=NUM_IM_CLASSES)

    graph = NeuralGraphicalModel()
    graph.addvar(GaussianVariable(name='Image', predictor_fn=None))  # always observed
    # segmentation of shapes--9 different shapes plus a category for 'no shape/background'
    graph.addvar(CategoricalVariable(NUM_SHAPES+1, name='ShapeSeg', predictor_fn=shape_pred, parents=[graph['Image']]))
    graph.addvar(ProbSpaceCategoricalVariable(ELLIPSE_OR_POL_DIM, name='EllipseOrPol', predictor_fn=EllipseOrPol(), parents=[graph['ShapeSeg']], can_be_supervised=False))
    graph.addvar(ProbSpaceCategoricalVariable(IS_PARALLELOGRAM_DIM, name='IsParallelogram', predictor_fn=IsParallelogram(), parents=[graph['ShapeSeg']], can_be_supervised=False))
    graph.addvar(MultiplyPredictedCategoricalVariable(NUM_IM_CLASSES, name='ImageClass', predictor_fns=[direct_class_pred, shape2class],\
                per_prediction_parents=[[graph['Image']],[graph['ShapeSeg'], graph['IsParallelogram'], graph['EllipseOrPol']]]))
    return graph 


def main_v4(datapath: Path, trainset_sz: int, is_partially_supervised = False, drop_type='orig'):

    # dataset
    train_set = ShapeImageDataset(datapath / "Train", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, trainset_sz, is_partially_supervised=is_partially_supervised, drop_type=drop_type)
    val_set   = ShapeImageDataset(datapath / "Val", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, trainset_sz)
    test_set  = ShapeImageDataset(datapath / "Test", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, TESTSET_SZ)
    
    stopping_set = torch.utils.data.Subset(val_set, list(range(0,trainset_sz//2)))
    calibration_set = torch.utils.data.Subset(val_set, list(range(trainset_sz//2, trainset_sz)))
    
    if is_partially_supervised:
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=NUM_WORKER, batch_sampler=PartialSupervisionSampler(trainset_sz, DEF_BATCH_SIZE))
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    stopping_loader = torch.utils.data.DataLoader(stopping_set, batch_size=DEF_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    calibration_loader = torch.utils.data.DataLoader(calibration_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
    
    # model + training objects
    graph = model().cuda()
    if not is_partially_supervised and trainset_sz > 4000:
        optimizer = torch.optim.Adam(graph.parameters(), lr=DEF_LR, weight_decay=DEF_WEIGHT_DECAY, eps=0.1)
        print("Setting Adam epsilon to 0.1")
    else:
        optimizer = torch.optim.Adam(graph.parameters(), lr=DEF_LR, weight_decay=DEF_WEIGHT_DECAY)
    calibration_optimizer = torch.optim.Adam(graph.calibration_parameters(), lr=DEF_LR, weight_decay=0)
    summary_writer = SummaryWriter()
    scaler = GradScaler()

    # training    
    patience = DEF_PATIENCE_EPOCHS
    epoch_per_check = max(DEF_EPOCH_PER_CHECK, DEF_CALIBRATE_ITER//(trainset_sz//DEF_BATCH_SIZE))
    downweight_iter = 32 * (trainset_sz//DEF_BATCH_SIZE)
    print(f"patience: {patience}")
    print(f"epoch_per_check: {epoch_per_check}")
    iter = 0
    epochs_without_improvement = 0
    best_holdout_loss = float('inf')
    graph.train()    
    nepoch = DEF_TRAIN_ITER // (trainset_sz //DEF_BATCH_SIZE)
    for epoch in range(nepoch):
        graph.train()
        for data in train_loader:
            
            data = dict_to_gpu(data)    
            unsuper_weight = min(1, iter / downweight_iter)
            unsupervised_loss_weight = torch.tensor(unsuper_weight, dtype=torch.float32).cuda()
            
            if is_partially_supervised:
                optimizer.param_groups[0]['weight_decay'] = unsuper_weight * WEIGHT_DECAY
            
            optimizer.zero_grad()
            logthis = (iter == 0 or (iter+1) % (trainset_sz//DEF_BATCH_SIZE) == 0)
            
            with autocast():
                loss = graph.loss(data, unsupervised_loss_weight, samples_in_pass=SAMPLES_PER_PASS, summary_writer=summary_writer if logthis else None, global_step=iter)
            if logthis:
                summary_writer.add_scalar('trainloss', loss, iter)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1
                        
        print(f"Epoch {epoch} complete.")
        
        if epoch % epoch_per_check == 0:
            # must re-calculate calibration prior to stopping check
            graph.reset_calibration()
            graph.validate()
            nepoch = int(DEF_CALIBRATE_ITER / (trainset_sz/2/DEF_BATCH_SIZE))
            cal_iter = 0
            for epoch in range(nepoch):
                for data in calibration_loader:
                    data = dict_to_gpu(data)
                    calibration_optimizer.zero_grad()
                    with autocast():
                        loss = graph.loss(data, None, samples_in_pass=SAMPLES_PER_PASS, \
                                    summary_writer=summary_writer if (cal_iter+1 == DEF_CALIBRATE_ITER) else None, global_step=iter)
          
                    scaler.scale(loss).backward()
                    scaler.step(calibration_optimizer)
                    scaler.update()
                
                    cal_iter += 1
              
            # early stopping
            stopping_loss = 0
            for data in stopping_loader:
                data = dict_to_gpu(data)
                with torch.no_grad():
                    stopping_loss += graph.loss(data, None, samples_in_pass=SAMPLES_PER_PASS, summary_writer=None)
            summary_writer.add_scalar('stopping_loss', stopping_loss, iter)
            if stopping_loss < best_holdout_loss:
                best_holdout_loss = stopping_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += epoch_per_check
                if epochs_without_improvement >= patience:
                    break
              
    print(f"Training stopped after {iter} iterations.")

    # temperature scaling
    graph.validate()
    nepoch = int(DEF_CALIBRATE_ITER / (trainset_sz/2/DEF_BATCH_SIZE))
    for epoch in range(nepoch):
        for data in calibration_loader:
            data = dict_to_gpu(data)
            calibration_optimizer.zero_grad()
            with autocast():
                loss = graph.loss(data, None, samples_in_pass=SAMPLES_PER_PASS, \
                            summary_writer=summary_writer if ((iter+1) % int(DEF_CALIBRATE_ITER / (trainset_sz/2/DEF_BATCH_SIZE)) == 0) else None, global_step=iter)
  
            scaler.scale(loss).backward()
            scaler.step(calibration_optimizer)
            scaler.update()
        
            iter += 1
    
    print(f"Calibration done after {iter} iterations total.")
    torch.save(graph.state_dict(), f"shape_graph_trsz_{trainset_sz}.pth")            
        
    # evaluation
    graph.eval()
    confusion = np.zeros((NUM_IM_CLASSES, NUM_IM_CLASSES), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            data = dict_to_gpu(data)
            observations = {'Image': data['Image']}
            marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass'], samples_per_pass=32, num_passes=1)
            confusion[data['ImageClass'], torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs))] += 1

    print(f"Confusion Matrix: {confusion}")
    

def benchmark_v4(datapath: Path, trainset_sz: int):
    """
    Train normal ResNet18 on data and evaluate performance.
    """
    
    # dataset
    train_set = ShapeImageDataset(datapath / "Train", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, trainset_sz)
    val_set   = ShapeImageDataset(datapath / "Val", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, trainset_sz)
    test_set  = ShapeImageDataset(datapath / "Test", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, TESTSET_SZ)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=DEF_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=DEF_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
    
    # model + training objects
    model = ResNet(layers=[2,2,2,2], block= BasicBlock, num_classes=NUM_IM_CLASSES).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=DEF_LR, weight_decay=DEF_WEIGHT_DECAY)
    summary_writer = SummaryWriter()
    scaler = GradScaler()

    # training
    patience = DEF_PATIENCE_EPOCHS
    print(f"patience: {patience}")
    iter = 0
    epochs_without_improvement = 0
    best_holdout_loss = float('inf')    
    nepoch = DEF_TRAIN_ITER // (trainset_sz //DEF_BATCH_SIZE)
    for epoch in range(nepoch):
        model.train()
        for data in train_loader:

            data = dict_to_gpu(data)
            optimizer.zero_grad()
            with autocast():
                output = model(data['Image'])
                loss = loss_fn(output, data['ImageClass'])
            
            if (iter == 0 or (iter +1) % trainset_sz//DEF_BATCH_SIZE == 0):
                summary_writer.add_scalar('trainloss', loss, iter)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter += 1

        # early stopping
        # we can check every epoch because no calibration is needed
        model.eval()
        stopping_loss = 0
        for data in val_loader:
            data = dict_to_gpu(data)
            with torch.no_grad():
                output = model(data['Image'])
                stopping_loss += loss_fn(output, data['ImageClass'])
        summary_writer.add_scalar('stopping_loss', stopping_loss, iter)
        if stopping_loss < best_holdout_loss:
            best_holdout_loss = stopping_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break
        print(f"Epoch {epoch} complete.")
    
    torch.save(model.state_dict(), f"shape_resnet_trsz_{trainset_sz}.pth")    
    print(f"Training stopped after {iter} iterations.")
        
    # evaluation
    model.eval()
    confusion = np.zeros((NUM_IM_CLASSES, NUM_IM_CLASSES), dtype=np.int32)
    with torch.no_grad():
        for data in test_loader:
            data = dict_to_gpu(data)
            output = torch.squeeze(model(data['Image']))
            confusion[data['ImageClass'], torch.argmax(output)] += 1

    print(f"Confusion Matrix: {confusion}")
    
    
def visualize_mistakes(netpath: Path, datapath: Path, savedir: Path):
    """
    Print out some of the images that the NGM model gets wrong, along with predictions.
    """
    
    graph = model().cuda()
    graph.load_state_dict(torch.load(str(netpath)))
    
    test_set  = ShapeImageDataset(datapath / "Test", IMAGE_INPUT_SZ, SHAPE_SEG_SZ, TESTSET_SZ)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=NUM_WORKER)
    
    trans = transforms.ToPILImage(mode='RGB')
    color_map = torch.tensor(   [[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0],
                                [1, 1, 1],
                                [0.5, 0.5,  0],
                                [0, 0.5, 0.5]])
    
    # evaluation
    torch.set_printoptions(threshold=10000)
    graph.eval()
    confusion = np.zeros((NUM_IM_CLASSES, NUM_IM_CLASSES), dtype=np.int32)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = dict_to_gpu(data)
            observations = {'Image': data['Image']}
            marginals = graph.predict_marginals(observations, to_predict_marginal=['ImageClass', 'ShapeSeg'], samples_per_pass=DEF_BATCH_SIZE, num_passes=1)
            
            y = data['ImageClass'].cpu().item()
            y_pred = torch.argmax(torch.squeeze(marginals[graph['ImageClass']].probs)).cpu().item()
            
            if y != y_pred:
                subdir = "Mistakes"
            else:
                subdir = "Correct Classifications"
            
            savepath = savedir / subdir / f"im_{i}_y_{y}_pred_{y_pred}.png"
            segpath = savedir / subdir / f"seg_{i}_y_{y}_pred_{y_pred}.png"

            imdat = data['Image'][0]
            trans(imdat).save(savepath)

            seg = marginals[graph['ShapeSeg']].sample()[0]
            seg_im = color_map[seg]
            trans(torch.movedim(seg_im, 2, 0)).save(segpath)
                
            print("Saved  images.")