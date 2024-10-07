"""
Calibrate DetNGM on small amount of target data to investigate impact of domain shift on performance.
"""

from . import bdd100k_dataset
from .det_ngm import build_model
from .fcos import fcos_targets
from .train_det_ngm import SEED, CAL_LR, L2, CAL_MOMENTUM, GAMMA, NUM_WORKER, CALIBRATE_EPOCH, CLIP, NVAL, NUM_CHECKPOINTS
from ..utils import dict_to_gpu, DictToTensors
from ..distributed_graphical_model import DDPNGM
import torch 
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys, os
import random
import timeit
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writeable")
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
TH=720
TW=1280


def cal(bdd_root, json, rank, world_size, netpath, BATCH_SIZE=1):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    random.seed(SEED)
    
    #model
    logger.info("Building model...")
    graph = build_model(rank, TH, TW)
    graph.cuda(rank)
    DDPNGM(graph, device_ids=[rank])
    netpath = Path(netpath)
    savename = netpath.parent / ('extra_target_calibrated_' + netpath.name)
    
    weights_dict = torch.load(netpath)
    if 'model' in weights_dict:
        graph.load_state_dict(weights_dict['model'])
    else:
        graph.load_state_dict(weights_dict)

    # training and optimization objects
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")    
    calibration_optimizer = torch.optim.SGD(graph.calibration_parameters(), lr=CAL_LR, weight_decay=0, momentum=CAL_MOMENTUM)
   
    summary_writer = SummaryWriter() if rank == 0 else None
    scaler = GradScaler()
    
    logger.info("Building training data...")
    format_transform = fcos_targets.FCOSTargets(TH, TW)
    #to_tensor_transform = DictToTensors()
    resize = fcos_targets.ResizeDetectionTargets()
    centerness = fcos_targets.BinarizeCenterness()
    #depth = fcos_targets.DontCareDepth()
    normalize = fcos_targets.NormalizeIm()
    flip = fcos_targets.RandomLeftRightFlip()
    transforms = torchvision.transforms.Compose([format_transform, resize, normalize, flip, centerness])
    
    cal_data = bdd100k_dataset.BDD100KDataset(
        root = bdd_root,
        det_json = json,
        transforms = transforms)
    
    inds = list(range(len(cal_data)))
    random.shuffle(inds)
    cal_data = torch.utils.data.Subset(cal_data, inds[:NVAL])
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(cal_data, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    calibration_loader = torch.utils.data.DataLoader(cal_data, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKER)
    
    # loop over epochs but stop condition will be based on itr
    itr = 0
    # calibrate and save
    #graph.reset_calibration()  # should laready be reset...
    graph.validate()
    for epoch in range(CALIBRATE_EPOCH):
        for calidx, data in enumerate(calibration_loader):
            calibration_optimizer.zero_grad()
            with autocast():
                data = dict_to_gpu(data, rank=rank)
                loss = graph.loss(data, samples_in_pass=1, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'],\
                                        summary_writer=summary_writer, global_step=itr)
                
            scaler.scale(loss).backward()
            scaler.unscale_(calibration_optimizer)
            torch.nn.utils.clip_grad_norm_(graph.calibration_parameters(), CLIP)
            scaler.step(calibration_optimizer)
            scaler.update()
            loss = None
            itr += 1
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"epoch {epoch}")
                       
    if rank == 0:
        torch.save(graph.state_dict(), savename)            

    dist.destroy_process_group()
    logger.info("Done.")


if __name__=='__main__':
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])
    dataroot = sys.argv[1]
    json = sys.argv[2]
    netpath = sys.argv[3]
    cal(dataroot, json, rank, num_gpus, netpath)
