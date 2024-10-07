"""
Specifically: Caibrate the NGM more, and keep a careful tensorboard record, in 
order to see if it was calibrated sufficeintly the first time.

Mostly just a copy-paste of the relevant bits from train_det_ngm.
"""

from neural_graphical_model.detection_exp import cityscapes_dataset
from neural_graphical_model.detection_exp.det_ngm import build_model, build_model_baseline, build_model_loopy, build_model_discrete, build_model_progressive
from neural_graphical_model.detection_exp.fcos import fcos_targets
from neural_graphical_model.detection_exp.train_det_ngm import SEED, H, W, CAL_LR, L2, CAL_MOMENTUM, GAMMA, NUM_WORKER, CALIBRATE_EPOCH, CLIP, NVAL, NUM_CHECKPOINTS
from neural_graphical_model.utils import dict_to_gpu, DictToTensors, ImToTensor
from neural_graphical_model.distributed_graphical_model import DDPNGM
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
CACHE_TRAINDAT = True


def cal(cityscapes_root, global_rank, local_rank, world_size, netpath, BATCH_SIZE=1):

    dist.init_process_group("nccl")
    random.seed(SEED)
    
    #model
    logger.info("Building model...")
    graph = build_model_progressive(global_rank)
    graph.cuda(local_rank)
    DDPNGM(graph, device_ids=[local_rank])
    netpath = Path(netpath)
    savename = netpath.parent / ('extra_calibrated_' + netpath.name)
    
    weights_dict = torch.load(netpath)
    if 'model' in weights_dict:
        graph.load_state_dict(weights_dict['model'])
    else:
        graph.load_state_dict(weights_dict)

    # training and optimization objects
    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")    
    calibration_optimizer = torch.optim.SGD(graph.calibration_parameters(), lr=CAL_LR, weight_decay=0, momentum=CAL_MOMENTUM)
   
    summary_writer = SummaryWriter() if global_rank == 0 else None
    scaler = GradScaler()
    
    logger.info("Building training data...")
    format_transform = fcos_targets.FCOSTargets()
    #to_tensor_transform = DictToTensors()
    resize = fcos_targets.FCOSStyleResize(image_size=(W,H))
    disc_centerness = fcos_targets.BinarizeCenterness()
    disc_bbox = fcos_targets.DiscretizeBbox()
    #auxload = fcos_targets.AuxilliaryToTensor()
    imtrans = ImToTensor()
    normalize = fcos_targets.NormalizeIm()
    boxcls = fcos_targets.BooleanBoxCls()
    flip = fcos_targets.RandomLeftRightFlip()
    cached_transforms = torchvision.transforms.Compose([imtrans, resize, format_transform, normalize, boxcls])

    transforms = flip
    train_data = cityscapes_dataset.Cityscapes(
        root = cityscapes_root,
        split = "train",
        mode = "fine",
        target_type = ['Detection'],
        #target_type = ['Depth', 'Segmentation', 'Detection'],
        cached_transforms = cached_transforms,
        transforms = transforms,
        cache = CACHE_TRAINDAT
        )
    

    inds = list(range(len(train_data)))
    random.shuffle(inds)
    val_data = torch.utils.data.Subset(train_data, inds[:NVAL])
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=global_rank, shuffle=True, seed=SEED)
    calibration_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKER)
    
    # loop over epochs but stop condition will be based on itr
    itr = 0
    # calibrate and save
    #graph.reset_calibration()  # should laready be reset...
    graph.validate()
    for epoch in range(CALIBRATE_EPOCH):
        for calidx, data in enumerate(calibration_loader):
            calibration_optimizer.zero_grad()
            with autocast():
                data = dict_to_gpu(data, rank=local_rank)
                loss = graph.loss(data, samples_in_pass=1, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'],\
                                        summary_writer=summary_writer, global_step=itr)
                
            scaler.scale(loss).backward()
            scaler.unscale_(calibration_optimizer)
            torch.nn.utils.clip_grad_norm_(graph.calibration_parameters(), CLIP)
            scaler.step(calibration_optimizer)
            scaler.update()
            loss = None
            itr += 1
                       
    if global_rank == 0:
        torch.save(graph.state_dict(), savename)            

    dist.destroy_process_group()
    logger.info("Done.")


if __name__=='__main__':
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else torch.cuda.device_count()
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    dataroot = sys.argv[1]
    netpath = sys.argv[2]
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    else:
        batch_size = 1
    cal(dataroot, global_rank, local_rank, num_gpus, netpath, batch_size)
