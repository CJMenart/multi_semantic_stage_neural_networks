"""
Maybe the 'top-level' script of detection_exp.

Take our proposed NGM for 2d object detection, produced by det_ngm,
and train it on the Cityscapes dataset.

Usage:
python -m torchrun \
    --nproc_per_node=8 [Or number of GPUs] \
    train_det_ngm.py \
    [path to Cityscapes]

or

python -m torch.distributed.launch \
    --use_env \ 
    --nproc_per_node=8 [Or number of GPUs] \
    train_det_ngm.py \
    [path to Cityscapes]
    

TODO final pass to make sure we remembered to copy all FCOS stuff????
"""

from neural_graphical_model.detection_exp import cityscapes_dataset
from neural_graphical_model.detection_exp.det_ngm import *  # all versions of 'build_model'
from neural_graphical_model.detection_exp.fcos import fcos_targets
from neural_graphical_model.detection_exp.extra_cityscapes_sampler import ExtraCityscapesSampler
from neural_graphical_model.utils import dict_to_gpu, DictToTensors, ImToTensor
from neural_graphical_model.distributed_graphical_model import DDPNGM
import torch 
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ConstantLR, SequentialLR
import torchvision
import sys, os
import random
import timeit
import warnings
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writeable")
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

SEED = 1492
H = 2**10
W = 2**11
FORCE_PREDICT_CHANCE = 0.25
# Hyparparams taken from FCOS, except where absolutely had to change
LR = 0.01  #0.01
L2 = 0.0001
#BATCH_SIZE = 2 #16
BIAS_LR_FACTOR = 2
BIAS_L2 = 0
MOMENTUM = 0.9
CAL_MOMENTUM = 0
GAMMA = 0.1
CAL_LR = LR
CAL_MOMENTUM = 0
NUM_WORKER = 1
NUM_CHECKPOINTS = 1
ITR_PER_CHECKPOINT = 90000
MAX_ITR = ITR_PER_CHECKPOINT*NUM_CHECKPOINTS
MILESTONES = (MAX_ITR//9*6, MAX_ITR//9*8)
SUPERVISION_SCHEDULE = (MAX_ITR//3, MAX_ITR//3*2)  # how long to only use supervised losses
CALIBRATE_EPOCH = 128
CLIP = 64
NVAL = 16
CACHE_TRAINDAT = True
CAL_IN_TRAIN = False
ITR_PRINT_GNORM = 256
WARMUP_ITER = 500*NUM_CHECKPOINTS
WARMUP_FACTOR = 3.0
SUPERVISED_ITR = 500


def roll_forced_predictions(graph, itr):
    force_predicted_input = []
    if itr < SUPERVISION_SCHEDULE[0] or FORCE_PREDICT_CHANCE == 0:
        return force_predicted_input
    elif itr >= SUPERVISION_SCHEDULE[1]:
        force_predict_chance = FORCE_PREDICT_CHANCE
    else:
        force_predict_chance = FORCE_PREDICT_CHANCE * (itr - SUPERVISION_SCHEDULE[0]) / (SUPERVISION_SCHEDULE[1] - SUPERVISION_SCHEDULE[0])
    
    for var in graph:
        if var.name != "Image" and var.name != "Depth" and var.can_be_supervised and random.random() < force_predict_chance:
            force_predicted_input.append(var)
    # Roll both depth + Seg or neither
    if graph["Segmentation"] in force_predicted_input:
        force_predicted_input.append(graph["Depth"])
    
    """
    # Version used for cyclic model
    if random.random() < force_predict_chance:
        force_predicted_input.append(graph["Depth"])
        force_predicted_input.append(graph["Segmentation"])
    """
    return force_predicted_input

# Default normal training on regular Cityscapes only for now
def train(cityscapes_root, local_rank, global_rank, world_size, BATCH_SIZE=2, batch_per_update=1, load_path=None, use_trainextra=False):

    dist.init_process_group("gloo")
    random.seed(SEED)
    
    #model
    logger.info("Building model...")
    # You can build differnet kinds of models, but may want to use different transforms if you are discrete
    #graph = build_model_loopy(global_rank)
    graph = build_model_progressive(global_rank)

    logger.debug(f"Sending model to CUDA {local_rank}...")
    graph.cuda(local_rank)
    logger.debug("Parallelizing model...")
    DDPNGM(graph, device_ids=[local_rank])
    weights_savename = f"det_graph.pth"
    #torch.save(graph.state_dict(), f"initial_" + weights_savename)            

    params = [] 
    for key, value in graph.named_parameters():
        if not value.requires_grad:
            continue
        lr = LR
        weight_decay = L2
        if "bias" in key:
            lr = lr * BIAS_LR_FACTOR
            weight_decay = BIAS_L2
        if "offsets" in key or "combination" in key:
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, LR, momentum=MOMENTUM)
    warmup = ConstantLR(optimizer, factor=1/WARMUP_FACTOR, total_iters=WARMUP_ITER)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, MILESTONES, gamma=GAMMA)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, scheduler], milestones=[WARMUP_ITER])

    logger.debug(f"calibration parameters: {graph.calibration_parameters()}")    
    calibration_optimizer = torch.optim.SGD(graph.calibration_parameters(), lr=CAL_LR, weight_decay=0, momentum=CAL_MOMENTUM)
   
    summary_writer = SummaryWriter() if global_rank == 0 else None
    scaler = GradScaler()
    
    logger.info("Building training data...")
    format_transform = fcos_targets.FCOSTargets()
    #to_tensor_transform = DictToTensors()
    resize = fcos_targets.FCOSStyleResize(image_size=(W,H))
    imtrans = ImToTensor()
    auxload = fcos_targets.AuxilliaryToTensor()
    # Discretization transforms--use only if running discrete model
    # =========================================
    disc_centerness = fcos_targets.BinarizeCenterness()
    disc_bbox = fcos_targets.DiscretizeBbox()
    # =========================================
    flip = fcos_targets.RandomLeftRightFlip()
    normalize = fcos_targets.NormalizeIm()
    boxcls = fcos_targets.BooleanBoxCls()
    depth = fcos_targets.DontCareDepth()
    # If resize is dynamic, cannot cache transforms
    cached_transforms = torchvision.transforms.Compose([imtrans, auxload, resize, format_transform, normalize, boxcls, depth])
    # For discrete model
    #cached_transforms = torchvision.transforms.Compose([imtrans, auxload, resize, format_transform, normalize, boxcls, disc_centerness, disc_bbox, depth])
    transforms = flip
    train_data = cityscapes_dataset.Cityscapes(
        root = cityscapes_root,
        split = "train",
        mode = "fine",
        target_type = ['Depth', 'Segmentation', 'Detection'],
        cached_transforms = cached_transforms,
        transforms = transforms,
        cache = CACHE_TRAINDAT)

    inds = list(range(len(train_data)))
    random.shuffle(inds)
    val_data = torch.utils.data.Subset(train_data, inds[:NVAL])
    train_data = torch.utils.data.Subset(train_data, inds[NVAL:])
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=global_rank, shuffle=True, seed=SEED)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=world_size, rank=global_rank, shuffle=True, seed=SEED)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKER, pin_memory=False)
    calibration_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=NUM_WORKER)
    
    if use_trainextra:
        trainextra_data = cityscapes_dataset.Cityscapes(
                root = cityscapes_root,
                split = "train_extra",
                mode = "coarse",
                target_type = ['Depth'],
                cached_transforms = None,
                transforms = torchvision.transforms.Compose([imtrans, auxload, resize, normalize]),
                cache = False)
        combined_train_data = torch.utils.data.ConcatDataset([train_data, trainextra_data])
        trainextra_sampler = ExtraCityscapesSampler(combined_train_data, num_replicas=world_size, \
                rank=global_rank, seed=SEED, batch_size=BATCH_SIZE, nsupervised=len(train_data))
        trainextra_loader = torch.utils.data.DataLoader(combined_train_data, batch_sampler=trainextra_sampler, num_workers=NUM_WORKER)

    # Time training
    start_sec = timeit.default_timer()
    logger.info("Time to start training!")
    # loop over epochs but stop condition will be based on itr
    itr = 0
    accum_idx = 0
    loss_accum = 0
    epoch = 0

    if load_path:
        state_dict = torch.load(load_path)
        itr = state_dict['itr']
        optimizer.load_state_dict(state_dict['optimizer'])
        graph.load_state_dict(state_dict['model'])

    while(True):
        first_itr_in_epoch = True
        graph.train()
        sampler.set_epoch(epoch)
        epoch += 1

        cur_train_loader = trainextra_loader if (use_trainextra and itr >= SUPERVISED_ITR) else train_loader

        for data in cur_train_loader:
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"data shapes: {[(k, data[k].shape) for k in data]}")

            with autocast(enabled=True, cache_enabled=False):
                data = dict_to_gpu(data, local_rank)
                loss = graph.loss(data,  \
                    samples_in_pass=1, 
                    force_predicted_input = [] if use_trainextra else roll_forced_predictions(graph, itr),
                    summary_writer=summary_writer if first_itr_in_epoch else None, 
                    #summary_writer=summary_writer, 
                    global_step=itr)
            if first_itr_in_epoch:
                first_itr_in_epoch = False
                if summary_writer:
                    summary_writer.add_scalar('trainloss', loss.detach(), itr)
            loss = loss / batch_per_update  # This may be redundant w/ grad clipping but whatevs
            if local_rank == 0:
                loss_accum += loss.detach().cpu()
            scaler.scale(loss).backward()
            #loss.backward()
            accum_idx = (accum_idx + 1) % batch_per_update
            loss = None

            if accum_idx == 0:

                if local_rank == 0:
                    logger.info(f"itr {itr} trainloss: {loss_accum}")
                    loss_accum = 0

                scaler.unscale_(optimizer)
                # DEBUG
                if itr % ITR_PRINT_GNORM == 0:
                    total_norm = 0
                    for p in graph.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    logger.info(f"grad norm: {total_norm}")
                torch.nn.utils.clip_grad_norm_(graph.parameters(), CLIP)
                
                scaler.step(optimizer)
                #optimizer.step()
                scaler.update()
                scheduler.step()  # goes herabouts
                itr += 1
                optimizer.zero_grad()

            if accum_idx == 0 and itr % ITR_PER_CHECKPOINT == 0:
               
                if CAL_IN_TRAIN:
                    # calibrate and save
                    graph.reset_calibration()
                    graph.validate()
                    for epoch in range(CALIBRATE_EPOCH):
                        for calidx, data in enumerate(calibration_loader):
                            calibration_optimizer.zero_grad()
                            with autocast(enabled=False):
                                data = dict_to_gpu(data, rank=local_rank)
                                loss = graph.loss(data, samples_in_pass=1, force_predicted_input=[rvar for rvar in graph if rvar.name != 'Image'],\
                                        summary_writer=summary_writer if (epoch==(CALIBRATE_EPOCH-1) and calidx==0) else None, global_step=itr)
                        
                            #scaler.scale(loss).backward()
                            loss.backward()
                            #scaler.unscale_(calibration_optimizer)
                            torch.nn.utils.clip_grad_norm_(graph.calibration_parameters(), CLIP)
                            #scaler.step(calibration_optimizer)
                            calibration_optimizer.step()
                            #scaler.update()
                            loss = None
                    graph.train()
                if global_rank == 0:
                    rnd = itr//ITR_PER_CHECKPOINT

                    state_dict = {
                        'itr': itr,
                        'model': graph.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }
                    torch.save(state_dict, f"round_{rnd}_" + weights_savename)            
                    logger.info(f"Saved rnd {rnd} network.")

            if itr >= MAX_ITR:
                break
      
        if itr >= MAX_ITR:
            break
        
        if global_rank == 0:
            state_dict = {
                'itr': itr,
                'model': graph.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(state_dict, weights_savename)
        logger.info(f"Epoch complete.")
        sys.stdout.flush()
        
    logger.info(f"Training stopped after {itr} iterations.")
    stop_sec = timeit.default_timer()
    logger.info(f"Total training time: {stop_sec - start_sec} seconds.")
    dist.destroy_process_group()


if __name__=='__main__':
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else torch.cuda.device_count()
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    logger.debug(f"local rank {local_rank}")
    logger.debug(f"global rank {global_rank}")
    logger.debug(f"num_gpus {num_gpus}")
    dataroot = sys.argv[1]
    BATCH_SIZE = int(sys.argv[2])
    load_path = None
    if len(sys.argv) >= 4:
        batch_per_update = int(sys.argv[3])
    else:
        batch_per_update = 1
    if len(sys.argv) >= 5:
        load_path = sys.argv[4]
        if load_path == 'None':
            load_path = None
    if len(sys.argv) >= 6:
        use_trainextra = bool(sys.argv[5])
    else:
        use_trainextra = False
    logger.handlers[0].flush()
    train(dataroot, local_rank, global_rank, num_gpus, BATCH_SIZE, batch_per_update, load_path, use_trainextra)
