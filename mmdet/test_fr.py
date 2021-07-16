import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.nn as nn
import torch.nn.functional as F



import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


import pathlib
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse_conv_bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    # parser.add_argument(
        # '--options', nargs='+', action=MultipleKVAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument('--mia_meta', default='',
                        help='Directory for meta mia file')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
    
    




if False:
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and open_images.')

    parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_dataset', help='Dataset directory path')
    parser.add_argument('--balance_data', action='store_true',
                        help="Balance training data by down-sampling more frequent labels.")



    parser.add_argument('--split'  , default=None, help='split of dataset')
    parser.add_argument('--onlyflip', default=0, type=int, help='split of dataset')
    parser.add_argument('--norandcrop', default=0, type=int, help='split of dataset')


    parser.add_argument('--net', default="vgg16-ssd",
                        help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument('--freeze_base_net', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")

    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')
    parser.add_argument("--nms_method", type=str, default="hard")


    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                        help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")

    # Params for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', default=120, type=int,
                        help='the number epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int,
                        help='the number epochs')
    parser.add_argument('--debug_steps', default=100, type=int,
                        help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')

    parser.add_argument('--checkpoint_folder', default='models/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--mia_meta', default='',
                        help='Directory for meta mia file')
    parser.add_argument('--trained_model', default='',
                        help='trained_model')
    parser.add_argument('--label_file', default='',
                        help='label files')

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()



DEVICE = torch.device("cuda:0")
args =parse_args()

torch.backends.cudnn.benchmark = True
logging.info("Use Cuda.")


####################################### PARAMETER ZONE #################################################
MAX_LEN           = 5000
shuffle_boxes     = False
input_size        = 300
ball_size         = input_size*0.1
QUICK             = 0 
TRANSFORM         = True
SAVE_MODEL        = True
PRETRAIN          = False
CANDIDATE_SIZE    = -1 


EPOCHS = 90
NMS_THRES = 1.0
CANVAS_TYPE = 'uniform'
LOG_SCORE = 2 
ATTACK_MODEL = 'squeezenet1_1' # squeezenet1_1

########################################################################################################



args = parse_args()
cfg = mmcv.Config.fromfile(args.config)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
dataset_train= build_dataset(cfg.data_07_left.train)
dataset_test = build_dataset(cfg.data_07_left.test)

data_loader_train = build_dataloader(
        dataset_train,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
data_loader_test = build_dataloader(
        dataset_test,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
        
        
# dataset_train = VOCDataset_split(voc2007_dataset     , is_test=False, split = args.split)
# dataset_test  = VOCDataset_split(voc2007_dataset_test, is_test=True , split = args.split)

print(len(dataset_train))
print(len(dataset_test))



def generate_pointsets(data_loader,model,num_logit_feature=1,max_len=MAX_LEN,input_size = 300,regard_in_set=True,shuffle_boxes = shuffle_boxes):
    
    size300_train07_dataset = []
    max_feat_len = 0
    min_feat_len = 50000
    mark = 0
    jump = 10
    # with chainer.using_config('train', False),chainer.function.no_backprop_mode():
    with torch.no_grad():

        feat_len_info = []
        for i, data in enumerate(data_loader):
            tr_bboxes_list=[]
            tr_labels_list=[]
            tr_scores_list=[]
            tr_img_sizes= []
            x = []
            sizes = []
            #for img in imgs[mark:mark+jump]:
            
            
            # image = dataset.get_image(i)
            # data  = dataset[i]
            image = data['img'][0]
            # data['img'][0] = image[None]
            # print(data)
            # print(image.shape ,'image.shape' )
            show = False
            with torch.no_grad():
                result = model(return_loss=False, rescale=not show, **data)
            
            
           
    return size300_train07_dataset





# true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
if  True:
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg_custom)
    model = MMDataParallel(model, device_ids=[0])

    
    
    

for i, data in enumerate(data_loader):

    x = []
    sizes = []

    image = data['img'][0]

    show = False
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            result = model(return_loss=False, rescale=not show, **data)


    


results = []




# print(MIA_META_FILE)

#a_train = np.array(size300_meta_train_dataset)

generate_pointsets(data_loader_train,model ,regard_in_set=True ,num_logit_feature=1)
# size300_meta_test_dataset += generate_pointsets(data_loader_test ,model ,regard_in_set=False,num_logit_feature=1)