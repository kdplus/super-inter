import sys
import getopt
import math
import torch
import PIL
import PIL.Image
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np
import argparse
import os
import time
import dataset
from datetime import datetime
import random
from random import shuffle
import numpy
import numpy as np
#`import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import gc
from skimage import data, img_as_float
from skimage.measure import compare_ssim as get_ssim

from utils.image_utils import imwrite
from utils.flow_utils import bilinear_interp, bi_interp, to_there, forward_warp, meshgrid, prop_refine_rd, prop_refine_which
from utils.vis_utils import viz_flow
from utils.loss_utils import TVLoss
import json
import adabound

from model import FlowNet, SRNet, SharpNet_C2, SharpNet_C2_rf

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # change this if you have a multiple graphics cards and you want to utilize them
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
devices = [0]
torch.cuda.set_device(devices[0])


with open('settings.json') as f:
    settings = json.load(f)

# lr = settings['lr']
handle_size = 256
Max_steps = 10000000
batch_size = settings['batch_size']
keep_training = True
show_img_gap = 200
best_psnr = 0


FNet = SharpNet_C2(96, 48)
FNet = nn.DataParallel(FNet, device_ids=devices).cuda()

FNet_new = SharpNet_C2_rf(96, 48)
FNet_new = nn.DataParallel(FNet_new, device_ids=devices).cuda()

FNet.load_state_dict(torch.load('FNet.pkl'))


FNet_new.module.C0 = FNet.module.C0
FNet_new.module.D1 = FNet.module.D1
FNet_new.module.D2 = FNet.module.D2
FNet_new.module.D3 = FNet.module.D3
FNet_new.module.U1 = FNet.module.U1
FNet_new.module.U2 = FNet.module.U2
FNet_new.module.U3 = FNet.module.U3
FNet_new.module.C4 = FNet.module.C4
FNet_new.module.C5 = FNet.module.C5
FNet_new.module.C6 = FNet.module.C6

torch.save(FNet_new.state_dict(), 'pt_FNet_new.pkl')