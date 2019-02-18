import sys
import getopt
import math
import torch
import torch.utils.serialization
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
from skimage.measure import compare_ssim as ssim

from utils.image_utils import imwrite
from utils.flow_utils import bilinear_interp, bi_interp, to_there, forward_warp, meshgrid, prop_refine_rd, prop_refine_which
from utils.vis_utils import viz_flow
from utils.loss_utils import TVLoss

from model import Pyramid, Network, SharpNet, _NetG, Improc

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change this if you have a multiple graphics cards and you want to utilize them
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
devices = [0]
torch.cuda.set_device(devices[0])

handle_size = 256
Max_steps = 10000000
batch_size = 3
keep_training = True
show_img_gap = 200
best_psnr = 0

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default = None)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--pretrained', type=str, default="False")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--data', type=str, default="middlebury")
args = parser.parse_args()
#print args.gpus
#print args.batch_size

def show(img):
    npimg = img.numpy()
    return npimg[[2, 1, 0],:,:]

def show_f(img):
    npimg = img.numpy()
    #return (np.transpose(npimg, (1,2,0))).astype(numpy.uint8)
    return npimg


def clip(x, a, b):
    if x > torch.autograd.variable(b):
        return torch.autograd.variable(b)
    if x < torch.autograd.variable(a):
        return torch.autograd.variable(a)
    return x


coding = SharpNet(6, 5)
coding = nn.DataParallel(coding, device_ids=devices).cuda()

improc = Improc()
improc = nn.DataParallel(improc, device_ids=devices).cuda()

real_step = 0
coding.train()
improc.train()

if args.pretrained == "True":
    print("Load trained Net...")
    coding.load_state_dict(torch.load('best_coding.pkl'))
    improc.load_state_dict(torch.load('best_improc.pkl'))
    best_psnr = torch.load('best_psnr.pkl')
    print("Loaded")

reg = 10
cross_entory = torch.nn.CrossEntropyLoss()
criterion = torch.nn.L1Loss()
c2 = torch.nn.MSELoss()
lr = 0.00001
optimizer = optim.Adam(list(coding.parameters())+list(improc.parameters()), lr=lr, weight_decay=1e-5)
tvLoss = TVLoss()

def process(variableJoin, flowout, handle_size=handle_size, scale_down=1):
    batch_size = flowout.shape[0]
    variableUpsample1 = flowout
    flow10 = variableUpsample1[:, 0:2, :, :] / scale_down
    flow01 = variableUpsample1[:, 2:4, :, :] / scale_down
    mask = variableUpsample1[:, 4, :, :]

    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()

    flowt0 = -0.25 * flow01 + 0.25 * flow10
    flowt1 = 0.25 * flow01 - 0.25 * flow10

    coor_x_1 = grid_x + flowt0[:, 0, :, :]
    coor_y_1 = grid_y + flowt0[:, 1, :, :]

    coor_x_2 = grid_x + flowt1[:, 0, :, :]
    coor_y_2 = grid_y + flowt1[:, 1, :, :]

    output_1 = bilinear_interp(variableJoin[:, 0:3, :, :], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(variableJoin[:, 3:6, :, :], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)

    mask = mask.view(batch_size, -1, handle_size, handle_size).repeat(1, 3, 1, 1)
    net = (mask * output_1) + (1.0 - mask) * output_2
    return net, output_1, output_2, flow10, flow01

def process_fixed(variableJoin, flowout, handle_size=handle_size, scale_down=1):
    batch_size = flowout.shape[0]
    variableUpsample1 = flowout
    flow10 = variableUpsample1[:, 0:2, :, :] / scale_down
    flow01 = variableUpsample1[:, 2:4, :, :] / scale_down
    mask = variableUpsample1[:, 4, :, :]

    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()

    flowt0 = flow10 / 2.0
    flowt1 = flow01 / 2.0

    coor_x_1 = grid_x + flowt0[:, 0, :, :]
    coor_y_1 = grid_y + flowt0[:, 1, :, :]

    coor_x_2 = grid_x + flowt1[:, 0, :, :]
    coor_y_2 = grid_y + flowt1[:, 1, :, :]

    output_1 = bilinear_interp(variableJoin[:, 0:3, :, :], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(variableJoin[:, 3:6, :, :], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)

    mask = mask.view(batch_size, -1, handle_size, handle_size).repeat(1, 3, 1, 1)
    net = (mask * output_1) + (1.0 - mask) * output_2
    return net, output_1, output_2, flow10, flow01

def process_net2(variableJoin, flowout, handle_size=handle_size, scale_down=1):
    batch_size = flowout.shape[0]
    variableUpsample1 = flowout
#scale_down = flowout.shape[2] / 2.0
#scale_down = 1
    handle_size = flowout.shape[2]
    flowt0 = flowout[:, 0:2, :, :] / scale_down
    flowt1 = flowout[:, 2:4, :, :] / scale_down

    mask = variableUpsample1[:, 4, :, :]

    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()

    coor_x_1 = grid_x + flowt0[:, 0, :, :]
    coor_y_1 = grid_y + flowt0[:, 1, :, :]

    coor_x_2 = grid_x + flowt1[:, 0, :, :]
    coor_y_2 = grid_y + flowt1[:, 1, :, :]

    output_1 = bilinear_interp(variableJoin[:, 0:3, :, :], coor_x_1, coor_y_1, 'interpolate')
    output_2 = bilinear_interp(variableJoin[:, 3:6, :, :], coor_x_2, coor_y_2, 'interpolate')

    mask = 0.5 * (1.0 + mask)

    mask = mask.view(batch_size, -1, handle_size, handle_size).repeat(1, 3, 1, 1)
    net = (mask * output_1) + (1.0 - mask) * output_2
    return net, output_1, output_2, flowt0, flowt1

def blending(f0, f1, mask):
    batch_size = f0.shape[0]
    handle_size_x = f0.shape[2]
    handle_size_y = f0.shape[3]
    mask = 0.5 * (1.0 + mask)
    mask = mask.view(batch_size, -1, handle_size_x, handle_size_y).repeat(1, f0.shape[1], 1, 1)
    ret = (mask * f0) + (1.0 - mask) * f1
    return ret

def warp(framein, flow, handle_size=256, scale_down=(1,1)):
    handle_size_x = flow.shape[2]
    handle_size_y = flow.shape[3]
    batch_size = flow.shape[0]
    grid_x, grid_y = meshgrid(handle_size_y, handle_size_x)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()
    coor_x = grid_x + flow[:, 0, :, :] / scale_down[0]
    coor_y = grid_y + flow[:, 1, :, :] / scale_down[1]

    output = bi_interp(framein, coor_x, coor_y, 'interpolate')
    return output

def dst(flow):
    handle_size = flow.shape[2]
    batch_size = flow.shape[0]
    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()
    coor_x = grid_x + flow[:, 0, :, :]
    coor_y = grid_y + flow[:, 1, :, :]

    coor_x = coor_x.view(batch_size, 1, handle_size, handle_size)
    coor_y = coor_y.view(batch_size, 1, handle_size, handle_size)
    output = torch.cat((coor_x, coor_y), 1)
    return output

def shift(flowa, flowb, handle_size=256):
    handle_size = flowb.shape[2]
    batch_size = flowb.shape[0]
    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()

    coor_x = grid_x + flowa[:, 0, :, :]
    coor_y = grid_y + flowa[:, 1, :, :]

    output = bilinear_interp(flowb, coor_x, coor_y, 'interpolate')
    return output

def AppLoss(flowt0, flowt1, flow10, flow01):
    Tto1 = shift(flowt0, flow01) + flowt0
    Tto0 = shift(flowt1, flow10) + flowt1

    flowt1_cp = torch.FloatTensor()
    flowt1_cp.resize_(flowt1.shape).copy_(flowt1.data)
    flowt0_cp = torch.FloatTensor()
    flowt0_cp.resize_(flowt0.shape).copy_(flowt0.data)
    flowt0_cp = torch.autograd.Variable(flowt0_cp.cuda())
    flowt1_cp = torch.autograd.Variable(flowt1_cp.cuda())

    apploss = criterion(Tto1, flowt1_cp) + criterion(Tto0, flowt0_cp)
    strloss = L1Loss(-1.0 * flowt0, flowt1)
    return apploss, strloss

def L1Loss(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.abs(tensora - tensorb).sum()/eles

def L1Loss_pixel_wise(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.sum(torch.abs(tensora - tensorb), 1) / s[1]

def patchLoss(invar, targetvar, flow, h_size):
    pad = torch.nn.ReplicationPad2d(1)
    flow_padded = pad(flow)
    start_list = [0, 1, 2]
    part_flow = None
    Loss = 0
    for h in range(0, 3):
        for w in range(0, 3):
            h_start = start_list[h]
            w_start = start_list[w]

            h_end = h_size + h_start
            w_end = h_size + w_start

            part_flow = flow_padded[:,:,h_start:h_end, w_start:w_end]
            warped_frame = warp(invar, part_flow, l0_size)
            Loss += criterion(warped_frame, targetvar)
    return Loss

def patch_it(tensor):
    ret = None
    start_list = [0, 1, 2]
    h_size = tensor.shape[2]
    pad = torch.nn.ReplicationPad2d(1)
    tensor_padded = pad(tensor)
    for h in range(0, 3):
        for w in range(0, 3):
            if h == w and h != 1:
                continue
            h_start = start_list[h]
            w_start = start_list[w]

            h_end = h_size + h_start
            w_end = h_size + w_start

            part_tensor = tensor_padded[:,:,h_start:h_end, w_start:w_end]
            if ret is None:
                ret = part_tensor
            else:
                ret = torch.cat((ret, part_tensor), 1)
    return ret

def prop_refine(invar, targetvar, flow, rf_times, middle_target=None, scale_down=1):
    with torch.no_grad():
        batch_size = flow.shape[0]
        t1 = time.time()
        refine_flow = torch.zeros(flow.shape).cuda()
        pad = torch.nn.ReplicationPad2d(2)
        for i in range(0, rf_times):
            refine_flow = refine_flow * 0.0
            flow_padded = pad(flow)
            start_list = [0, 1, 2, 3, 4]
            part_flow = None
            sub_ret = None
            h_size = flow.shape[2]
            flow_record = None
            for h in range(0, 5):
                for w in range(0, 5):
                    h_start = start_list[h]
                    w_start = start_list[w]

                    h_end = h_size + h_start
                    w_end = h_size + w_start

                    part_flow = flow_padded[:,:,h_start:h_end, w_start:w_end]
                    if flow_record is None:
                        flow_record = part_flow.view(1, batch_size, 2, flow.shape[2], flow.shape[3])
                    else:
                        flow_record = torch.cat((flow_record, part_flow.view(1, batch_size, 2, flow.shape[2], flow.shape[3])
     ), 0)
                    warped_frame = warp(invar, part_flow, h_size, scale_down=scale_down)
                    sub_ret_1c = L1Loss_pixel_wise(warped_frame, targetvar).view(batch_size, 1, flow.shape[2], flow.shape[3])
                    if middle_target is not None:
                        middle_warped_frame = warp(invar, part_flow / 2.0, h_size, scale_down=scale_down)
                        sub_ret_1c += L1Loss_pixel_wise(middle_warped_frame, middle_target).view(batch_size, 1, flow.shape[2], flow.shape[3])

                    if sub_ret is  None:
                        sub_ret = sub_ret_1c
                    else:
                        sub_ret = torch.cat((sub_ret, sub_ret_1c), 1)

#                del part_flow
#                del warped_frame
#                del sub_ret_1c
#gc.collect()

            _, which = torch.min(sub_ret, 1)
            which = which.view(batch_size, 1, flow.shape[2], flow.shape[2]).repeat(1, 2, 1, 1)
            for j in range(0, 25):
                refine_flow = refine_flow + flow_record[j] * ((which == j).float())
#warped_frame = warp(invar, refine_flow, l0_size)
#            print(i, L1Loss(warped_frame, targetvar))
            flow = refine_flow
            del flow_record
            del flow_padded
            del sub_ret
            del which
            gc.collect()

        t3 = time.time()
    return flow

def one_bd_loss(flow):
    bd = torch.ones_like(flow)
    s = torch.abs(bd - flow) + torch.abs(-1.0*bd - flow)
    bd_loss = criterion(s, bd * 2.0)
    return bd_loss

def oz_bd_loss(flow):
    bd = torch.ones_like(flow)
    s = torch.abs(bd - flow) + torch.abs(-0.0*bd - flow)
    bd_loss = criterion(s, bd * 1.0)
    return bd_loss

def mix_flow(flow, refine_weight):
    batch_size = flow.shape[0]
    refine_flow = torch.zeros(flow.shape).cuda()
    pad = torch.nn.ReplicationPad2d(1)
    refine_flow = refine_flow * 0.0
    flow_padded = pad(flow)
    start_list = [0, 1, 2]
    part_flow = None
    sub_ret = None
    h_size = flow.shape[2]
    flow_record_x = None
    flow_record_y = None
    for h in range(0, 3):
        for w in range(0, 3):
            h_start = start_list[h]
            w_start = start_list[w]

            h_end = h_size + h_start
            w_end = h_size + w_start

            part_flow = flow_padded[:,:,h_start:h_end, w_start:w_end]
            if flow_record_x is None:
                flow_record_x = part_flow.view(batch_size, 2, flow.shape[2], flow.shape[3])[:,0:1]
                flow_record_y = part_flow.view(batch_size, 2, flow.shape[2], flow.shape[3])[:,1:2]
            else:
                flow_record_x = torch.cat((flow_record_x, part_flow.view(batch_size, 2, flow.shape[2], flow.shape[3])[:,0:1]), 1)
                flow_record_y = torch.cat((flow_record_y, part_flow.view(batch_size, 2, flow.shape[2], flow.shape[3])[:,1:2]), 1)

#     print(refine_weight.shape)
    refine_flow_x = flow_record_x * refine_weight
    refine_flow_y = flow_record_y * refine_weight
#     print(refine_flow_x.shape, "x")
    refine_flow_x = torch.sum(refine_flow_x, 1).view(batch_size, 1, flow.shape[2], flow.shape[2])
    refine_flow_y = torch.sum(refine_flow_y, 1).view(batch_size, 1, flow.shape[2], flow.shape[2])
#     print(refine_flow_x.shape, " ax")

    refine_flow = torch.cat((refine_flow_x, refine_flow_y), 1)
#     print(refine_flow.shape, "refine_flow")
    return refine_flow


def pad_x(x, p):
    pad = torch.nn.ReplicationPad2d(p)
    x_padded = pad(x)
    start_list = [k for k in range(0, 2*p+1)]
    part_x = None
    h_size = x.shape[2]
    w_size = x.shape[3]
    x_record = None
    batch_size = x.shape[0]
    for h in range(0, 2*p+1):
        for w in range(0, 2*p+1):
            h_start = start_list[h]
            w_start = start_list[w]

            h_end = h_size + h_start
            w_end = w_size + w_start

            part_x = x_padded[:,:,h_start:h_end, w_start:w_end]
            if x_record is None:
                x_record = part_x.view(batch_size, 3, x.shape[2], x.shape[3])
            else:
                x_record = torch.cat((x_record, part_x.view(batch_size, 3, x.shape[2], x.shape[3])), 1)
    return x_record


def No_grad(x):
    return x.detach().clone() 


def synthesis(f0, f1, k0, k1):
    ret = (f0 * k0 + f1 * k1) / 2
    return torch.cat((torch.sum(ret[:,0::3], 1).view(ret.shape[0], 1, ret.shape[2], ret.shape[3]), 
                      torch.sum(ret[:,1::3], 1).view(ret.shape[0], 1, ret.shape[2], ret.shape[3]),
                      torch.sum(ret[:,2::3], 1).view(ret.shape[0], 1, ret.shape[2], ret.shape[3])), 1)


def visual_flow(flow, batch_size):
    show_flow = None
    for ch in range(0, batch_size):
        flowimg = viz_flow(flow[ch,0,:,:].data.cpu().numpy() * 10, flow[ch,1,:,:].data.cpu().numpy() * 10)
        flowimg = np.expand_dims(flowimg, axis=0)
        if show_flow is None:
            show_flow = flowimg
        else:
            show_flow = np.concatenate((show_flow, flowimg), 0)
    return show_flow


def test(normal=1, writer=None, skip_num=1):
    
    coding.eval()
    improc.eval()
    have_f2 = True
    if writer is None:
        writer = SummaryWriter()
    
    vimeos_data_list_path = "data_list/sep_testlist.txt"

    VIMEOS_PATH_BASE = '/data/wyuxi/vimeo/vimeo_septuplet/sequences/'

    vimeos_dataset_frames = dataset.Dataset(vimeos_data_list_path, DATA_PATH_BASE=VIMEOS_PATH_BASE)

    data_list = vimeos_dataset_frames.read_data_list_file()
    seed_key = time.time()
#     random.seed(seed_key)
#     shuffle(data_list)

    batch_size = 4
    data_size = len(data_list)
    epoch_num = int(data_size / batch_size)
    croptimes = 6

    cnt = 0
    PSNR = 0
    SSIM = 0
    psnr_a = 0
    ssim_a = 0
    down_sampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    with torch.no_grad():
    #     softmax = nn.LogSoftmax(dim=1)
        t = tqdm(range(data_size//batch_size), desc='loop')
        for step in t:
            batch_idx = step % epoch_num
            epoch_idx = step / epoch_num

            batch_data_list_frames = []
            batch_data_list = data_list[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
            unit_size = 7

            for i in range(1, unit_size + 1):
                batch_data_list_frames.append(list(map(lambda e : e + '/im' + str(i) + '.png', batch_data_list)))
            img = PIL.Image.open(batch_data_list_frames[0][0])

            size_h, size_w = img.size
            min_len = min(size_h, size_w)
            rate = min_len / (handle_size + 120)
            ori_w = size_h #int(np.floor(size_h / rate))#720  #1080
            ori_h = size_w #int(np.floor(size_w / rate))#1280 #1920
            # Load batch data.
            batch_data_frames = []
            for i in range(0, unit_size):
                batch_data_frames.append(
                    torch.FloatTensor(np.array(
                            [numpy.rollaxis(numpy.asarray(
                                PIL.Image.open(line).resize((ori_w,ori_h)))[:,:,::-1], 2, 0).astype(numpy.float32) 
                             for line in batch_data_list_frames[i]]) / 255.0))

            in_frames = batch_data_frames[0] 
            for i in range(1, unit_size):
                in_frames = torch.cat((in_frames, batch_data_frames[i]), 1)#[:,:,crop_idx_x:crop_idx_x+handle_size, crop_idx_y:crop_idx_y+handle_size]
            in_frames = in_frames.cuda(devices[0])

            invar_sr = torch.autograd.Variable(in_frames)
            invar = up_sampling(up_sampling(down_sampling(down_sampling(invar_sr))))
    #             input_var = torch.autograd.Variable(torch.cat((invar[:,0:3], invar[:,6:9]), 1))
    #             target_var = torch.autograd.Variable(invar_sr[:,3:6])
            l0_size = handle_size

            loss = 0
            inter_loss = 0
            tvl = 0
            improc_loss = 0
            output_cat = None

            pair_index_list = [(0, 6), (1, 5), (2, 4)]
            tar_i = 3
            f05_sr = invar_sr[:, tar_i*3:tar_i*3+3]
            input_var = torch.cat((invar[:,0:3], invar[:,6*3:6*3+3]), 1)
            target_var = invar_sr[:, tar_i*3:tar_i*3+3]

            for pair_index in pair_index_list:
                f0_i = pair_index[0]
                f1_i = pair_index[1]
                f0 = invar[:,f0_i*3:f0_i*3+3]
                f1 = invar[:,f1_i*3:f1_i*3+3]
    #             f0 = pad_x(input_var[:,0:3], 2)
    #             f1 = pad_x(input_var[:,3:6], 2)
    #             f05 = pad_x(target_var, 2)

                input_var = torch.cat((f0, f1), 1)
                out = coding(input_var)

                flowt0_r = out[:,0:2,:,:]
                flowt1_r = out[:,2:4,:,:]
                mask = out[:,4:5,:,:]
                flowt1 = 0.5 * flowt1_r - 0.5 * flowt0_r
                flowt0 = -flowt1 

                warped_framet_f0 = warp(f0, flowt0, l0_size, scale_down=(1,1))
                warped_framet_f1 = warp(f1, flowt1, l0_size, scale_down=(1,1))

                output = blending(warped_framet_f0, warped_framet_f1, mask)
                if output_cat is None:
                    output_cat = output
                else:
                    output_cat = torch.cat((output_cat, output), 1)

                ## loss term 
                tvl += tvLoss(flowt0)+ tvLoss(flowt1)
                inter_loss += criterion(output, f05_sr)

            final_output = improc(output_cat) 
            improc_loss += criterion(final_output, f05_sr)

            output = final_output

            for img in range(0, batch_size):
                store_base = '/mnt/fin3/wyuxi/super-inter-test-nbn/'
                PIL.Image.fromarray((output[img].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(store_base + str(step*batch_size+img) + 'im4-si.png')         
                out = (final_output[img].clamp(0.0, 1.0).data.cpu().numpy() * 255.0).astype(numpy.uint8)
                tar = (target_var[img].data.cpu().numpy()*255.0).astype(numpy.uint8)
                tar = img_as_float(np.array(tar))
                out = img_as_float(np.array(out))
                psnr_a += 10*np.log10(1.0/np.square(np.subtract(np.array(out), np.array(tar))).mean())
                tar = tar.transpose(1, 2, 0)
                out = out.transpose(1, 2, 0)
                ssim_a += ssim(tar, out, data_range=out.max() - out.min(), multichannel=True)

            frame2 = (numpy.rollaxis(target_var[0].clamp(0.0, 1.0).cpu().detach().numpy(), 0, 3)[:,:,::-1] * 255.0).astype(numpy.uint8)
            out = (numpy.rollaxis(output[0].clamp(0.0, 1.0).cpu().detach().numpy(), 0, 3)[:,:,::-1] * 255.0).astype(numpy.uint8)
    #         out = PIL.Image.fromarray(out).resize(ori_size)
    #         frame1 = frame1.resize(ori_size)
            cnt += 1
    #     PSNR += 10*np.log10(255.0*255.0/(np.sum(np.square(out-Tar))/(batch_size*3*256*256)))
#             print("Overall PSNR: %f db" % (psnr_a/(cnt*batch_size)))
#             print("Overall SSIM: %f db" % (ssim_a/(cnt*batch_size)))
            t.set_description('psnr: %g ssim: %g' % ((psnr_a/(cnt*batch_size)), (ssim_a/(cnt*batch_size))))
            if have_f2 == True:
                tar = img_as_float(np.array(frame2))
                ou = img_as_float(np.array(out))
                PSNR += 10*np.log10(1.0/np.square(np.subtract(np.array(ou), np.array(tar))).mean())
                SSIM += ssim(tar, ou, data_range=ou.max() - ou.min(), multichannel=True)
            if cnt % 100 == 0: 
                show_flow = visual_flow(flowt0, batch_size)
                writer.add_image('Test_Image/flow', show_f(vutils.make_grid(torch.from_numpy(show_flow).permute(0,3,1,2), normalize=False, scale_each=True)), real_step + cnt)
                writer.add_image('Test_Image/in1', show(vutils.make_grid(input_var[:, 0:3].cpu(), normalize=True, scale_each=True)), real_step + cnt)
                writer.add_image('Test_Image/in2', show(vutils.make_grid(input_var[:, 3:6].cpu(), normalize=True, scale_each=True)),real_step + cnt)
                writer.add_image('Test_Image/out_a', show(vutils.make_grid(final_output.data.cpu(), normalize=True, scale_each=True)), real_step + cnt)
                writer.add_image('Test_Image/target', show(vutils.make_grid(target_var.data.cpu(), normalize=True, scale_each=True)), real_step + cnt)
        
    print("Overall ssim:", (ssim_a/(cnt*batch_size)))
    print("Overall PSNR: %f db" % (psnr_a/(cnt*batch_size)))
    print("cnt", cnt)
    if skip_num > 1:
        writer.add_scalar('ssim_test_f', (SSIM / cnt), real_step)
        writer.add_scalar('psnr_test_f', (PSNR / cnt), real_step)
    else:
        writer.add_scalar('ssim_test', (ssim_a/(cnt*batch_size)), real_step)
        writer.add_scalar('psnr_test', (psnr_a/(cnt*batch_size)), real_step)
    coding.train()
    return (psnr_a/(cnt*batch_size))



if args.mode == "train":
# Load data
    vimeos_data_list_path = "data_list/sep_trainlist.txt"

    VIMEOS_PATH_BASE = '/data/wyuxi/vimeo/vimeo_septuplet/sequences/'

    vimeos_dataset_frames = dataset.Dataset(vimeos_data_list_path, DATA_PATH_BASE=VIMEOS_PATH_BASE)

    data_list = vimeos_dataset_frames.read_data_list_file()
    seed_key = time.time()
    random.seed(seed_key)
    shuffle(data_list)


    data_size = len(data_list)
    epoch_num = int(data_size / batch_size)
    croptimes = 1
    writer = SummaryWriter()

    down_sampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
#     softmax = nn.LogSoftmax(dim=1)

    for step in range(0, Max_steps):
        batch_idx = step % epoch_num
        epoch_idx = step / epoch_num

        if batch_idx == 0 and step != 0:
        # Shuffle data at each epoch.
            seed_key = time.time()
            random.seed(seed_key)
            shuffle(data_list)
            print("Shuffled data!")
            print('Epoch Number: %d' % int(real_step / epoch_num))

        batch_data_list_frames = []
        batch_data_list = data_list[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
        unit_size = 7
        for i in range(1, unit_size + 1):
            batch_data_list_frames.append(list(map(lambda e : e + '/im' + str(i) + '.png', batch_data_list)))
        
        img = PIL.Image.open(batch_data_list_frames[0][0])
        size_h, size_w = img.size
        min_len = min(size_h, size_w)
        rate = min_len / (handle_size + 120)
        ori_w = size_h #int(np.floor(size_h / rate))#720  #1080
        ori_h = size_w #int(np.floor(size_w / rate))#1280 #1920

        # Load batch data.
        batch_data_frames = []
        for i in range(0, unit_size):
            batch_data_frames.append(
                torch.FloatTensor(np.array(
                        [numpy.rollaxis(numpy.asarray(
                            PIL.Image.open(line).resize((ori_w,ori_h)))[:,:,::-1], 2, 0).astype(numpy.float32) 
                         for line in batch_data_list_frames[i]]) / 255.0))
        
        for keep_crop in range(0, croptimes):
            real_step += 1

            if real_step % (epoch_num * 5) == epoch_num * 5 - 1 and lr >= 1e-5:
                print("change lr")
                lr = lr / 2
                optimizer = optim.Adam(list(coding.parameters())+list(improc.parameters()), lr=lr, weight_decay=1e-5)

#             crop_idx_x = np.random.randint(ori_h - handle_size)
#             crop_idx_y = np.random.randint(ori_w - handle_size)
            in_frames = batch_data_frames[0] 
            for i in range(1, unit_size):
                in_frames = torch.cat((in_frames, batch_data_frames[i]), 1)#[:,:,crop_idx_x:crop_idx_x+handle_size, crop_idx_y:crop_idx_y+handle_size]
            in_frames = in_frames.cuda(devices[0])

            invar_sr = torch.autograd.Variable(in_frames)
            invar = up_sampling(up_sampling(down_sampling(down_sampling(invar_sr))))
#             input_var = torch.autograd.Variable(torch.cat((invar[:,0:3], invar[:,6:9]), 1))
#             target_var = torch.autograd.Variable(invar_sr[:,3:6])
            l0_size = handle_size

            optimizer.zero_grad()
            loss = 0
            inter_loss = 0
            tvl = 0
            improc_loss = 0
            output_cat = None

            def No_grad(x):
                return x.detach().clone() 
            pair_index_list = [(0, 6), (1, 5), (2, 4)]
            tar_i = 3
            f05_sr = invar_sr[:, tar_i*3:tar_i*3+3]
            input_var = torch.cat((invar[:,0:3], invar[:,6*3:6*3+3]), 1)
            target_var = invar_sr[:, tar_i*3:tar_i*3+3]
            
            for pair_index in pair_index_list:
                f0_i = pair_index[0]
                f1_i = pair_index[1]
                f0 = invar[:,f0_i*3:f0_i*3+3]
                f1 = invar[:,f1_i*3:f1_i*3+3]
    #             f0 = pad_x(input_var[:,0:3], 2)
    #             f1 = pad_x(input_var[:,3:6], 2)
    #             f05 = pad_x(target_var, 2)
    
                input_var = torch.cat((f0, f1), 1)
                out = coding(input_var)

                flowt0_r = out[:,0:2,:,:]
                flowt1_r = out[:,2:4,:,:]
                mask = out[:,4:5,:,:]
                flowt1 = 0.5 * flowt1_r - 0.5 * flowt0_r
                flowt0 = -flowt1 

                warped_framet_f0 = warp(f0, flowt0, l0_size, scale_down=(1,1))
                warped_framet_f1 = warp(f1, flowt1, l0_size, scale_down=(1,1))

                output = blending(warped_framet_f0, warped_framet_f1, mask)
                if output_cat is None:
                    output_cat = output
                else:
                    output_cat = torch.cat((output_cat, output), 1)

                ## loss term 
                tvl += tvLoss(flowt0)+ tvLoss(flowt1)
                inter_loss += criterion(output, f05_sr)

            final_output = improc(output_cat) 
            improc_loss += criterion(final_output, f05_sr)
            
            loss +=  1 * inter_loss + 10 * improc_loss #+ 2 * tvl
                                 
            loss.backward()
            optimizer.step()

            
            psnr_a = 0
            for img in range(0, batch_size):
                out = (final_output[img].clamp(0.0, 1.0).data.cpu().numpy() * 255.0).astype(numpy.uint8)
                tar = (target_var[img].data.cpu().numpy()*255.0).astype(numpy.uint8)
                tar = img_as_float(np.array(tar))
                out = img_as_float(np.array(out))
                psnr_a += 10*np.log10(1.0/np.square(np.subtract(np.array(out), np.array(tar))).mean())
            psnr_a = psnr_a / batch_size

#             psnr_b = 0
#             for img in range(0, batch_size):
#                 out = (pred_tar[img].clamp(0.0, 1.0).data.cpu().numpy() * 255.0).astype(numpy.uint8)
#                 tar = (target_var[img].data.cpu().numpy()*255.0).astype(numpy.uint8)
#                 psnr_b += 10*np.log10(255.0*255.0/np.square(np.subtract(out, tar)).mean())
#             psnr_b = psnr_b / batch_size

            ## Log information
            if real_step % 10 == 0:
                writer.add_scalar('loss', loss.data, real_step)
                writer.add_scalar('tvl', tvl.data, real_step)
                writer.add_scalar('psnr_a', psnr_a, real_step)
                writer.add_scalar('inter_loss', inter_loss.data, real_step)
                writer.add_scalar('improc_loss', improc_loss.data, real_step)

            if real_step % 3000 == 0 and real_step > 10:
                print("Test!")
                now_psnr = test(0, writer, 1)
                print("Test end.")
                
                if now_psnr > best_psnr:
                    best_psnr = now_psnr
                    torch.save(coding.state_dict(), 'best_coding.pkl')
                    torch.save(improc.state_dict(), 'best_improc.pkl')
                    torch.save(best_psnr, 'best_psnr.pkl')
                    
                coding.train()
                improc.train()
                
            if real_step % 300 == 0 and real_step > 10:
#                 print("little Test!")
#                 test(0, writer, 50)
#                 print("Test end.")
                
                    
                torch.save(coding.state_dict(), 'coding.pkl')
                torch.save(improc.state_dict(), 'improc.pkl')
                torch.save(best_psnr, 'best_psnr.pkl')
                coding.train()
                improc.train()


            if real_step % show_img_gap == 0:
                print("Loss at step %d epoch %f batch %d: %f, lr=%.15f best_psnr%.4f" % (real_step, real_step / (epoch_num*1.0), batch_idx, loss.data, lr, best_psnr))
                if real_step % show_img_gap == 0:
                    ## vis flow
                    def visual_flow(flow, batch_size):
                        show_flow = None
                        for ch in range(0, batch_size):
                            flowimg = viz_flow(flow[ch,0,:,:].data.cpu().numpy() * 10, flow[ch,1,:,:].data.cpu().numpy() * 10)
                            flowimg = np.expand_dims(flowimg, axis=0)
                            if show_flow is None:
                                show_flow = flowimg
                            else:
                                show_flow = np.concatenate((show_flow, flowimg), 0)
                        return show_flow
                    show_flow = visual_flow(flowt0, batch_size)

                    writer.add_image('Image/flow', show_f(vutils.make_grid(torch.from_numpy(show_flow).permute(0,3,1,2), normalize=False, scale_each=True)), real_step)
                    writer.add_image('Image/in1', show(vutils.make_grid(invar[:, 0:3].cpu(), normalize=True, scale_each=True)), real_step)
                    writer.add_image('Image/in1_sr', show(vutils.make_grid(invar_sr[:, 0:3].cpu(), normalize=True, scale_each=True)), real_step)
                    writer.add_image('Image/in2', show(vutils.make_grid(invar[:, 6*3:6*3+3].cpu(), normalize=True, scale_each=True)), real_step)
                    writer.add_image('Image/out_a', show(vutils.make_grid(final_output.data.cpu(), normalize=True, scale_each=True)), real_step)
                    writer.add_image('Image/target', show(vutils.make_grid(target_var.data.cpu(), normalize=True, scale_each=True)), real_step)

if args.mode == "test":
    test(1, None)
