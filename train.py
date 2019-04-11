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
from torchvision.models.vgg import vgg16


from utils.image_utils import imwrite
from utils.flow_utils import bilinear_interp_large, bilinear_interp, bi_interp_3d, bi_interp, to_there, forward_warp,meshgrid_3d, meshgrid, prop_refine_rd, prop_refine_which
from utils.vis_utils import viz_flow
from utils.loss_utils import TVLoss
import json
import adabound

from model import FlowNet, SRNet, SharpNet_C2_rf

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # change this if you have a multiple graphics cards and you want to utilize them
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
devices = [0]
torch.cuda.set_device(devices[0])


with open('settings.json') as f:
    settings = json.load(f)

lr_sr = settings['lr_sr']
lr_flow = settings['lr_flow']
batch_size = settings['batch_size']

handle_size = 256
Max_steps = 10000000
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


# FNet = FlowNet(6, 2)
# FNet = nn.DataParallel(FNet, device_ids=devices).cuda()

# SR = SRNet(51)
# SR = nn.DataParallel(SR, device_ids=devices).cuda()

FNet = SharpNet_C2_rf(6, 2)
FNet = nn.DataParallel(FNet, device_ids=devices).cuda()

PNet = FlowNet(11, 25)
PNet = nn.DataParallel(PNet, device_ids=devices).cuda()

RNet = FlowNet(11, 2)
RNet = nn.DataParallel(RNet, device_ids=devices).cuda()


real_step = 0
FNet.train()
PNet.train()
RNet.train()

vgg = vgg16(pretrained=True).cuda()
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
mse_loss = nn.MSELoss()
for param in loss_network.parameters():
    param.requires_grad = False

def perception_loss(out, tar):
    a = loss_network(out)
    b = loss_network(tar)
    per_loss = mse_loss(a, b)
    return per_loss

if args.pretrained == "True":
    print("Load trained Net...")
    best_psnr = torch.load('best_psnr.pkl')
    FNet.load_state_dict(torch.load('FNet.pkl'))
    RNet.load_state_dict(torch.load('RNet.pkl'))
    PNet.load_state_dict(torch.load('PNet.pkl'))
    print("Loaded")

    
    
reg = 10
cross_entory = torch.nn.CrossEntropyLoss()
criterion = torch.nn.L1Loss()
tvLoss = TVLoss()

# optimizer = optim.Adam(
optimizer = optim.Adamax(
[
    {"params": FNet.parameters(), 'lr':lr_flow},
],
lr=1e-5
)

def patch_it(tensor):
    ret = None
    start_list = [0, 1, 2, 3, 4]
    h_size = tensor.shape[2]
    w_size = tensor.shape[3]
    pad = torch.nn.ReplicationPad2d(2)
    tensor_padded = pad(tensor)
    for h in range(0, 5):
        for w in range(0, 5):
#             if h == w and h != 1:
#                 continue
            h_start = start_list[h]
            w_start = start_list[w]

            h_end = h_size + h_start
            w_end = w_size + w_start

            part_tensor = tensor_padded[:,:,h_start:h_end, w_start:w_end]
            if ret is None:
                ret = part_tensor
            else:
                ret = torch.cat((ret, part_tensor), 1)
    return ret

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

#     output = bi_interp(framein, coor_x, coor_y, 'interpolate')
    output = bilinear_interp(framein, coor_x, coor_y, 'interpolate')
    return output
def warp_block(framein, flow, handle_size=256, scale_down=(1,1)):
    handle_size_x = flow.shape[2]
    handle_size_y = flow.shape[3]
    batch_size = flow.shape[0]
    grid_x, grid_y = meshgrid(handle_size_y, handle_size_x)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()
    coor_x = grid_x + flow[:, 0, :, :] / scale_down[0]
    coor_y = grid_y + flow[:, 1, :, :] / scale_down[1]

    output = bilinear_interp_large(framein, coor_x, coor_y, 'interpolate')
    return output

def cube_warp(cube_in, flow_cube, handle_size=256, scale_down=(1,1)):
    cube_in = cube_in.view(cube_in.shape[0], 16, 3, cube_in.shape[2], cube_in.shape[3]).permute(0,2,1,3,4)
#     flow_cube = flow_cube.view(flow_cube.shape[0], 1, flow_cube.shape[1], flow_cube.shape[2], flow_cube.shape[3])
    handle_size_x = flow_cube.shape[2]
    handle_size_y = flow_cube.shape[3]
    handle_size_z = flow_cube.shape[1]
    batch_size = flow_cube.shape[0]
    grid_y, grid_z, grid_x = meshgrid_3d(handle_size_y, handle_size_x, handle_size_z)
#     print(grid_x[1,:,:], 'grid_x')
#     print(grid_y[1,:,:], 'grid_y')
#     print(grid_z[:,1,1], 'grid_z')
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1, 1]))).cuda()
    grid_z = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_z, [batch_size, 1, 1, 1]))).cuda()
    coor_x = grid_x + flow_cube[:, :, :, :, 0] / scale_down[0]
    coor_y = grid_y + flow_cube[:, :, :, :, 1] / scale_down[1]
    coor_z = grid_z + flow_cube[:, :, :, :, 2] / 1 
#     print(coor_x[0,0:2,0:3,0:3],'x')
#     print(coor_y[0,0:2,0:3,0:3],'y')
#     print(coor_z[0,0:2,0:3,0:3],'z')
#   cube_in shape 2 3 32 64 112
#     output = bi_interp_3d(cube_in, coor_x, coor_y, coor_z, 'interpolate')
    coor = torch.stack([ coor_x, coor_y,coor_z], dim=4)
    output = torch.nn.functional.grid_sample(
            cube_in,
            coor,
            padding_mode='border')
#     print(coor.shape, coor[0,0:2,0:2,0:2,0:3])
#     print(output.shape, output[0,0,0:2,0:2,0:2])
#     print(cube_in.shape, cube_in[0,0,0:2,0:2,0:2])
    return output

def L1Loss(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.abs(tensora - tensorb).sum()/eles

def L1Loss_pixel_wise(tensora, tensorb):
    s = tensora.size()
    eles = s[0] * s[1] * s[2] * s[3]
    return torch.sum(torch.abs(tensora - tensorb), 1) / s[1]

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

def No_grad(x):
    return x.detach().clone() 

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

def get_psnr_ssim(out, tar):
    out = (out.clamp(0.0, 1.0).data.cpu().numpy() * 255.0).astype(numpy.uint8)
    tar = (tar.data.cpu().numpy()*255.0).astype(numpy.uint8)
    tar = img_as_float(np.array(tar))
    out = img_as_float(np.array(out))
    eps = 1e-7
    denominator = np.square(np.subtract(np.array(out), np.array(tar))).mean()
    if denominator == 0:
        print("warning manually: got a 0 psnr")
    denominator = denominator + eps
    psnr = 10*np.log10(1.0 / denominator)
    tar = tar.transpose(1, 2, 0)
    out = out.transpose(1, 2, 0)
    ssim = get_ssim(tar, out, data_range=out.max() - out.min(), multichannel=True)
    return psnr, ssim

def test(normal=1, writer=None, skip_num=1):
    
    FNet.eval()
    SR.eval()
    have_f2 = True
    if writer is None:
        writer = SummaryWriter()
    
    vimeos_data_list_path = "data_list/sep_testlist.txt"

    VIMEOS_PATH_BASE_SR = '/home/ubuntu/data/vimeo/vimeo_septuplet/sequences/'
    VIMEOS_PATH_BASE = '/home/ubuntu/data/vimeo/vimeo_super_resolution_test/low_resolution/'

    vimeos_dataset_frames = dataset.Dataset(vimeos_data_list_path, DATA_PATH_BASE=VIMEOS_PATH_BASE)
    vimeos_dataset_frames_sr = dataset.Dataset(vimeos_data_list_path, DATA_PATH_BASE=VIMEOS_PATH_BASE_SR)

    data_list = vimeos_dataset_frames.read_data_list_file()
    data_list_sr = vimeos_dataset_frames_sr.read_data_list_file()
    
    ## no need in the testing phase
#     seed_key = time.time()
#     random.seed(seed_key)
#     shuffle(data_list)
#     shuffle(data_list_sr)

    batch_size = 16
    data_size = len(data_list)
    epoch_num = int(data_size / batch_size)
    croptimes = 6

    cnt = 0
    PSNR = 0
    SSIM = 0
    psnr_a = 0
    psnr_given = 0
    psnr_list = [0 for i in range(0 , 7)]
    psnr_middle = 0
    ssim_a = 0
    down_sampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
#     up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    
    ## TESTING PHASE
    with torch.no_grad():
        t = tqdm(range(data_size//batch_size), desc='loop')
        for step in t:
            ## if little test then skip
            if step % skip_num != 0:
                continue
                
            batch_idx = step % epoch_num
            epoch_idx = step / epoch_num

            ## DATA READING
            batch_data_list_frames = []
            batch_data_list_frames_sr = []
            batch_data_list = data_list[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
            batch_data_list_sr = data_list_sr[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
            unit_size = 7
            
            for i in range(1, unit_size + 1):
                batch_data_list_frames.append(list(map(lambda e : e + '/im' + str(i) + '.png', batch_data_list)))
                batch_data_list_frames_sr.append(list(map(lambda e : e + '/im' + str(i) + '.png', batch_data_list_sr)))
            img = PIL.Image.open(batch_data_list_frames[0][0])
            img = PIL.Image.open(batch_data_list_frames_sr[0][0])

            size_h, size_w = img.size
            min_len = min(size_h, size_w)
            rate = min_len / (handle_size + 120)
            ori_w = size_h #int(np.floor(size_h / rate))#720  #1080
            ori_h = size_w #int(np.floor(size_w / rate))#1280 #1920
            
            # Load batch data.
            batch_data_frames = []
            batch_data_frames_sr = []
            for i in range(0, unit_size):
                batch_data_frames.append(
                    torch.FloatTensor(np.array(
                            [numpy.rollaxis(numpy.asarray(
                                PIL.Image.open(line))[:,:,::-1], 2, 0).astype(numpy.float32) 
                             for line in batch_data_list_frames[i]]) / 255.0))
                batch_data_frames_sr.append(
                    torch.FloatTensor(np.array(
                            [numpy.rollaxis(numpy.asarray(
                                PIL.Image.open(line))[:,:,::-1], 2, 0).astype(numpy.float32) 
                             for line in batch_data_list_frames_sr[i]]) / 255.0))

            in_frames = batch_data_frames[0] 
            in_frames_sr = batch_data_frames_sr[0] 
            for i in range(1, unit_size):
                in_frames = torch.cat((in_frames, batch_data_frames[i]), 1)#[:,:,crop_idx_x:crop_idx_x+handle_size, crop_idx_y:crop_idx_y+handle_size]
                in_frames_sr = torch.cat((in_frames_sr, batch_data_frames_sr[i]), 1)
                
            in_frames = in_frames.cuda(devices[0])
            in_frames_sr = in_frames_sr.cuda(devices[0])

            invar_lr = torch.autograd.Variable(in_frames)
            invar_sr = torch.autograd.Variable(in_frames_sr)
#             invar = torch.nn.functional.interpolate(torch.nn.functional.interpolate(down_sampling(down_sampling(invar_sr)),
            invar = torch.nn.functional.interpolate(torch.nn.functional.interpolate(invar_lr,
                                                   scale_factor=2, mode='bilinear',align_corners=False), 
                                                   scale_factor=2, mode='bilinear',align_corners=False)
            l0_size = handle_size
            loss = 0
            inter_loss = 0
            tvl = 0
            improc_loss = 0
            output_cat = None
            no_middle = False

            ## FEED MODLE
            if normal == 1:
                pair_index_list = [(0, 2), (2, 4), (4, 6)]
                ind_a = [0, 1, 2, 3, 4, 5, 6]
            elif normal == 2 :
                no_middle = True
                pair_index_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
                ind_a = [0, 1, 2, 3, 4, 5, 6]

#             if rand_int  == 0:
#                 unit_size = 7
#                 pair_index_list = [(0, 2), (2, 4), (4, 6)]
#                 ind_a = [0, 1, 2, 3, 4, 5, 6]
#             elif rand_int == 1:
#                 unit_size = 5
#                 pair_index_list = [(1, 3), (3, 5)]
#                 ind_a = [1, 2, 3, 4, 5]
#             elif rand_int == 2:
#                 unit_size = 7
#                 no_middle = True 
#                 pair_index_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
#                 ind_a = [0, 1, 2, 3, 4, 5, 6]
                
            ## for display
            input_var = torch.cat((invar[:,0:3], invar[:,6*3:6*3+3]), 1)
#             target_var = invar_sr[:, tar_i*3:tar_i*3+3]
            
            SR_ret = None
            SR_list = []
            left_list = []
            k = torch.zeros([48, 3, 4, 4])
            for dim in range(0, 16):
                x = int(dim//4)
                y = int(dim % 4)
                k[dim*3, 0, x, y] = 1
                k[dim*3+1, 1, x, y] = 1 
                k[dim*3+2, 2, x, y] = 1 
            kernel = torch.FloatTensor(k)#.unsqueeze(0).unsqueeze(0)
            weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
            
            for pair_index in pair_index_list:
                f0_i = pair_index[0]
                f1_i = pair_index[1]
                tar_i = (f0_i + f1_i) // 2
                
                f0_lr = invar_lr[:,f0_i*3:f0_i*3+3]
                f1_lr = invar_lr[:,f1_i*3:f1_i*3+3]
                f0_enl = invar[:,f0_i*3:f0_i*3+3]
                f1_enl = invar[:,f1_i*3:f1_i*3+3]
                f0_sr = invar_sr[:,f0_i*3:f0_i*3+3]
                f1_sr = invar_sr[:,f1_i*3:f1_i*3+3]
    
                input_var_lr = torch.cat((f0_lr, f1_lr), 1)
            
                # for given frame
                flow10 = FNet(input_var_lr)[:,2:4]
                
                # process slide flow
                flow10_enl = torch.nn.functional.interpolate(flow10, scale_factor=4, mode='bilinear',align_corners=False) 
                
                # open recurrent
                if SR_ret is not None:
                    f0_enl = SR_ret
                    
                # process given frame
                warped_frame1_f0 = warp(f0_enl, flow10_enl, l0_size, scale_down=(1,1))
                
                warped_frame1_f0_lr = warp(f0_lr, flow10, l0_size, scale_down=(1,1))
                
                warped_frame1_f0_4_lr = torch.nn.functional.conv2d(warped_frame1_f0, weight, bias=None, stride=4, padding=0)
                
                if SR_ret is None:
                    f0_food = torch.cat((f0_lr, torch.zeros_like(warped_frame1_f0_4_lr)), 1)
                f1_food = torch.cat((f1_lr, warped_frame1_f0_4_lr), 1)
                
                if SR_ret is None:
                    f0_ret = SR(f0_food)
                f1_ret = SR(f1_food)
                
                if SR_ret is None:
                    SR_list.append(f0_ret)
                    
                SR_ret = f1_ret
                SR_list.append(f1_ret)
                
                left_list.append(warped_frame1_f0)
                left_list.append(warped_frame1_f0_lr)

            ## COMPUTER PSNR SSIM and SAVE RESULT
            for ind in range(0, 7):
                final_output = SR_list[ind]
                target_var = invar_sr[:,ind*3:ind*3+3]
                for img in range(0, batch_size):
                   # if skip_num == 1:
                   #     ## save result picture
                   #     store_base = './super-inter-test-nbn/'
                   #     PIL.Image.fromarray((output[img].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(store_base + str(step*batch_size+img) + 'im4-si.png')         

                    ## get psnr ssim
                    psnr_tmp, ssim_tmp = get_psnr_ssim(final_output[img], target_var[img])
                    psnr_a += psnr_tmp
                    ssim_a += ssim_tmp
                    psnr_list[ind] += psnr_tmp
                    if ind % 2 == 1:
                        psnr_middle += psnr_tmp
                    else:
                        psnr_given += psnr_tmp

            ## update progress bar
            cnt += 1
            t.set_description('psnr: %g ssim: %g' % ((psnr_a/(cnt*batch_size*7)), (ssim_a/(cnt*batch_size*7))))
            
            ## show sample result on tensorboard
            if cnt % 50 == 0 or step == 0: 
                show_flow = visual_flow(flow10, batch_size)
                writer.add_image('Test_Image/flow', show_f(vutils.make_grid(torch.from_numpy(show_flow).permute(0,3,1,2), normalize=False, scale_each=True)), real_step + cnt)
                writer.add_image('Test_Image/in1', show(vutils.make_grid(input_var[:, 0:3].cpu(), normalize=False, scale_each=True)), real_step + cnt)
                writer.add_image('Test_Image/in2', show(vutils.make_grid(input_var[:, 3:6].cpu(), normalize=False, scale_each=True)),real_step + cnt)
                prefix = 'Test_Image'
                if normal == 2:
                    prefix = 'SR_image'
                for ind in range(0, 7):
                    writer.add_image(prefix + '/out_a'+str(ind), show(vutils.make_grid(SR_list[ind].clamp(0.0, 1.0).data.cpu(), normalize=False, scale_each=True)), real_step + cnt)
                    writer.add_image(prefix + '/target'+str(ind), show(vutils.make_grid(invar_sr[:, ind*3:ind*3+3].clamp(0.0, 1.0).data.cpu(), normalize=False, scale_each=True)), real_step + cnt)
        
    ## show result in terminal
    for ind in range(0, 7):
        psnr_list[ind] /= (cnt * batch_size)
        if skip_num == 1:
            writer.add_scalar('psnr_'+ str(normal) + '_test_' + str(ind), psnr_list[ind], real_step)
        else:
            writer.add_scalar('psnr_'+ str(normal) + '_' + str(ind), psnr_list[ind], real_step)
    
    final_ssim = ssim_a / (cnt*batch_size*7) 
    final_psnr = psnr_a / (cnt*batch_size*7)
    final_psnr_given = psnr_given / (cnt*batch_size*4)
    final_psnr_middle = psnr_middle / (cnt*batch_size*3)
    print("Overall ssim:", final_ssim)
    print("Overall PSNR: %f db, given %f db, middle %f db" % (final_psnr, final_psnr_given, final_psnr_middle))
    print("cnt", cnt)
    
    ## record score in tensorboard
    if skip_num > 1:
        if normal == 1:
            writer.add_scalar('ssim_test_f', final_ssim, real_step)
            writer.add_scalar('psnr_test_f', final_psnr, real_step)
            writer.add_scalar('psnr_test_given_f', final_psnr_given, real_step)
            writer.add_scalar('psnr_test_middle_f', final_psnr_middle, real_step)
        elif normal == 2 :
            writer.add_scalar('ssim_sr_f', final_ssim, real_step)
            writer.add_scalar('psnr_sr_f', final_psnr, real_step)
            
    else:
        if normal == 1:
            writer.add_scalar('ssim_test', final_ssim, real_step)
            writer.add_scalar('psnr_test', final_psnr, real_step)
            writer.add_scalar('psnr_test_given', final_psnr_given, real_step)
            writer.add_scalar('psnr_test_middle', final_psnr_middle, real_step)
        elif normal == 2 :
            writer.add_scalar('ssim_sr', final_ssim, real_step)
            writer.add_scalar('psnr_sr', final_psnr, real_step)
        
    return final_psnr, final_ssim 



if args.mode == "train":
# Load data
    vimeos_data_list_path = "data_list/sep_trainlist.txt"

    VIMEOS_PATH_BASE = '/home/ubuntu/data/vimeo/vimeo_septuplet/sequences/'

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

    for step in range(0, Max_steps):
        data_size = len(data_list)
        epoch_num = int(data_size / batch_size)
        batch_idx = step % epoch_num
        epoch_idx = step / epoch_num

        if batch_idx == 0 and step != 0:
        # Shuffle data at each epoch.
            seed_key = time.time()
            random.seed(seed_key)
            shuffle(data_list)
            print("Shuffled data!")
            print('Epoch Number: %d' % int(real_step / epoch_num))
            
        with open('settings.json') as f:
            settings = json.load(f)
        new_lr_sr = settings['lr_sr']
        new_lr_flow = settings['lr_flow']
        s_test_gap = settings['s_gap']
        b_test_gap = settings['b_gap']
        scalar_gap = settings['scalar_gap']
        show_img_gap = settings['show_img_gap']
        reload = settings['reload']
        sim_on = settings['sim_on']
        tvl_w = settings['tvl_w']
        srl_w = settings['srl_w']
        bdl_w = settings['bdl_w']
        pl_w = settings['pl_w']
        loss_deg = settings['loss_deg']
        batch_size = settings['batch_size']

        if reload == 'True':
            print("Reload best Net...")
            FNet.load_state_dict(torch.load('best_FNet.pkl'))
            SR.load_state_dict(torch.load('best_SR.pkl'))
            best_psnr = torch.load('best_psnr.pkl')
            print("Reloaded")


            ## write reload to false
            with open('settings.json', 'w') as f:
                settings['reload'] = "False"
                json.dump(settings, f)


        ## IF I changed LR in settings file then change the lr of optimizer
        if new_lr_sr != lr_sr or new_lr_flow != lr_flow:
            print("change lr_sr")
            lr_sr = new_lr_sr
            lr_flow = new_lr_flow
            optimizer = optim.Adamax(
            [
                {"params": FNet.parameters(), 'lr':lr_flow},
            ],
            lr=1e-5
            )

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

            in_frames = batch_data_frames[0] 
            for i in range(1, unit_size):
                in_frames = torch.cat((in_frames, batch_data_frames[i]), 1)#[:,:,crop_idx_x:crop_idx_x+handle_size, crop_idx_y:crop_idx_y+handle_size]
            in_frames = in_frames.cuda(devices[0])

            invar_sr = torch.autograd.Variable(in_frames)
            invar_lr = torch.nn.functional.interpolate(torch.nn.functional.interpolate(invar_sr,
                                                   scale_factor=0.5, mode='bilinear',align_corners=False), 
                                                   scale_factor=0.5, mode='bilinear',align_corners=False)
            invar = torch.nn.functional.interpolate(torch.nn.functional.interpolate(invar_lr,
                                                   scale_factor=2, mode='bilinear',align_corners=False), 
                                                   scale_factor=2, mode='bilinear',align_corners=False)
            l0_size = handle_size

            output_cat = None
            no_middle = False

            unit_size = 5
            no_middle = True 
            index_list = [0, 1, 2, 3, 4]
            ind_a = [2, 3, 4, 5, 6]
                
            ## for display
            input_var = torch.cat((invar[:,0:3], invar[:,6*3:6*3+3]), 1)
            
            SR_ret = None
            SR_list = []
            left_list = []
            
            k = torch.zeros([48, 3, 4, 4])
            for dim in range(0, 16):
                x = int(dim//4)
                y = int(dim % 4)
                k[dim*3, 0, x, y] = 1
                k[dim*3+1, 1, x, y] = 1 
                k[dim*3+2, 2, x, y] = 1 
            kernel = torch.FloatTensor(k)#.unsqueeze(0).unsqueeze(0)
            weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
            
            
            weight_on_loss = 1
            pixel_shuffle = nn.PixelShuffle(4)
            def depth_to_space(x_in):
                # x_in should have order of rgb or xyz on the dimension 2. 
                # this x_in is b * 3 * 16 * h * w
                # output of this is b * 3(rgb or xyz) * 4h * 4w
                x_ret = None
                for ch in range(0, 3):
                    x_in_ch = x_in[:, ch:ch+1]
                    x_ret_ch = pixel_shuffle(x_in_ch[:,0,0:16,:,:])
                    if x_ret is None:
                        x_ret = x_ret_ch
                    else:
                        x_ret = torch.cat((x_ret, x_ret_ch), 1)
                return x_ret
            for index in index_list:
                optimizer.zero_grad()
                loss = 0
                inter_loss = 0
                tvl = 0
                improc_loss = 0
                bd_loss = 0
                sr_loss = 0
                flow_loss = 0
                flow_loss_large = 0
                per_loss = 0
                per_loss_large = 0
                sim_loss = 0
                cyc_loss = 0
                rf_loss = 0

                f0_i = index
                f0_sr = invar_sr[:,f0_i*3:f0_i*3+3]
    
                f05_i = index + 1
                f05_sr = invar_sr[:,f05_i*3:f05_i*3+3]
                
                f1_i = index + 2
                f1_sr = invar_sr[:,f1_i*3:f1_i*3+3]
                
                flow10 = FNet(torch.cat((f0_sr, f1_sr), 1))
                warped_frame1_f0 = warp(f0_sr, flow10, l0_size, scale_down=(1,1))
                
                left_list.append(warped_frame1_f0)
                flow_loss += criterion(warped_frame1_f0, f1_sr)
                
                tvl += 5 *  tvLoss(flow10) 
                per_loss += 5 * perception_loss(warped_frame1_f0, f1_sr) 
                bd_loss += one_bd_loss(flow10)
                
#                 x, y = np.meshgrid(np.linspace(-1,1,7), np.linspace(-1,1,7))
#                 d = np.sqrt(x*x+y*y)
#                 sigma, mu = 0.2, 0.0
#                 g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
#                 g = g / g.sum()
#                 g = torch.tensor(g).cuda()
#                 warp_ret = None
#                 pad = torch.nn.ReplicationPad2d(3)
#                 f0_sr_padded = pad(f0_sr)
#                 x_s = flow10.shape[2]
#                 y_s = flow10.shape[3]
#                 for x_i in range(0, 7):
#                     for y_i in range(0, 7):
#                         flow10_bak = flow10
# #                         flow10_bak[:,0] = flow10_bak[:,0] + x_i - 3
# #                         flow10_bak[:,1] = flow10_bak[:,1] + y_i - 3
#                         warped_frame1_f0_tmp = warp(f0_sr_padded[:,:,x_i:x_i+x_s,y_i:y_i+y_s], flow10_bak, l0_size, scale_down=(1,1))
#                         if warp_ret is None:
#                             warp_ret= torch.zeros_like(warped_frame1_f0_tmp)
#                         warp_ret = warp_ret + g[x_i, y_i] * warped_frame1_f0_tmp
#                 flow_loss_large = criterion(warp_ret, f1_sr)
#                 per_loss_large = 5 * perception_loss(warp_ret, f1_sr) 
#                 left_list.append(warp_ret)
#                 warp_ret = warp_block(f0_sr, flow10, l0_size, scale_down=(1,1))
#                 flow_loss_large = criterion(warp_ret, f1_sr)
#                 per_loss_large = 5 * perception_loss(warp_ret, f1_sr) 
#                 left_list.append(warp_ret)
                
                
                        
                        
                    
                
                
#                 rf_times = 2
#                 for rf_t in range(0, rf_times):
#                     ## PNET
#                     error_map = torch.sum(torch.abs(f1_sr - warped_frame1_f0), 1).unsqueeze(1)
#                     p_weight = torch.nn.functional.relu(PNet(torch.cat((f0_sr, f1_sr, torch.abs(f1_sr - warped_frame1_f0), flow10), 1))) 
# #                     p_weight = p_weight + torch.ones_like(p_weight) * 1e-8
#                     p_weight = torch.nn.functional.softmax(p_weight, dim=1)
# #                     print(p_weight[0,:,48:50,48:50])
                                                        
#                     patched_flow_x = patch_it(flow10[:,0:1])
#                     patched_flow_y = patch_it(flow10[:,1:2])
#                     p_weight_sum = torch.sum(p_weight, 1)
# #                     print(p_weight_sum)
                    
# #                     print(patched_flow_x.shape, p_weight.shape)
#                     propa_flow_x = torch.sum(p_weight * patched_flow_x, 1) #/ p_weight_sum 
#                     propa_flow_y = torch.sum(p_weight * patched_flow_y, 1) #/ p_weight_sum 
                    
#                     flow10 = torch.cat((propa_flow_x.unsqueeze(1), propa_flow_y.unsqueeze(1)), 1)
#                     warped_frame1_f0 = warp(f0_sr, flow10, l0_size, scale_down=(1,1))
#                     left_list.append(warped_frame1_f0)
                    
#                     if real_step % scalar_gap == 0:
#                         last_loss = now_loss
#                         now_loss = criterion(warped_frame1_f0, f1_sr)
#                         cmp_loss = now_loss - last_loss
#                         writer.add_scalar('cmp_loss_' + str(rf_t)+'_p', cmp_loss, real_step)
                    
#                     rf_loss += criterion(warped_frame1_f0 * error_map, f1_sr * error_map)
#                     tvl +=   tvLoss(flow10) 
#                     per_loss += perception_loss(warped_frame1_f0, f1_sr) 
#                     bd_loss += one_bd_loss(flow10)
#                     error_map = torch.sum(torch.abs(f1_sr - warped_frame1_f0), 1).unsqueeze(1)
                    
                    
#                     ## RNET
#                     flow10_r = RNet(torch.cat((f0_sr, f1_sr, torch.abs(f1_sr - warped_frame1_f0),  flow10), 1))
                    
#                     flow10 = flow10 + flow10_r
#                     warped_frame1_f0 = warp(f0_sr, flow10, l0_size, scale_down=(1,1))
#                     left_list.append(warped_frame1_f0)
#                     rf_loss += criterion(warped_frame1_f0 * error_map, f1_sr * error_map)
#                     if real_step % scalar_gap == 0:
#                         last_loss = now_loss
#                         now_loss = criterion(warped_frame1_f0, f1_sr)
#                         cmp_loss = now_loss - last_loss
#                         writer.add_scalar('cmp_loss_' + str(rf_t)+'_rf', cmp_loss, real_step)
#                     tvl +=   tvLoss(flow10) 
#                     per_loss += perception_loss(warped_frame1_f0, f1_sr) 
#                     bd_loss += one_bd_loss(flow10)
                    
                SR_list.append(warped_frame1_f0)

                ## loss term 

                # for recurrent
        
#                 weight_on_loss /= loss_deg

#             if bd_loss > 0.0001:
                loss += bdl_w * bd_loss
    #             if tvl > 0.01 :
                loss += tvl_w * tvl
                loss += srl_w * flow_loss + pl_w * per_loss + 10 * rf_loss + srl_w * flow_loss_large + pl_w * per_loss_large



                loss.backward()
#                 for m in FNet.modules():
#                     if isinstance(m, nn.Conv2d):
#                         print(m.weight.grad)
                optimizer.step()

            

            ## Log information
            if real_step % scalar_gap == 0:
                psnr_a = 0
                for ind in range(0, unit_size):
                    final_output = SR_list[ind]
                    target_var = invar_sr[:,ind_a[ind]*3:ind_a[ind]*3+3]
                    for img in range(0, batch_size):
                        out = (final_output[img].clamp(0.0, 1.0).data.cpu().numpy() * 255.0).astype(numpy.uint8)
                        tar = (target_var[img].data.cpu().numpy()*255.0).astype(numpy.uint8)
                        tar = img_as_float(np.array(tar))
                        out = img_as_float(np.array(out))
                        psnr_a += 10*np.log10(1.0/np.square(np.subtract(np.array(out), np.array(tar))).mean())
                psnr_a = psnr_a / (batch_size * unit_size)
                writer.add_scalar('loss', loss.data, real_step)
                writer.add_scalar('per_loss', per_loss.data, real_step)
                writer.add_scalar('rf_loss', rf_loss, real_step)
                writer.add_scalar('tvl', tvl.data, real_step)
                writer.add_scalar('bd_loss', bd_loss.data, real_step)
                writer.add_scalar('psnr_a', psnr_a, real_step)
#                 writer.add_scalar('sr_loss', sr_loss.data, real_step)
#                 writer.add_scalar('sim_loss', sim_loss.data, real_step)
                writer.add_scalar('flow_loss', flow_loss.data, real_step)
                writer.add_scalar('flow_loss_large', flow_loss_large, real_step)

            epoch_now = real_step / (epoch_num*1.0)
            if real_step % b_test_gap == 0 and real_step > 10:# and epoch_num > 2:
                print("Test!!!!!!!!!!!!!!") 
                now_psnr, now_ssim = test(2, writer, 1) 
                print("Test end.")  
                if now_psnr > best_psnr: 
                    best_psnr = now_psnr 
                    torch.save(FNet.state_dict(), 'best_FNet.pkl')
                    torch.save(RNet.state_dict(), 'best_RNet.pkl')
                    torch.save(PNet.state_dict(), 'best_PNet.pkl')
                    torch.save(best_psnr, 'best_psnr.pkl')
                    torch.save(now_ssim, 'now_ssim.pkl')
                    
                FNet.train()
                SR.train()
                
            if real_step % s_test_gap == 0 and real_step > 10:
                torch.save(FNet.state_dict(), 'FNet.pkl')
                torch.save(RNet.state_dict(), 'RNet.pkl')
                torch.save(PNet.state_dict(), 'PNet.pkl')
#                 torch.save(SR.state_dict(), 'SR.pkl')
#                 torch.save(best_psnr, 'best_psnr.pkl')
#                 print("little Test!")
#                 test(2, writer, 25)
#                 print("Test end.")
                    
#                 FNet.train()
#                 SR.train()


            if real_step % show_img_gap == 0:
                print("Loss at step %d epoch %f batch %d: %f, lr=%.15f best_psnr%.4f" % (real_step, real_step / (epoch_num*1.0), batch_idx, loss.data, lr_sr, best_psnr))
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
                    show_flow = visual_flow(flow10, batch_size)
        
                    writer.add_image('Image/flow10', show_f(vutils.make_grid(torch.from_numpy(show_flow).permute(0,3,1,2), normalize=False, scale_each=True, nrow=4)), real_step)
                    writer.add_image('Image/in1_sr', show(vutils.make_grid(invar_sr[:, 0:3].cpu(), normalize=False, scale_each=True, nrow=4)), real_step)
                    writer.add_image('Image/in2_sr', show(vutils.make_grid(invar_sr[:, 6:9].cpu(), normalize=False, scale_each=True, nrow=4)), real_step)
                    for sr_out_ind in range(0, len(SR_list)):
                        writer.add_image('Image/out_'+str(sr_out_ind), show(vutils.make_grid(SR_list[sr_out_ind].clamp(0.0, 1.0).data.cpu(), nrow=4, normalize=False, scale_each=True)), real_step)
                    for ind in range(0, len(left_list)):
                        writer.add_image('Image/left_'+str(ind), show(vutils.make_grid(left_list[ind].clamp(0.0, 1.0).data.cpu(), normalize=False, nrow=4, scale_each=True)), real_step)
                    for tar_ind in range(0, unit_size):
                        writer.add_image('Image/target_'+str(tar_ind), show(vutils.make_grid(invar_sr[:, 3*ind_a[tar_ind]:3*ind_a[tar_ind] + 3].clamp(0.0, 1.0).data.cpu(), nrow=4, normalize=False, scale_each=True)), real_step)

if args.mode == "test":
    test(2, None, 25)
