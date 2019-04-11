from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import getopt
import math
import time
import numpy
import numpy as np
import torch
# import torch.utils.serialization
import PIL
import PIL.Image
import gc
from .loss_utils import L1Loss, L1Loss_pixel_wise


debug = False
debug2 = False

def meshgrid(height, width):
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y

def meshgrid_3d(dim, height, width):
    x = np.linspace(-1, 1, height)
    y = np.linspace(-1, 1, width)
    z = np.linspace(-1, 1, dim)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    return grid_x, grid_y, grid_z

def bi_interp(framein, coor_x, coor_y, t):
    return torch.nn.functional.grid_sample(
            framein,
            torch.stack([coor_x, coor_y], dim=3),
            padding_mode='border')

def bi_interp_3d(framein, coor_x, coor_y, coor_z, t):
    return torch.nn.functional.grid_sample(
            framein,
            torch.stack([coor_z, coor_x, coor_y], dim=4),
            padding_mode='border')

def bilinear_interp(im, x, y, name):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Args:
    im: Tensor of size [batch_size, height, width, depth]
    x: Tensor of size [batch_size, height, width, 1]
    y: Tensor of size [batch_size, height, width, 1]
    name: String for the name for this opt.
    Returns:
    Tensor of size [batch_size, height, width, depth]
    """

    # constants
    num_batch = np.shape(im)[0]
    channels = np.shape(im)[1]
    height = np.shape(im)[2]
    width = np.shape(im)[3]

    x = x.view(-1, height * width).repeat(1,channels).view(-1)
    y = y.view(-1, height * width).repeat(1,channels).view(-1)
#     print(y.shape, "yshape")
    if debug2:
        print(y,"y")

#     print(im.shape)
#     x = torch.FloatTensor(x)
#     y = torch.FloatTensor(y)
#     height_f = np.cast(height, 'float32')
#     width_f = np.cast(width, 'float32')
    height_f = height
    width_f =width

    zero = np.int32(0)

    max_x = width - 1
    max_y = height - 1
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0
    if debug2:
        print(y,"yyyy")
        print(max_x, max_y)

    # Sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, float(max_x)).long()
    x1 = torch.clamp(x1, 0, float(max_x)).long()
    y0 = torch.clamp(y0, 0, float(max_y)).long()
    y1 = torch.clamp(y1, 0, float(max_y)).long()

    dim2 = width
    dim1 = width * height

    # Create base index
    base = torch.LongTensor(torch.arange(num_batch*channels, out=torch.LongTensor()) * dim1)
    base = base.view([-1,1])
    base = base.repeat(1, height * width)
    if debug:
        print(base)
    base = torch.autograd.Variable(base.view([-1])).cuda()
    if debug:
        print(base.data,"basedata")
#     print(base.shape, "baseshape")
#     print(y0.shape, "y0shape")
    base_y0 = base + y0 * dim2
#     print(base_y0.shape,"basey0_shape")
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    if debug2:
        print(y0, "y0")
        print(idx_a, "idx_a")
        print(base_y0, "base_y0")

    # Use indices to look up pixels
    im_flat = im.contiguous().view([-1])
#     print(im_flat.shape, "im_flatshape")
#     im_flat = np.float(im_flat)
#     print(idx_a.shape, "idxa.shape")
#     print(idx_a.view(-1,1).repeat(1, 3).shape, "idxrepeat")
    pixel_a = torch.gather(im_flat, 0, idx_a.view(-1))
    pixel_b = torch.gather(im_flat, 0, idx_b.view(-1))
    pixel_c = torch.gather(im_flat, 0, idx_c.view(-1))
    pixel_d = torch.gather(im_flat, 0, idx_d.view(-1))

    if debug:
        print(pixel_a, pixel_a.shape)
    # Interpolate the values
    x1_f = x1.float()
    y1_f = y1.float()

    wa = (x1_f - x) * (y1_f - y)
    wa = wa.view(-1)
    wb = (x1_f - x) * (1.0 - (y1_f - y))
    wb = wb.view(-1)
    wc = (1.0 - (x1_f - x)) * (y1_f - y)
    wc = wc.view(-1)
    wd = (1.0 - (x1_f - x)) * (1.0 - (y1_f - y))
    wd = wd.view(-1)
    if debug:
        print(pixel_a.shape,wa.shape)
    output = wa*pixel_a + wb*pixel_b + wc*pixel_c + wd*pixel_d
    output = output.view([num_batch, channels, height, width])
    return output

def bilinear_interp_large(im, x, y, name):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Args:
    im: Tensor of size [batch_size, height, width, depth]
    x: Tensor of size [batch_size, height, width, 1]
    y: Tensor of size [batch_size, height, width, 1]
    name: String for the name for this opt.
    Returns:
    Tensor of size [batch_size, height, width, depth]
    """

    # constants
    num_batch = np.shape(im)[0]
    channels = np.shape(im)[1]
    height = np.shape(im)[2]
    width = np.shape(im)[3]

    x = x.view(-1, height * width).repeat(1,channels).view(-1)
    y = y.view(-1, height * width).repeat(1,channels).view(-1)
#     print(y.shape, "yshape")
    if debug2:
        print(y,"y")

#     print(im.shape)
#     x = torch.FloatTensor(x)
#     y = torch.FloatTensor(y)
#     height_f = np.cast(height, 'float32')
#     width_f = np.cast(width, 'float32')
    height_f = height
    width_f =width

    zero = np.int32(0)

    max_x = width - 1
    max_y = height - 1
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0
    if debug2:
        print(y,"yyyy")
        print(max_x, max_y)

    # Sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, float(max_x)).long()
    x1 = torch.clamp(x1, 0, float(max_x)).long()
    y0 = torch.clamp(y0, 0, float(max_y)).long()
    y1 = torch.clamp(y1, 0, float(max_y)).long()

    dim2 = width
    dim1 = width * height

    # Create base index
    base = torch.LongTensor(torch.arange(num_batch*channels, out=torch.LongTensor()) * dim1)
    base = base.view([-1,1])
    base = base.repeat(1, height * width)
    if debug:
        print(base)
    base = torch.autograd.Variable(base.view([-1])).cuda()
    if debug:
        print(base.data,"basedata")
#     print(base.shape, "baseshape")
#     print(y0.shape, "y0shape")
    base_y0 = base + y0 * dim2
#     print(base_y0.shape,"basey0_shape")
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    
    def get_idx(x, y):
        x = torch.clamp(x, 0, float(max_x)).long()
        y = torch.clamp(y, 0, float(max_y)).long()
        base_y = base + y * dim2
        idx = base_y + x
        return idx
        
    if debug2:
        print(y0, "y0")
        print(idx_a, "idx_a")
        print(base_y0, "base_y0")

    # Use indices to look up pixels
    im_flat = im.contiguous().view([-1])
#     print(im_flat.shape, "im_flatshape")
#     im_flat = np.float(im_flat)
#     print(idx_a.shape, "idxa.shape")
#     print(idx_a.view(-1,1).repeat(1, 3).shape, "idxrepeat")
    x_g, y_g = np.meshgrid(np.linspace(-1,1,6), np.linspace(-1,1,6))
    d = np.sqrt(x_g*x_g+y_g*y_g)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    g = g / g.sum() * 4
    g = torch.tensor(g).cuda()
    def get_pixel_block(pos, x, y):
        if pos == 'a':
            dir_x = -1
            dir_y = -1
            wi = 2
            wj = 2
        if pos == 'b':
            dir_x = -1
            dir_y = 1
            wi = 2
            wj = 3
        if pos == 'd':
            dir_x = 1
            dir_y = 1
            wi = 3
            wj = 3
        if pos == 'c':
            dir_x = 1
            dir_y = -1
            wi = 3
            wj = 2
        block_a = None
        for i in range(0, 3):
            for j in range(0, 3):
                wi_n = wi + dir_x * i
                x_n = x + dir_x * i
                wj_n = wj + dir_y * j
                y_n = y + dir_y * j
                idx = get_idx(x_n, y_n)
                pixel = torch.gather(im_flat, 0, idx.view(-1))
                if block_a is None:
                    block_a = torch.zeros_like(pixel)
                block_a += pixel * g[wi_n, wj_n]
        return block_a
                
#     pixel_a_orig = torch.gather(im_flat, 0, idx_a.view(-1))

    pixel_a = get_pixel_block('a', x0, y0)
    pixel_b = get_pixel_block('b', x0, y1)
    pixel_c = get_pixel_block('c', x1, y0)
    pixel_d = get_pixel_block('d', x1, y1)
#     print(torch.sum(torch.abs(pixel_a_orig-pixel_a)), '!!!!!')

    if debug:
        print(pixel_a, pixel_a.shape)
    # Interpolate the values
    x1_f = x1.float()
    y1_f = y1.float()

    wa = (x1_f - x) * (y1_f - y)
    wa = wa.view(-1)
    wb = (x1_f - x) * (1.0 - (y1_f - y))
    wb = wb.view(-1)
    wc = (1.0 - (x1_f - x)) * (y1_f - y)
    wc = wc.view(-1)
    wd = (1.0 - (x1_f - x)) * (1.0 - (y1_f - y))
    wd = wd.view(-1)
    if debug:
        print(pixel_a.shape,wa.shape)
    output = wa*pixel_a + wb*pixel_b + wc*pixel_c + wd*pixel_d
    output = output.view([num_batch, channels, height, width])
    return output


def to_there(im, x, y, src):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Args:
    im: Tensor of size [batch_size, height, width, depth] 
    x: Tensor of size [batch_size, height, width, 1]
    y: Tensor of size [batch_size, height, width, 1]
    name: String for the name for this opt.
    Returns:
    Tensor of size [batch_size, height, width, depth]
    """

    # constants
    num_batch = np.shape(im)[0]
    channels = np.shape(im)[1]
    height = np.shape(im)[2]
    width = np.shape(im)[3]
    
    x = x.view(-1, height * width).repeat(1,channels).view(-1)
    y = y.view(-1, height * width).repeat(1,channels).view(-1)
#     print(y.shape, "yshape")
    if debug2:
        print(y,"y")


    
#     print(im.shape)
#     x = torch.FloatTensor(x)
#     y = torch.FloatTensor(y)
#     height_f = np.cast(height, 'float32')
#     width_f = np.cast(width, 'float32')
    height_f = height
    width_f =width
    
    zero = np.int32(0)

    max_x = width - 1
    max_y = height - 1
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0
    if debug2:
        print(y,"yyyy")
        print(max_x, max_y)
     
    # Sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    

    x0 = torch.clamp(x0, 0, float(max_x)).long()
    x1 = torch.clamp(x1, 0, float(max_x)).long()
    y0 = torch.clamp(y0, 0, float(max_y)).long()
    y1 = torch.clamp(y1, 0, float(max_y)).long()

    dim2 = width 
    dim1 = width * height


    # Create base index
    base = torch.LongTensor(torch.arange(num_batch*channels, out=torch.LongTensor()) * dim1)
    base = base.view([-1,1])
    base = base.repeat(1, height * width)
    if debug:
        print(base)
    base = torch.autograd.Variable(base.view([-1])).cuda()
    if debug:
        print(base.data,"basedata")
#     print(base.shape, "baseshape")
#     print(y0.shape, "y0shape")
    base_y0 = base + y0 * dim2
#     print(base_y0.shape,"basey0_shape")
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0 
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    if debug2:
        print(y0, "y0")
        print(idx_a, "idx_a")
        print(base_y0, "base_y0")
    

    # Use indices to look up pixels 
    im_flat_a = im.contiguous().view([-1])
    im_flat_b = im_flat_a * 1.0
    im_flat_c = im_flat_a * 1.0
    im_flat_d = im_flat_a * 1.0
    src_flat = src.contiguous().view([-1])
    im_flat_a[idx_a] = src_flat
    im_flat_b[idx_b] = src_flat
    im_flat_c[idx_c] = src_flat
    im_flat_d[idx_d] = src_flat
    
    x1_f = x1.float()
    y1_f = y1.float()
    
    wa = (x1_f - x) * (y1_f - y)
    wa = wa.view(-1)
    wb = (x1_f - x) * (1.0 - (y1_f - y))
    wb = wb.view(-1)
    wc = (1.0 - (x1_f - x)) * (y1_f - y)
    wc = wc.view(-1)
    wd = (1.0 - (x1_f - x)) * (1.0 - (y1_f - y))
    wd = wd.view(-1)
    
    im_flat = wa*im_flat_a + wb*im_flat_b + wc*im_flat_c + wd*im_flat_d
    output = im_flat.view([num_batch, channels, height, width])
    return output


def warp(framein, flow, handle_size=256, scale_down=1):
    handle_size = flow.shape[2]
    batch_size = flow.shape[0]
    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()
    coor_x = grid_x + flow[:, 0, :, :] / scale_down
    coor_y = grid_y + flow[:, 1, :, :] / scale_down

    output = bi_interp(framein, coor_x, coor_y, 'interpolate')
    return output


def forward_warp(flowab, src, scale_down=1):
    handle_size = flowab.shape[2]
    batch_size = flowab.shape[0]
    grid_x, grid_y = meshgrid(handle_size, handle_size)
    grid_x = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_x, [batch_size, 1, 1]))).cuda()
    grid_y = torch.autograd.Variable(torch.FloatTensor(np.tile(grid_y, [batch_size, 1, 1]))).cuda()

    coor_x = grid_x + flowab[:, 0, :, :] / scale_down 
    coor_y = grid_y + flowab[:, 1, :, :] / scale_down
    
    empty_im = torch.zeros_like(src).cuda()
    output = to_there(empty_im.clone(), coor_x, coor_y, src)
    return output


def prop_refine_rd(invar, targetvar, flowa, rf_times, middle_target=None, scale_down=1):
    with torch.no_grad():
        flow = flowa
        batch_size = flow.shape[0]
        t1 = time.time()
        refine_flow = torch.zeros(flow.shape).cuda()
        pad = torch.nn.ReplicationPad2d(2)
        now_flow = 0
        for i in range(0, rf_times):
            refine_flow = refine_flow * 0.0
            h_size = flow.shape[2]
            flow = torch.FloatTensor(batch_size, 2, h_size, h_size).normal_(1e-4, 1e-2).cuda() - torch.FloatTensor(batch_size, 2, h_size, h_size).normal_(1e-4, 1e-2).cuda()
            flow_padded = pad(flow)
            start_list = [0, 1, 2, 3, 4]
            part_flow = None
            sub_ret = None
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
                    abflow = None
                    if flowa is None:
                        abflow = part_flow
                    else:
                        abflow = part_flow + flowa
                    warped_frame = warp(invar, abflow, h_size, scale_down=scale_down)
                    sub_ret_1c = L1Loss_pixel_wise(warped_frame, targetvar).view(batch_size, 1, flow.shape[2], flow.shape[3])
                    if middle_target is not None:
                        middle_warped_frame = warp(invar, abflow / 2.0, h_size, scale_down=scale_down)
                        sub_ret_1c += L1Loss_pixel_wise(middle_warped_frame, middle_target).view(batch_size, 1, flow.shape[2], flow.shape[3])

                    if sub_ret is  None:
                        sub_ret = sub_ret_1c
                    else:
                        sub_ret = torch.cat((sub_ret, sub_ret_1c), 1)

#                del part_flow 
#                del warped_frame
#                del sub_ret_1c
#gc.collect()
            if flowa is not None:
#                 if i == 0:
                flow_record = torch.cat((flow_record, torch.zeros(1, batch_size, 2, flow.shape[2], flow.shape[3]).cuda()), 0)
                now_flow = torch.zeros(batch_size, 2, flow.shape[2], flow.shape[3]).cuda()
#                 else:
#                     flow_record = torch.cat((flow_record, now_flow.view(1, batch_size, 2, flow.shape[2], flow.shape[3])), 0)
                warped_frame = warp(invar, flowa, h_size, scale_down=scale_down)
                sub_ret_1c = L1Loss_pixel_wise(warped_frame, targetvar).view(batch_size, 1, flow.shape[2], flow.shape[3])
                sub_ret = torch.cat((sub_ret, sub_ret_1c), 1)
#                 print("here")
                
            _, which = torch.min(sub_ret, 1)
#             print(which, which.view(-1).float().sum())
            which = which.view(batch_size, 1, flow.shape[2], flow.shape[2]).repeat(1, 2, 1, 1)
            for j in range(0, 26):
                refine_flow = refine_flow + flow_record[j] * ((which == j).float())
#             warped_frame = warp(invar, flowa, h_size, scale_down=scale_down)
#             print(i, L1Loss(warped_frame, targetvar))
#             warped_frame = warp(invar, flowa + refine_flow, h_size, scale_down=scale_down)
#             print(i, L1Loss(warped_frame, targetvar))
            now_flow = refine_flow
            flowa = flowa + now_flow
            del flow_record
            del flow_padded 
            del sub_ret
            del which
            gc.collect()
        t3 = time.time()
#         print(t3-t1)
    return flowa


def prop_refine_which(invar, targetvar, flow, rf_times, middle_target=None, scale_down=1):
#with torch.no_grad():
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

        _, which_ori = torch.min(sub_ret, 1)
        which = which_ori.view(batch_size, 1, flow.shape[2], flow.shape[2]).repeat(1, 2, 1, 1)
        for j in range(0, 25):
            refine_flow = refine_flow + flow_record[j] * ((which == j).float())
#         warped_frame = warp(invar, refine_flow, h_size)
#         print(i, L1Loss(warped_frame, targetvar))
        flow = refine_flow
        del flow_record
        del flow_padded
        del sub_ret
        del which
        gc.collect()
    return which_ori, flow
