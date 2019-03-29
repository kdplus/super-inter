import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np

class FlowNet(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(FlowNet, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(2, stride=2),
            )
        
        def Basic_bilinear(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
            )

        self.C0 = Basic(in_c, 32) 
        self.C1 = Basic(32, 64) 
        self.C2 = Basic(64, 128) 
        
        self.D2 = Basic_bilinear(128, 256)
        self.D1 = Basic_bilinear(256, 128)
        self.D0 = Basic_bilinear(128, 64)
        
        self.C3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=32, out_channels=out_c, kernel_size=3, stride=1, padding=1),
#                 nn.Tanh(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.normal_(m.bias)
#                 nn.init.kaiming_uniform_(m.weight)
                nn.init.xavier_normal_(m.weight)


    def forward(self, vin):
        vC0 = self.C0(vin)
        vC1 = self.C1(vC0)
        vC2 = self.C2(vC1)
        vD2 = self.D2(vC2)
        vD1 = self.D1(vD2)
        vD0 = self.D0(vD1)
        vC3 = self.C3(vD0)
#         vC3[:,0] /= (vC3.shape[3] / 2)
#         vC3[:,1] /= (vC3.shape[2] / 2)
        return vC3

    
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output,identity_data)
        return output 

class SRNet(nn.Module):
    def __init__(self, in_c):
        super(SRNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=in_c, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block, 10)
        
        self.conv_trans = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.normal_(m.bias)
                nn.init.xavier_normal_(m.weight)
                
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.normal_(m.bias)
                nn.init.xavier_normal_(m.weight)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
#         residual = out
        out = self.residual(out)
        out = self.conv_trans(out)
        out = self.conv_output(out)
        return out 


class SharpNet_C2(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(SharpNet_C2, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.C0 = torch.nn.Conv2d(in_channels=in_c, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        self.D1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.D2 = Basic(128, 256)
        self.D3 = Basic(256, 512)
        self.U1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.C4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.U2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.C5 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.U3 = torch.nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1)
        self.C6 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=96, out_channels=out_c, kernel_size=3, stride=1, padding=1),
        )
#         self.C6_2 = torch.nn.Sequential(
#                 torch.nn.Conv2d(in_channels=64, out_channels=56, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 torch.nn.Conv2d(in_channels=56, out_channels=5, kernel_size=3, stride=1, padding=1),
#         )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, vin):
        vC0 = self.C0(vin)
        vD1 = self.D1(vC0)
        vD2 = self.D2(vD1)
        vD3 = self.D3(vD2)
        
        vU1 = self.U1(vD3)
        vC4 = self.C4(vU1 + vD2)
        
        vU2 = self.U2(vC4)
        vC5 = self.C5(vU2 + vD1)
        
        vU3 = self.U3(vC5)
        vC6 = self.C6(vU3)
#         return torch.tanh(vC6)
        return vC6
