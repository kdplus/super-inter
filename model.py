import sys
import os
import time
import math
import numpy as np
import torch
import torch.utils.serialization
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np

class Pyramid(torch.nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.LeakyReLU(0.1),
            )
        def Basic_safe(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.LeakyReLU(0.1),
            )

        self.Conv1 = Basic(3, 9)
        self.Conv2 = Basic(9, 32)

        self.DeConv2 = Basic_safe(32, 9)
        self.Upsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=9, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        self.DeConv1 = Basic_safe(9, 3)
        self.Upsample1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
#torch.nn.BatchNorm2d(num_features=16, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, It):
        c1 = self.Conv1(It)
        c2 = self.Conv2(c1)
        d2 = self.DeConv2(c2)
        u2 = self.Upsample2(d2)
        d1 = self.DeConv1(u2)
        u1 = self.Upsample1(d1)
        return [It, c1, c2, u2, u1]


class Network(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(Network, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
                torch.nn.ReLU(inplace=False),
            )

        self.moduleConv1 = Basic(in_c, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=512, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(64, momentum=0.5),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv1 = Basic(64, out_c)
        self.moduleUpsample1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=5, stride=1, padding=2),
#            torch.nn.BatchNorm2d(out_c, momentum=0.5),
#            torch.nn.ReLU(inplace=False)
        )

    #         self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
    # end

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not none:
                    nn.init.uniform_(m.bias)
                nn.init.kaiming_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, variableJoin):
        variableConv1 = self.moduleConv1(variableJoin)
        variablePool1 = self.modulePool1(variableConv1)

        variableConv2 = self.moduleConv2(variablePool1)
        variablePool2 = self.modulePool2(variableConv2)

        variableConv3 = self.moduleConv3(variablePool2)
        variablePool3 = self.modulePool3(variableConv3)

        variableConv4 = self.moduleConv4(variablePool3)
        variablePool4 = self.modulePool4(variableConv4)

        variableConv5 = self.moduleConv5(variablePool4)
        variablePool5 = self.modulePool5(variableConv5)

        variableDeconv5 = self.moduleDeconv5(variablePool5)
        variableUpsample5 = self.moduleUpsample5(variableDeconv5)

        variableDeconv4 = self.moduleDeconv4(variableUpsample5 + variableConv5)
        variableUpsample4 = self.moduleUpsample4(variableDeconv4)

        variableDeconv3 = self.moduleDeconv3(variableUpsample4 + variableConv4)
        variableUpsample3 = self.moduleUpsample3(variableDeconv3)

        variableDeconv2 = self.moduleDeconv2(variableUpsample3 + variableConv3)
        variableUpsample2 = self.moduleUpsample2(variableDeconv2)

        variableDeconv1 = self.moduleDeconv1(variableUpsample2 + variableConv2)
        variableUpsample1 = self.moduleUpsample1(variableDeconv1)

        return variableUpsample1

    
class SharpNet(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super(SharpNet, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=2, padding=1),
#                 torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
#                 nn.InstanceNorm2d(intOutput, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
#                 nn.InstanceNorm2d(intOutput, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
#                 nn.InstanceNorm2d(intOutput, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
#                 nn.InstanceNorm2d(intOutput, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.C0 = torch.nn.Conv2d(in_channels=in_c, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        self.D1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
#                 torch.nn.BatchNorm2d(num_features=64, momentum=0.5),
#                 nn.InstanceNorm2d(64, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=128, momentum=0.5),
#                 nn.InstanceNorm2d(128, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=128, momentum=0.5),
#                 nn.InstanceNorm2d(128, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.D2 = Basic(128, 256)
        self.D3 = Basic(256, 512)
        self.U1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.C4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=256, momentum=0.5),
#                 nn.InstanceNorm2d(256, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=256, momentum=0.5),
#                 nn.InstanceNorm2d(256, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=256, momentum=0.5),
#                 nn.InstanceNorm2d(256, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.U2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.C5 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=128, momentum=0.5),
#                 nn.InstanceNorm2d(128, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=64, momentum=0.5),
#                 nn.InstanceNorm2d(64, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
        )
        self.U3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.C6 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=56, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=56, momentum=0.5),
#                 nn.InstanceNorm2d(56, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=56, out_channels=out_c, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(num_features=intOutput, momentum=0.5),
#                nn.LeakyReLU(0.2, inplace=True),
        )

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
        
        #return vin[:,3:6] + vC6
        return vC6

    
    

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out1 = self.residual(out)
        out2 = self.bn_mid(self.conv_mid(out1))
        out3 = torch.add(out2,residual)
        out4 = self.upscale4x(out3)
        out5 = self.conv_output(out4)
        return [residual, out1, out2, out3, out4, out5]


class Improc(nn.Module):
    def __init__(self):
        super(Improc, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        out = self.conv4(x)
        return out 
