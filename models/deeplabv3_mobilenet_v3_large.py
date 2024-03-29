import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish0 = Hardswish()
        self.conv2d1 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d1 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d4 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d6 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        self.batchnorm2d7 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        self.batchnorm2d10 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d11 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
        self.relu7 = ReLU()
        self.conv2d12 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid0 = Hardsigmoid()
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d13 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d16 = Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
        self.relu10 = ReLU()
        self.conv2d17 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid1 = Hardsigmoid()
        self.conv2d18 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d16 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d21 = Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
        self.relu13 = ReLU()
        self.conv2d22 = Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid2 = Hardsigmoid()
        self.conv2d23 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish1 = Hardswish()
        self.conv2d25 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d19 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish2 = Hardswish()
        self.conv2d26 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish3 = Hardswish()
        self.conv2d28 = Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
        self.batchnorm2d22 = BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish4 = Hardswish()
        self.conv2d29 = Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d30 = Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish5 = Hardswish()
        self.conv2d31 = Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        self.batchnorm2d25 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish6 = Hardswish()
        self.conv2d32 = Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish7 = Hardswish()
        self.conv2d34 = Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        self.batchnorm2d28 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish8 = Hardswish()
        self.conv2d35 = Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish9 = Hardswish()
        self.conv2d37 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d31 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish10 = Hardswish()
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d38 = Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))
        self.relu14 = ReLU()
        self.conv2d39 = Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid3 = Hardsigmoid()
        self.conv2d40 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish11 = Hardswish()
        self.conv2d42 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
        self.batchnorm2d34 = BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish12 = Hardswish()
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d43 = Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
        self.relu15 = ReLU()
        self.conv2d44 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish13 = Hardswish()
        self.conv2d47 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=672, bias=False)
        self.batchnorm2d37 = BatchNorm2d(672, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish14 = Hardswish()
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d48 = Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
        self.relu16 = ReLU()
        self.conv2d49 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid5 = Hardsigmoid()
        self.conv2d50 = Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish15 = Hardswish()
        self.conv2d52 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=960, bias=False)
        self.batchnorm2d40 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish16 = Hardswish()
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d53 = Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
        self.relu17 = ReLU()
        self.conv2d54 = Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d55 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d56 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish17 = Hardswish()
        self.conv2d57 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=960, bias=False)
        self.batchnorm2d43 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish18 = Hardswish()
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d58 = Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
        self.relu18 = ReLU()
        self.conv2d59 = Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid7 = Hardsigmoid()
        self.conv2d60 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d61 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.hardswish19 = Hardswish()
        self.conv2d62 = Conv2d(960, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU()
        self.conv2d63 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False)
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU()
        self.conv2d64 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
        self.batchnorm2d48 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU()
        self.conv2d65 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), bias=False)
        self.batchnorm2d49 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU()
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d66 = Conv2d(960, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU()
        self.conv2d67 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU()
        self.conv2d69 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d70 = Conv2d(40, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.conv2d71 = Conv2d(10, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x0=x
        x1=builtins.getattr(x0, 'shape')
        x2=operator.getitem(x1, slice(-2, None, None))
        x3=self.conv2d0(x0)
        x4=self.batchnorm2d0(x3)
        x5=self.hardswish0(x4)
        x6=self.conv2d1(x5)
        x7=self.batchnorm2d1(x6)
        x8=self.relu0(x7)
        x9=self.conv2d2(x8)
        x10=self.batchnorm2d2(x9)
        x11=operator.add(x10, x5)
        x12=self.conv2d3(x11)
        x13=self.batchnorm2d3(x12)
        x14=self.relu1(x13)
        x15=self.conv2d4(x14)
        x16=self.batchnorm2d4(x15)
        x17=self.relu2(x16)
        x18=self.conv2d5(x17)
        x19=self.batchnorm2d5(x18)
        x20=self.conv2d6(x19)
        x21=self.batchnorm2d6(x20)
        x22=self.relu3(x21)
        x23=self.conv2d7(x22)
        x24=self.batchnorm2d7(x23)
        x25=self.relu4(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x28=operator.add(x27, x19)
        x29=self.conv2d9(x28)
        x30=self.batchnorm2d9(x29)
        x31=self.relu5(x30)
        x32=self.conv2d10(x31)
        x33=self.batchnorm2d10(x32)
        x34=self.relu6(x33)
        x35=self.adaptiveavgpool2d0(x34)
        x36=self.conv2d11(x35)
        x37=self.relu7(x36)
        x38=self.conv2d12(x37)
        x39=self.hardsigmoid0(x38)
        x40=operator.mul(x39, x34)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d11(x41)
        x43=self.conv2d14(x42)
        x44=self.batchnorm2d12(x43)
        x45=self.relu8(x44)
        x46=self.conv2d15(x45)
        x47=self.batchnorm2d13(x46)
        x48=self.relu9(x47)
        x49=self.adaptiveavgpool2d1(x48)
        x50=self.conv2d16(x49)
        x51=self.relu10(x50)
        x52=self.conv2d17(x51)
        x53=self.hardsigmoid1(x52)
        x54=operator.mul(x53, x48)
        x55=self.conv2d18(x54)
        x56=self.batchnorm2d14(x55)
        x57=operator.add(x56, x42)
        x58=self.conv2d19(x57)
        x59=self.batchnorm2d15(x58)
        x60=self.relu11(x59)
        x61=self.conv2d20(x60)
        x62=self.batchnorm2d16(x61)
        x63=self.relu12(x62)
        x64=self.adaptiveavgpool2d2(x63)
        x65=self.conv2d21(x64)
        x66=self.relu13(x65)
        x67=self.conv2d22(x66)
        x68=self.hardsigmoid2(x67)
        x69=operator.mul(x68, x63)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d17(x70)
        x72=operator.add(x71, x57)
        x73=self.conv2d24(x72)
        x74=self.batchnorm2d18(x73)
        x75=self.hardswish1(x74)
        x76=self.conv2d25(x75)
        x77=self.batchnorm2d19(x76)
        x78=self.hardswish2(x77)
        x79=self.conv2d26(x78)
        x80=self.batchnorm2d20(x79)
        x81=self.conv2d27(x80)
        x82=self.batchnorm2d21(x81)
        x83=self.hardswish3(x82)
        x84=self.conv2d28(x83)
        x85=self.batchnorm2d22(x84)
        x86=self.hardswish4(x85)
        x87=self.conv2d29(x86)
        x88=self.batchnorm2d23(x87)
        x89=operator.add(x88, x80)
        x90=self.conv2d30(x89)
        x91=self.batchnorm2d24(x90)
        x92=self.hardswish5(x91)
        x93=self.conv2d31(x92)
        x94=self.batchnorm2d25(x93)
        x95=self.hardswish6(x94)
        x96=self.conv2d32(x95)
        x97=self.batchnorm2d26(x96)
        x98=operator.add(x97, x89)
        x99=self.conv2d33(x98)
        x100=self.batchnorm2d27(x99)
        x101=self.hardswish7(x100)
        x102=self.conv2d34(x101)
        x103=self.batchnorm2d28(x102)
        x104=self.hardswish8(x103)
        x105=self.conv2d35(x104)
        x106=self.batchnorm2d29(x105)
        x107=operator.add(x106, x98)
        x108=self.conv2d36(x107)
        x109=self.batchnorm2d30(x108)
        x110=self.hardswish9(x109)
        x111=self.conv2d37(x110)
        x112=self.batchnorm2d31(x111)
        x113=self.hardswish10(x112)
        x114=self.adaptiveavgpool2d3(x113)
        x115=self.conv2d38(x114)
        x116=self.relu14(x115)
        x117=self.conv2d39(x116)
        x118=self.hardsigmoid3(x117)
        x119=operator.mul(x118, x113)
        x120=self.conv2d40(x119)
        x121=self.batchnorm2d32(x120)
        x122=self.conv2d41(x121)
        x123=self.batchnorm2d33(x122)
        x124=self.hardswish11(x123)
        x125=self.conv2d42(x124)
        x126=self.batchnorm2d34(x125)
        x127=self.hardswish12(x126)
        x128=self.adaptiveavgpool2d4(x127)
        x129=self.conv2d43(x128)
        x130=self.relu15(x129)
        x131=self.conv2d44(x130)
        x132=self.hardsigmoid4(x131)
        x133=operator.mul(x132, x127)
        x134=self.conv2d45(x133)
        x135=self.batchnorm2d35(x134)
        x136=operator.add(x135, x121)
        x137=self.conv2d46(x136)
        x138=self.batchnorm2d36(x137)
        x139=self.hardswish13(x138)
        x140=self.conv2d47(x139)
        x141=self.batchnorm2d37(x140)
        x142=self.hardswish14(x141)
        x143=self.adaptiveavgpool2d5(x142)
        x144=self.conv2d48(x143)
        x145=self.relu16(x144)
        x146=self.conv2d49(x145)
        x147=self.hardsigmoid5(x146)
        x148=operator.mul(x147, x142)
        x149=self.conv2d50(x148)
        x150=self.batchnorm2d38(x149)
        x151=self.conv2d51(x150)
        x152=self.batchnorm2d39(x151)
        x153=self.hardswish15(x152)
        x154=self.conv2d52(x153)
        x155=self.batchnorm2d40(x154)
        x156=self.hardswish16(x155)
        x157=self.adaptiveavgpool2d6(x156)
        x158=self.conv2d53(x157)
        x159=self.relu17(x158)
        x160=self.conv2d54(x159)
        x161=self.hardsigmoid6(x160)
        x162=operator.mul(x161, x156)
        x163=self.conv2d55(x162)
        x164=self.batchnorm2d41(x163)
        x165=operator.add(x164, x150)
        x166=self.conv2d56(x165)
        x167=self.batchnorm2d42(x166)
        x168=self.hardswish17(x167)
        x169=self.conv2d57(x168)
        x170=self.batchnorm2d43(x169)
        x171=self.hardswish18(x170)
        x172=self.adaptiveavgpool2d7(x171)
        x173=self.conv2d58(x172)
        x174=self.relu18(x173)
        x175=self.conv2d59(x174)
        x176=self.hardsigmoid7(x175)
        x177=operator.mul(x176, x171)
        x178=self.conv2d60(x177)
        x179=self.batchnorm2d44(x178)
        x180=operator.add(x179, x165)
        x181=self.conv2d61(x180)
        x182=self.batchnorm2d45(x181)
        x183=self.hardswish19(x182)
        x184=self.conv2d62(x183)
        x185=self.batchnorm2d46(x184)
        x186=self.relu19(x185)
        x187=self.conv2d63(x183)
        x188=self.batchnorm2d47(x187)
        x189=self.relu20(x188)
        x190=self.conv2d64(x183)
        x191=self.batchnorm2d48(x190)
        x192=self.relu21(x191)
        x193=self.conv2d65(x183)
        x194=self.batchnorm2d49(x193)
        x195=self.relu22(x194)
        x196=builtins.getattr(x183, 'shape')
        x197=operator.getitem(x196, slice(-2, None, None))
        x198=self.adaptiveavgpool2d8(x183)
        x199=self.conv2d66(x198)
        x200=self.batchnorm2d50(x199)
        x201=self.relu23(x200)
        x202=torch.nn.functional.interpolate(x201,size=x197, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)
        x203=torch.cat([x186, x189, x192, x195, x202],dim=1)
        x204=self.conv2d67(x203)
        x205=self.batchnorm2d51(x204)
        x206=self.relu24(x205)
        x207=self.dropout0(x206)
        x208=self.conv2d68(x207)
        x209=self.batchnorm2d52(x208)
        x210=self.relu25(x209)
        x211=self.conv2d69(x210)
        x212=torch.nn.functional.interpolate(x211,size=x2, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)
        x213=self.conv2d70(x42)
        x214=self.batchnorm2d53(x213)
        x215=self.relu26(x214)
        x216=self.dropout1(x215)
        x217=self.conv2d71(x216)
        x218=torch.nn.functional.interpolate(x217,size=x2, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)

m = M().eval()
x = torch.rand(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
