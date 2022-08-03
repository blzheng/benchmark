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
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm2d0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d6 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d9 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d12 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d16 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d19 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d22 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d25 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d27 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d29 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d32 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d35 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d38 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d41 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d44 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d46 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d48 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d51 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear0 = Linear(in_features=2048, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        print('x0: {}'.format(x0.shape))
        x1=self.conv2d0(x0)
        print('x1: {}'.format(x1.shape))
        x2=self.batchnorm2d0(x1)
        print('x2: {}'.format(x2.shape))
        x3=self.relu0(x2)
        print('x3: {}'.format(x3.shape))
        x4=self.maxpool2d0(x3)
        print('x4: {}'.format(x4.shape))
        x5=self.conv2d1(x4)
        print('x5: {}'.format(x5.shape))
        x6=self.batchnorm2d1(x5)
        print('x6: {}'.format(x6.shape))
        x7=self.relu1(x6)
        print('x7: {}'.format(x7.shape))
        x8=self.conv2d2(x7)
        print('x8: {}'.format(x8.shape))
        x9=self.batchnorm2d2(x8)
        print('x9: {}'.format(x9.shape))
        x10=self.relu1(x9)
        print('x10: {}'.format(x10.shape))
        x11=self.conv2d3(x10)
        print('x11: {}'.format(x11.shape))
        x12=self.batchnorm2d3(x11)
        print('x12: {}'.format(x12.shape))
        x13=self.conv2d4(x4)
        print('x13: {}'.format(x13.shape))
        x14=self.batchnorm2d4(x13)
        print('x14: {}'.format(x14.shape))
        x15=operator.add(x12, x14)
        print('x15: {}'.format(x15.shape))
        x16=self.relu1(x15)
        print('x16: {}'.format(x16.shape))
        x17=self.conv2d5(x16)
        print('x17: {}'.format(x17.shape))
        x18=self.batchnorm2d5(x17)
        print('x18: {}'.format(x18.shape))
        x19=self.relu4(x18)
        print('x19: {}'.format(x19.shape))
        x20=self.conv2d6(x19)
        print('x20: {}'.format(x20.shape))
        x21=self.batchnorm2d6(x20)
        print('x21: {}'.format(x21.shape))
        x22=self.relu4(x21)
        print('x22: {}'.format(x22.shape))
        x23=self.conv2d7(x22)
        print('x23: {}'.format(x23.shape))
        x24=self.batchnorm2d7(x23)
        print('x24: {}'.format(x24.shape))
        x25=operator.add(x24, x16)
        print('x25: {}'.format(x25.shape))
        x26=self.relu4(x25)
        print('x26: {}'.format(x26.shape))
        x27=self.conv2d8(x26)
        print('x27: {}'.format(x27.shape))
        x28=self.batchnorm2d8(x27)
        print('x28: {}'.format(x28.shape))
        x29=self.relu7(x28)
        print('x29: {}'.format(x29.shape))
        x30=self.conv2d9(x29)
        print('x30: {}'.format(x30.shape))
        x31=self.batchnorm2d9(x30)
        print('x31: {}'.format(x31.shape))
        x32=self.relu7(x31)
        print('x32: {}'.format(x32.shape))
        x33=self.conv2d10(x32)
        print('x33: {}'.format(x33.shape))
        x34=self.batchnorm2d10(x33)
        print('x34: {}'.format(x34.shape))
        x35=operator.add(x34, x26)
        print('x35: {}'.format(x35.shape))
        x36=self.relu7(x35)
        print('x36: {}'.format(x36.shape))
        x37=self.conv2d11(x36)
        print('x37: {}'.format(x37.shape))
        x38=self.batchnorm2d11(x37)
        print('x38: {}'.format(x38.shape))
        x39=self.relu10(x38)
        print('x39: {}'.format(x39.shape))
        x40=self.conv2d12(x39)
        print('x40: {}'.format(x40.shape))
        x41=self.batchnorm2d12(x40)
        print('x41: {}'.format(x41.shape))
        x42=self.relu10(x41)
        print('x42: {}'.format(x42.shape))
        x43=self.conv2d13(x42)
        print('x43: {}'.format(x43.shape))
        x44=self.batchnorm2d13(x43)
        print('x44: {}'.format(x44.shape))
        x45=self.conv2d14(x36)
        print('x45: {}'.format(x45.shape))
        x46=self.batchnorm2d14(x45)
        print('x46: {}'.format(x46.shape))
        x47=operator.add(x44, x46)
        print('x47: {}'.format(x47.shape))
        x48=self.relu10(x47)
        print('x48: {}'.format(x48.shape))
        x49=self.conv2d15(x48)
        print('x49: {}'.format(x49.shape))
        x50=self.batchnorm2d15(x49)
        print('x50: {}'.format(x50.shape))
        x51=self.relu13(x50)
        print('x51: {}'.format(x51.shape))
        x52=self.conv2d16(x51)
        print('x52: {}'.format(x52.shape))
        x53=self.batchnorm2d16(x52)
        print('x53: {}'.format(x53.shape))
        x54=self.relu13(x53)
        print('x54: {}'.format(x54.shape))
        x55=self.conv2d17(x54)
        print('x55: {}'.format(x55.shape))
        x56=self.batchnorm2d17(x55)
        print('x56: {}'.format(x56.shape))
        x57=operator.add(x56, x48)
        print('x57: {}'.format(x57.shape))
        x58=self.relu13(x57)
        print('x58: {}'.format(x58.shape))
        x59=self.conv2d18(x58)
        print('x59: {}'.format(x59.shape))
        x60=self.batchnorm2d18(x59)
        print('x60: {}'.format(x60.shape))
        x61=self.relu16(x60)
        print('x61: {}'.format(x61.shape))
        x62=self.conv2d19(x61)
        print('x62: {}'.format(x62.shape))
        x63=self.batchnorm2d19(x62)
        print('x63: {}'.format(x63.shape))
        x64=self.relu16(x63)
        print('x64: {}'.format(x64.shape))
        x65=self.conv2d20(x64)
        print('x65: {}'.format(x65.shape))
        x66=self.batchnorm2d20(x65)
        print('x66: {}'.format(x66.shape))
        x67=operator.add(x66, x58)
        print('x67: {}'.format(x67.shape))
        x68=self.relu16(x67)
        print('x68: {}'.format(x68.shape))
        x69=self.conv2d21(x68)
        print('x69: {}'.format(x69.shape))
        x70=self.batchnorm2d21(x69)
        print('x70: {}'.format(x70.shape))
        x71=self.relu19(x70)
        print('x71: {}'.format(x71.shape))
        x72=self.conv2d22(x71)
        print('x72: {}'.format(x72.shape))
        x73=self.batchnorm2d22(x72)
        print('x73: {}'.format(x73.shape))
        x74=self.relu19(x73)
        print('x74: {}'.format(x74.shape))
        x75=self.conv2d23(x74)
        print('x75: {}'.format(x75.shape))
        x76=self.batchnorm2d23(x75)
        print('x76: {}'.format(x76.shape))
        x77=operator.add(x76, x68)
        print('x77: {}'.format(x77.shape))
        x78=self.relu19(x77)
        print('x78: {}'.format(x78.shape))
        x79=self.conv2d24(x78)
        print('x79: {}'.format(x79.shape))
        x80=self.batchnorm2d24(x79)
        print('x80: {}'.format(x80.shape))
        x81=self.relu22(x80)
        print('x81: {}'.format(x81.shape))
        x82=self.conv2d25(x81)
        print('x82: {}'.format(x82.shape))
        x83=self.batchnorm2d25(x82)
        print('x83: {}'.format(x83.shape))
        x84=self.relu22(x83)
        print('x84: {}'.format(x84.shape))
        x85=self.conv2d26(x84)
        print('x85: {}'.format(x85.shape))
        x86=self.batchnorm2d26(x85)
        print('x86: {}'.format(x86.shape))
        x87=self.conv2d27(x78)
        print('x87: {}'.format(x87.shape))
        x88=self.batchnorm2d27(x87)
        print('x88: {}'.format(x88.shape))
        x89=operator.add(x86, x88)
        print('x89: {}'.format(x89.shape))
        x90=self.relu22(x89)
        print('x90: {}'.format(x90.shape))
        x91=self.conv2d28(x90)
        print('x91: {}'.format(x91.shape))
        x92=self.batchnorm2d28(x91)
        print('x92: {}'.format(x92.shape))
        x93=self.relu25(x92)
        print('x93: {}'.format(x93.shape))
        x94=self.conv2d29(x93)
        print('x94: {}'.format(x94.shape))
        x95=self.batchnorm2d29(x94)
        print('x95: {}'.format(x95.shape))
        x96=self.relu25(x95)
        print('x96: {}'.format(x96.shape))
        x97=self.conv2d30(x96)
        print('x97: {}'.format(x97.shape))
        x98=self.batchnorm2d30(x97)
        print('x98: {}'.format(x98.shape))
        x99=operator.add(x98, x90)
        print('x99: {}'.format(x99.shape))
        x100=self.relu25(x99)
        print('x100: {}'.format(x100.shape))
        x101=self.conv2d31(x100)
        print('x101: {}'.format(x101.shape))
        x102=self.batchnorm2d31(x101)
        print('x102: {}'.format(x102.shape))
        x103=self.relu28(x102)
        print('x103: {}'.format(x103.shape))
        x104=self.conv2d32(x103)
        print('x104: {}'.format(x104.shape))
        x105=self.batchnorm2d32(x104)
        print('x105: {}'.format(x105.shape))
        x106=self.relu28(x105)
        print('x106: {}'.format(x106.shape))
        x107=self.conv2d33(x106)
        print('x107: {}'.format(x107.shape))
        x108=self.batchnorm2d33(x107)
        print('x108: {}'.format(x108.shape))
        x109=operator.add(x108, x100)
        print('x109: {}'.format(x109.shape))
        x110=self.relu28(x109)
        print('x110: {}'.format(x110.shape))
        x111=self.conv2d34(x110)
        print('x111: {}'.format(x111.shape))
        x112=self.batchnorm2d34(x111)
        print('x112: {}'.format(x112.shape))
        x113=self.relu31(x112)
        print('x113: {}'.format(x113.shape))
        x114=self.conv2d35(x113)
        print('x114: {}'.format(x114.shape))
        x115=self.batchnorm2d35(x114)
        print('x115: {}'.format(x115.shape))
        x116=self.relu31(x115)
        print('x116: {}'.format(x116.shape))
        x117=self.conv2d36(x116)
        print('x117: {}'.format(x117.shape))
        x118=self.batchnorm2d36(x117)
        print('x118: {}'.format(x118.shape))
        x119=operator.add(x118, x110)
        print('x119: {}'.format(x119.shape))
        x120=self.relu31(x119)
        print('x120: {}'.format(x120.shape))
        x121=self.conv2d37(x120)
        print('x121: {}'.format(x121.shape))
        x122=self.batchnorm2d37(x121)
        print('x122: {}'.format(x122.shape))
        x123=self.relu34(x122)
        print('x123: {}'.format(x123.shape))
        x124=self.conv2d38(x123)
        print('x124: {}'.format(x124.shape))
        x125=self.batchnorm2d38(x124)
        print('x125: {}'.format(x125.shape))
        x126=self.relu34(x125)
        print('x126: {}'.format(x126.shape))
        x127=self.conv2d39(x126)
        print('x127: {}'.format(x127.shape))
        x128=self.batchnorm2d39(x127)
        print('x128: {}'.format(x128.shape))
        x129=operator.add(x128, x120)
        print('x129: {}'.format(x129.shape))
        x130=self.relu34(x129)
        print('x130: {}'.format(x130.shape))
        x131=self.conv2d40(x130)
        print('x131: {}'.format(x131.shape))
        x132=self.batchnorm2d40(x131)
        print('x132: {}'.format(x132.shape))
        x133=self.relu37(x132)
        print('x133: {}'.format(x133.shape))
        x134=self.conv2d41(x133)
        print('x134: {}'.format(x134.shape))
        x135=self.batchnorm2d41(x134)
        print('x135: {}'.format(x135.shape))
        x136=self.relu37(x135)
        print('x136: {}'.format(x136.shape))
        x137=self.conv2d42(x136)
        print('x137: {}'.format(x137.shape))
        x138=self.batchnorm2d42(x137)
        print('x138: {}'.format(x138.shape))
        x139=operator.add(x138, x130)
        print('x139: {}'.format(x139.shape))
        x140=self.relu37(x139)
        print('x140: {}'.format(x140.shape))
        x141=self.conv2d43(x140)
        print('x141: {}'.format(x141.shape))
        x142=self.batchnorm2d43(x141)
        print('x142: {}'.format(x142.shape))
        x143=self.relu40(x142)
        print('x143: {}'.format(x143.shape))
        x144=self.conv2d44(x143)
        print('x144: {}'.format(x144.shape))
        x145=self.batchnorm2d44(x144)
        print('x145: {}'.format(x145.shape))
        x146=self.relu40(x145)
        print('x146: {}'.format(x146.shape))
        x147=self.conv2d45(x146)
        print('x147: {}'.format(x147.shape))
        x148=self.batchnorm2d45(x147)
        print('x148: {}'.format(x148.shape))
        x149=self.conv2d46(x140)
        print('x149: {}'.format(x149.shape))
        x150=self.batchnorm2d46(x149)
        print('x150: {}'.format(x150.shape))
        x151=operator.add(x148, x150)
        print('x151: {}'.format(x151.shape))
        x152=self.relu40(x151)
        print('x152: {}'.format(x152.shape))
        x153=self.conv2d47(x152)
        print('x153: {}'.format(x153.shape))
        x154=self.batchnorm2d47(x153)
        print('x154: {}'.format(x154.shape))
        x155=self.relu43(x154)
        print('x155: {}'.format(x155.shape))
        x156=self.conv2d48(x155)
        print('x156: {}'.format(x156.shape))
        x157=self.batchnorm2d48(x156)
        print('x157: {}'.format(x157.shape))
        x158=self.relu43(x157)
        print('x158: {}'.format(x158.shape))
        x159=self.conv2d49(x158)
        print('x159: {}'.format(x159.shape))
        x160=self.batchnorm2d49(x159)
        print('x160: {}'.format(x160.shape))
        x161=operator.add(x160, x152)
        print('x161: {}'.format(x161.shape))
        x162=self.relu43(x161)
        print('x162: {}'.format(x162.shape))
        x163=self.conv2d50(x162)
        print('x163: {}'.format(x163.shape))
        x164=self.batchnorm2d50(x163)
        print('x164: {}'.format(x164.shape))
        x165=self.relu46(x164)
        print('x165: {}'.format(x165.shape))
        x166=self.conv2d51(x165)
        print('x166: {}'.format(x166.shape))
        x167=self.batchnorm2d51(x166)
        print('x167: {}'.format(x167.shape))
        x168=self.relu46(x167)
        print('x168: {}'.format(x168.shape))
        x169=self.conv2d52(x168)
        print('x169: {}'.format(x169.shape))
        x170=self.batchnorm2d52(x169)
        print('x170: {}'.format(x170.shape))
        x171=operator.add(x170, x162)
        print('x171: {}'.format(x171.shape))
        x172=self.relu46(x171)
        print('x172: {}'.format(x172.shape))
        x173=self.adaptiveavgpool2d0(x172)
        print('x173: {}'.format(x173.shape))
        x174=torch.flatten(x173, 1)
        print('x174: {}'.format(x174.shape))
        x175=self.linear0(x174)
        print('x175: {}'.format(x175.shape))

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.randn(batch_size, 3, 224, 224)
start_time=time.time()
for i in range(10):
    output = m(x)
total_iter_time = time.time() - start_time
Throughput = batch_size * 10 / total_iter_time
file_current = os.path.basename(__file__)
print(file_current,',',BS,',',Throughput) 
