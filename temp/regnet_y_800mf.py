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
        self.conv2d0 = Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=4, bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d4 = Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
        self.relu3 = ReLU()
        self.conv2d5 = Conv2d(8, 64, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d6 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(64, 144, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d8 = Conv2d(64, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d10 = Conv2d(144, 16, kernel_size=(1, 1), stride=(1, 1))
        self.relu7 = ReLU()
        self.conv2d11 = Conv2d(16, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d12 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d15 = Conv2d(144, 36, kernel_size=(1, 1), stride=(1, 1))
        self.relu11 = ReLU()
        self.conv2d16 = Conv2d(36, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d13 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d20 = Conv2d(144, 36, kernel_size=(1, 1), stride=(1, 1))
        self.relu15 = ReLU()
        self.conv2d21 = Conv2d(36, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d22 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(144, 320, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d15 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(144, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d17 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d26 = Conv2d(320, 36, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()
        self.conv2d27 = Conv2d(36, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d28 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d20 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d31 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu23 = ReLU()
        self.conv2d32 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d33 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d23 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d36 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu27 = ReLU()
        self.conv2d37 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d38 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d26 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d41 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu31 = ReLU()
        self.conv2d42 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d43 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d29 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d46 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d48 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d32 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.adaptiveavgpool2d9 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d51 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d53 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d35 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d56 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()
        self.conv2d57 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d58 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.batchnorm2d38 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d61 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu47 = ReLU()
        self.conv2d62 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d63 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(320, 784, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d40 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d65 = Conv2d(320, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(784, 784, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=49, bias=False)
        self.batchnorm2d42 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d67 = Conv2d(784, 80, kernel_size=(1, 1), stride=(1, 1))
        self.relu51 = ReLU()
        self.conv2d68 = Conv2d(80, 784, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d69 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(784, 784, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=49, bias=False)
        self.batchnorm2d45 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.adaptiveavgpool2d13 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d72 = Conv2d(784, 196, kernel_size=(1, 1), stride=(1, 1))
        self.relu55 = ReLU()
        self.conv2d73 = Conv2d(196, 784, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d74 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear0 = Linear(in_features=784, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        print('x0: {}'.format(x0.shape))
        x1=self.conv2d0(x0)
        print('x1: {}'.format(x1.shape))
        x2=self.batchnorm2d0(x1)
        print('x2: {}'.format(x2.shape))
        x3=self.relu0(x2)
        print('x3: {}'.format(x3.shape))
        x4=self.conv2d1(x3)
        print('x4: {}'.format(x4.shape))
        x5=self.batchnorm2d1(x4)
        print('x5: {}'.format(x5.shape))
        x6=self.conv2d2(x3)
        print('x6: {}'.format(x6.shape))
        x7=self.batchnorm2d2(x6)
        print('x7: {}'.format(x7.shape))
        x8=self.relu1(x7)
        print('x8: {}'.format(x8.shape))
        x9=self.conv2d3(x8)
        print('x9: {}'.format(x9.shape))
        x10=self.batchnorm2d3(x9)
        print('x10: {}'.format(x10.shape))
        x11=self.relu2(x10)
        print('x11: {}'.format(x11.shape))
        x12=self.adaptiveavgpool2d0(x11)
        print('x12: {}'.format(x12.shape))
        x13=self.conv2d4(x12)
        print('x13: {}'.format(x13.shape))
        x14=self.relu3(x13)
        print('x14: {}'.format(x14.shape))
        x15=self.conv2d5(x14)
        print('x15: {}'.format(x15.shape))
        x16=self.sigmoid0(x15)
        print('x16: {}'.format(x16.shape))
        x17=operator.mul(x16, x11)
        print('x17: {}'.format(x17.shape))
        x18=self.conv2d6(x17)
        print('x18: {}'.format(x18.shape))
        x19=self.batchnorm2d4(x18)
        print('x19: {}'.format(x19.shape))
        x20=operator.add(x5, x19)
        print('x20: {}'.format(x20.shape))
        x21=self.relu4(x20)
        print('x21: {}'.format(x21.shape))
        x22=self.conv2d7(x21)
        print('x22: {}'.format(x22.shape))
        x23=self.batchnorm2d5(x22)
        print('x23: {}'.format(x23.shape))
        x24=self.conv2d8(x21)
        print('x24: {}'.format(x24.shape))
        x25=self.batchnorm2d6(x24)
        print('x25: {}'.format(x25.shape))
        x26=self.relu5(x25)
        print('x26: {}'.format(x26.shape))
        x27=self.conv2d9(x26)
        print('x27: {}'.format(x27.shape))
        x28=self.batchnorm2d7(x27)
        print('x28: {}'.format(x28.shape))
        x29=self.relu6(x28)
        print('x29: {}'.format(x29.shape))
        x30=self.adaptiveavgpool2d1(x29)
        print('x30: {}'.format(x30.shape))
        x31=self.conv2d10(x30)
        print('x31: {}'.format(x31.shape))
        x32=self.relu7(x31)
        print('x32: {}'.format(x32.shape))
        x33=self.conv2d11(x32)
        print('x33: {}'.format(x33.shape))
        x34=self.sigmoid1(x33)
        print('x34: {}'.format(x34.shape))
        x35=operator.mul(x34, x29)
        print('x35: {}'.format(x35.shape))
        x36=self.conv2d12(x35)
        print('x36: {}'.format(x36.shape))
        x37=self.batchnorm2d8(x36)
        print('x37: {}'.format(x37.shape))
        x38=operator.add(x23, x37)
        print('x38: {}'.format(x38.shape))
        x39=self.relu8(x38)
        print('x39: {}'.format(x39.shape))
        x40=self.conv2d13(x39)
        print('x40: {}'.format(x40.shape))
        x41=self.batchnorm2d9(x40)
        print('x41: {}'.format(x41.shape))
        x42=self.relu9(x41)
        print('x42: {}'.format(x42.shape))
        x43=self.conv2d14(x42)
        print('x43: {}'.format(x43.shape))
        x44=self.batchnorm2d10(x43)
        print('x44: {}'.format(x44.shape))
        x45=self.relu10(x44)
        print('x45: {}'.format(x45.shape))
        x46=self.adaptiveavgpool2d2(x45)
        print('x46: {}'.format(x46.shape))
        x47=self.conv2d15(x46)
        print('x47: {}'.format(x47.shape))
        x48=self.relu11(x47)
        print('x48: {}'.format(x48.shape))
        x49=self.conv2d16(x48)
        print('x49: {}'.format(x49.shape))
        x50=self.sigmoid2(x49)
        print('x50: {}'.format(x50.shape))
        x51=operator.mul(x50, x45)
        print('x51: {}'.format(x51.shape))
        x52=self.conv2d17(x51)
        print('x52: {}'.format(x52.shape))
        x53=self.batchnorm2d11(x52)
        print('x53: {}'.format(x53.shape))
        x54=operator.add(x39, x53)
        print('x54: {}'.format(x54.shape))
        x55=self.relu12(x54)
        print('x55: {}'.format(x55.shape))
        x56=self.conv2d18(x55)
        print('x56: {}'.format(x56.shape))
        x57=self.batchnorm2d12(x56)
        print('x57: {}'.format(x57.shape))
        x58=self.relu13(x57)
        print('x58: {}'.format(x58.shape))
        x59=self.conv2d19(x58)
        print('x59: {}'.format(x59.shape))
        x60=self.batchnorm2d13(x59)
        print('x60: {}'.format(x60.shape))
        x61=self.relu14(x60)
        print('x61: {}'.format(x61.shape))
        x62=self.adaptiveavgpool2d3(x61)
        print('x62: {}'.format(x62.shape))
        x63=self.conv2d20(x62)
        print('x63: {}'.format(x63.shape))
        x64=self.relu15(x63)
        print('x64: {}'.format(x64.shape))
        x65=self.conv2d21(x64)
        print('x65: {}'.format(x65.shape))
        x66=self.sigmoid3(x65)
        print('x66: {}'.format(x66.shape))
        x67=operator.mul(x66, x61)
        print('x67: {}'.format(x67.shape))
        x68=self.conv2d22(x67)
        print('x68: {}'.format(x68.shape))
        x69=self.batchnorm2d14(x68)
        print('x69: {}'.format(x69.shape))
        x70=operator.add(x55, x69)
        print('x70: {}'.format(x70.shape))
        x71=self.relu16(x70)
        print('x71: {}'.format(x71.shape))
        x72=self.conv2d23(x71)
        print('x72: {}'.format(x72.shape))
        x73=self.batchnorm2d15(x72)
        print('x73: {}'.format(x73.shape))
        x74=self.conv2d24(x71)
        print('x74: {}'.format(x74.shape))
        x75=self.batchnorm2d16(x74)
        print('x75: {}'.format(x75.shape))
        x76=self.relu17(x75)
        print('x76: {}'.format(x76.shape))
        x77=self.conv2d25(x76)
        print('x77: {}'.format(x77.shape))
        x78=self.batchnorm2d17(x77)
        print('x78: {}'.format(x78.shape))
        x79=self.relu18(x78)
        print('x79: {}'.format(x79.shape))
        x80=self.adaptiveavgpool2d4(x79)
        print('x80: {}'.format(x80.shape))
        x81=self.conv2d26(x80)
        print('x81: {}'.format(x81.shape))
        x82=self.relu19(x81)
        print('x82: {}'.format(x82.shape))
        x83=self.conv2d27(x82)
        print('x83: {}'.format(x83.shape))
        x84=self.sigmoid4(x83)
        print('x84: {}'.format(x84.shape))
        x85=operator.mul(x84, x79)
        print('x85: {}'.format(x85.shape))
        x86=self.conv2d28(x85)
        print('x86: {}'.format(x86.shape))
        x87=self.batchnorm2d18(x86)
        print('x87: {}'.format(x87.shape))
        x88=operator.add(x73, x87)
        print('x88: {}'.format(x88.shape))
        x89=self.relu20(x88)
        print('x89: {}'.format(x89.shape))
        x90=self.conv2d29(x89)
        print('x90: {}'.format(x90.shape))
        x91=self.batchnorm2d19(x90)
        print('x91: {}'.format(x91.shape))
        x92=self.relu21(x91)
        print('x92: {}'.format(x92.shape))
        x93=self.conv2d30(x92)
        print('x93: {}'.format(x93.shape))
        x94=self.batchnorm2d20(x93)
        print('x94: {}'.format(x94.shape))
        x95=self.relu22(x94)
        print('x95: {}'.format(x95.shape))
        x96=self.adaptiveavgpool2d5(x95)
        print('x96: {}'.format(x96.shape))
        x97=self.conv2d31(x96)
        print('x97: {}'.format(x97.shape))
        x98=self.relu23(x97)
        print('x98: {}'.format(x98.shape))
        x99=self.conv2d32(x98)
        print('x99: {}'.format(x99.shape))
        x100=self.sigmoid5(x99)
        print('x100: {}'.format(x100.shape))
        x101=operator.mul(x100, x95)
        print('x101: {}'.format(x101.shape))
        x102=self.conv2d33(x101)
        print('x102: {}'.format(x102.shape))
        x103=self.batchnorm2d21(x102)
        print('x103: {}'.format(x103.shape))
        x104=operator.add(x89, x103)
        print('x104: {}'.format(x104.shape))
        x105=self.relu24(x104)
        print('x105: {}'.format(x105.shape))
        x106=self.conv2d34(x105)
        print('x106: {}'.format(x106.shape))
        x107=self.batchnorm2d22(x106)
        print('x107: {}'.format(x107.shape))
        x108=self.relu25(x107)
        print('x108: {}'.format(x108.shape))
        x109=self.conv2d35(x108)
        print('x109: {}'.format(x109.shape))
        x110=self.batchnorm2d23(x109)
        print('x110: {}'.format(x110.shape))
        x111=self.relu26(x110)
        print('x111: {}'.format(x111.shape))
        x112=self.adaptiveavgpool2d6(x111)
        print('x112: {}'.format(x112.shape))
        x113=self.conv2d36(x112)
        print('x113: {}'.format(x113.shape))
        x114=self.relu27(x113)
        print('x114: {}'.format(x114.shape))
        x115=self.conv2d37(x114)
        print('x115: {}'.format(x115.shape))
        x116=self.sigmoid6(x115)
        print('x116: {}'.format(x116.shape))
        x117=operator.mul(x116, x111)
        print('x117: {}'.format(x117.shape))
        x118=self.conv2d38(x117)
        print('x118: {}'.format(x118.shape))
        x119=self.batchnorm2d24(x118)
        print('x119: {}'.format(x119.shape))
        x120=operator.add(x105, x119)
        print('x120: {}'.format(x120.shape))
        x121=self.relu28(x120)
        print('x121: {}'.format(x121.shape))
        x122=self.conv2d39(x121)
        print('x122: {}'.format(x122.shape))
        x123=self.batchnorm2d25(x122)
        print('x123: {}'.format(x123.shape))
        x124=self.relu29(x123)
        print('x124: {}'.format(x124.shape))
        x125=self.conv2d40(x124)
        print('x125: {}'.format(x125.shape))
        x126=self.batchnorm2d26(x125)
        print('x126: {}'.format(x126.shape))
        x127=self.relu30(x126)
        print('x127: {}'.format(x127.shape))
        x128=self.adaptiveavgpool2d7(x127)
        print('x128: {}'.format(x128.shape))
        x129=self.conv2d41(x128)
        print('x129: {}'.format(x129.shape))
        x130=self.relu31(x129)
        print('x130: {}'.format(x130.shape))
        x131=self.conv2d42(x130)
        print('x131: {}'.format(x131.shape))
        x132=self.sigmoid7(x131)
        print('x132: {}'.format(x132.shape))
        x133=operator.mul(x132, x127)
        print('x133: {}'.format(x133.shape))
        x134=self.conv2d43(x133)
        print('x134: {}'.format(x134.shape))
        x135=self.batchnorm2d27(x134)
        print('x135: {}'.format(x135.shape))
        x136=operator.add(x121, x135)
        print('x136: {}'.format(x136.shape))
        x137=self.relu32(x136)
        print('x137: {}'.format(x137.shape))
        x138=self.conv2d44(x137)
        print('x138: {}'.format(x138.shape))
        x139=self.batchnorm2d28(x138)
        print('x139: {}'.format(x139.shape))
        x140=self.relu33(x139)
        print('x140: {}'.format(x140.shape))
        x141=self.conv2d45(x140)
        print('x141: {}'.format(x141.shape))
        x142=self.batchnorm2d29(x141)
        print('x142: {}'.format(x142.shape))
        x143=self.relu34(x142)
        print('x143: {}'.format(x143.shape))
        x144=self.adaptiveavgpool2d8(x143)
        print('x144: {}'.format(x144.shape))
        x145=self.conv2d46(x144)
        print('x145: {}'.format(x145.shape))
        x146=self.relu35(x145)
        print('x146: {}'.format(x146.shape))
        x147=self.conv2d47(x146)
        print('x147: {}'.format(x147.shape))
        x148=self.sigmoid8(x147)
        print('x148: {}'.format(x148.shape))
        x149=operator.mul(x148, x143)
        print('x149: {}'.format(x149.shape))
        x150=self.conv2d48(x149)
        print('x150: {}'.format(x150.shape))
        x151=self.batchnorm2d30(x150)
        print('x151: {}'.format(x151.shape))
        x152=operator.add(x137, x151)
        print('x152: {}'.format(x152.shape))
        x153=self.relu36(x152)
        print('x153: {}'.format(x153.shape))
        x154=self.conv2d49(x153)
        print('x154: {}'.format(x154.shape))
        x155=self.batchnorm2d31(x154)
        print('x155: {}'.format(x155.shape))
        x156=self.relu37(x155)
        print('x156: {}'.format(x156.shape))
        x157=self.conv2d50(x156)
        print('x157: {}'.format(x157.shape))
        x158=self.batchnorm2d32(x157)
        print('x158: {}'.format(x158.shape))
        x159=self.relu38(x158)
        print('x159: {}'.format(x159.shape))
        x160=self.adaptiveavgpool2d9(x159)
        print('x160: {}'.format(x160.shape))
        x161=self.conv2d51(x160)
        print('x161: {}'.format(x161.shape))
        x162=self.relu39(x161)
        print('x162: {}'.format(x162.shape))
        x163=self.conv2d52(x162)
        print('x163: {}'.format(x163.shape))
        x164=self.sigmoid9(x163)
        print('x164: {}'.format(x164.shape))
        x165=operator.mul(x164, x159)
        print('x165: {}'.format(x165.shape))
        x166=self.conv2d53(x165)
        print('x166: {}'.format(x166.shape))
        x167=self.batchnorm2d33(x166)
        print('x167: {}'.format(x167.shape))
        x168=operator.add(x153, x167)
        print('x168: {}'.format(x168.shape))
        x169=self.relu40(x168)
        print('x169: {}'.format(x169.shape))
        x170=self.conv2d54(x169)
        print('x170: {}'.format(x170.shape))
        x171=self.batchnorm2d34(x170)
        print('x171: {}'.format(x171.shape))
        x172=self.relu41(x171)
        print('x172: {}'.format(x172.shape))
        x173=self.conv2d55(x172)
        print('x173: {}'.format(x173.shape))
        x174=self.batchnorm2d35(x173)
        print('x174: {}'.format(x174.shape))
        x175=self.relu42(x174)
        print('x175: {}'.format(x175.shape))
        x176=self.adaptiveavgpool2d10(x175)
        print('x176: {}'.format(x176.shape))
        x177=self.conv2d56(x176)
        print('x177: {}'.format(x177.shape))
        x178=self.relu43(x177)
        print('x178: {}'.format(x178.shape))
        x179=self.conv2d57(x178)
        print('x179: {}'.format(x179.shape))
        x180=self.sigmoid10(x179)
        print('x180: {}'.format(x180.shape))
        x181=operator.mul(x180, x175)
        print('x181: {}'.format(x181.shape))
        x182=self.conv2d58(x181)
        print('x182: {}'.format(x182.shape))
        x183=self.batchnorm2d36(x182)
        print('x183: {}'.format(x183.shape))
        x184=operator.add(x169, x183)
        print('x184: {}'.format(x184.shape))
        x185=self.relu44(x184)
        print('x185: {}'.format(x185.shape))
        x186=self.conv2d59(x185)
        print('x186: {}'.format(x186.shape))
        x187=self.batchnorm2d37(x186)
        print('x187: {}'.format(x187.shape))
        x188=self.relu45(x187)
        print('x188: {}'.format(x188.shape))
        x189=self.conv2d60(x188)
        print('x189: {}'.format(x189.shape))
        x190=self.batchnorm2d38(x189)
        print('x190: {}'.format(x190.shape))
        x191=self.relu46(x190)
        print('x191: {}'.format(x191.shape))
        x192=self.adaptiveavgpool2d11(x191)
        print('x192: {}'.format(x192.shape))
        x193=self.conv2d61(x192)
        print('x193: {}'.format(x193.shape))
        x194=self.relu47(x193)
        print('x194: {}'.format(x194.shape))
        x195=self.conv2d62(x194)
        print('x195: {}'.format(x195.shape))
        x196=self.sigmoid11(x195)
        print('x196: {}'.format(x196.shape))
        x197=operator.mul(x196, x191)
        print('x197: {}'.format(x197.shape))
        x198=self.conv2d63(x197)
        print('x198: {}'.format(x198.shape))
        x199=self.batchnorm2d39(x198)
        print('x199: {}'.format(x199.shape))
        x200=operator.add(x185, x199)
        print('x200: {}'.format(x200.shape))
        x201=self.relu48(x200)
        print('x201: {}'.format(x201.shape))
        x202=self.conv2d64(x201)
        print('x202: {}'.format(x202.shape))
        x203=self.batchnorm2d40(x202)
        print('x203: {}'.format(x203.shape))
        x204=self.conv2d65(x201)
        print('x204: {}'.format(x204.shape))
        x205=self.batchnorm2d41(x204)
        print('x205: {}'.format(x205.shape))
        x206=self.relu49(x205)
        print('x206: {}'.format(x206.shape))
        x207=self.conv2d66(x206)
        print('x207: {}'.format(x207.shape))
        x208=self.batchnorm2d42(x207)
        print('x208: {}'.format(x208.shape))
        x209=self.relu50(x208)
        print('x209: {}'.format(x209.shape))
        x210=self.adaptiveavgpool2d12(x209)
        print('x210: {}'.format(x210.shape))
        x211=self.conv2d67(x210)
        print('x211: {}'.format(x211.shape))
        x212=self.relu51(x211)
        print('x212: {}'.format(x212.shape))
        x213=self.conv2d68(x212)
        print('x213: {}'.format(x213.shape))
        x214=self.sigmoid12(x213)
        print('x214: {}'.format(x214.shape))
        x215=operator.mul(x214, x209)
        print('x215: {}'.format(x215.shape))
        x216=self.conv2d69(x215)
        print('x216: {}'.format(x216.shape))
        x217=self.batchnorm2d43(x216)
        print('x217: {}'.format(x217.shape))
        x218=operator.add(x203, x217)
        print('x218: {}'.format(x218.shape))
        x219=self.relu52(x218)
        print('x219: {}'.format(x219.shape))
        x220=self.conv2d70(x219)
        print('x220: {}'.format(x220.shape))
        x221=self.batchnorm2d44(x220)
        print('x221: {}'.format(x221.shape))
        x222=self.relu53(x221)
        print('x222: {}'.format(x222.shape))
        x223=self.conv2d71(x222)
        print('x223: {}'.format(x223.shape))
        x224=self.batchnorm2d45(x223)
        print('x224: {}'.format(x224.shape))
        x225=self.relu54(x224)
        print('x225: {}'.format(x225.shape))
        x226=self.adaptiveavgpool2d13(x225)
        print('x226: {}'.format(x226.shape))
        x227=self.conv2d72(x226)
        print('x227: {}'.format(x227.shape))
        x228=self.relu55(x227)
        print('x228: {}'.format(x228.shape))
        x229=self.conv2d73(x228)
        print('x229: {}'.format(x229.shape))
        x230=self.sigmoid13(x229)
        print('x230: {}'.format(x230.shape))
        x231=operator.mul(x230, x225)
        print('x231: {}'.format(x231.shape))
        x232=self.conv2d74(x231)
        print('x232: {}'.format(x232.shape))
        x233=self.batchnorm2d46(x232)
        print('x233: {}'.format(x233.shape))
        x234=operator.add(x219, x233)
        print('x234: {}'.format(x234.shape))
        x235=self.relu56(x234)
        print('x235: {}'.format(x235.shape))
        x236=self.adaptiveavgpool2d14(x235)
        print('x236: {}'.format(x236.shape))
        x237=x236.flatten(start_dim=1)
        print('x237: {}'.format(x237.shape))
        x238=self.linear0(x237)
        print('x238: {}'.format(x238.shape))

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
