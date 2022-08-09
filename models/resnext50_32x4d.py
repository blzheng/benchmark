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
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        x4=self.maxpool2d0(x3)
        x5=self.conv2d1(x4)
        x6=self.batchnorm2d1(x5)
        x7=self.relu1(x6)
        x8=self.conv2d2(x7)
        x9=self.batchnorm2d2(x8)
        x10=self.relu1(x9)
        x11=self.conv2d3(x10)
        x12=self.batchnorm2d3(x11)
        x13=self.conv2d4(x4)
        x14=self.batchnorm2d4(x13)
        x15=operator.add(x12, x14)
        x16=self.relu1(x15)
        x17=self.conv2d5(x16)
        x18=self.batchnorm2d5(x17)
        x19=self.relu4(x18)
        x20=self.conv2d6(x19)
        x21=self.batchnorm2d6(x20)
        x22=self.relu4(x21)
        x23=self.conv2d7(x22)
        x24=self.batchnorm2d7(x23)
        x25=operator.add(x24, x16)
        x26=self.relu4(x25)
        x27=self.conv2d8(x26)
        x28=self.batchnorm2d8(x27)
        x29=self.relu7(x28)
        x30=self.conv2d9(x29)
        x31=self.batchnorm2d9(x30)
        x32=self.relu7(x31)
        x33=self.conv2d10(x32)
        x34=self.batchnorm2d10(x33)
        x35=operator.add(x34, x26)
        x36=self.relu7(x35)
        x37=self.conv2d11(x36)
        x38=self.batchnorm2d11(x37)
        x39=self.relu10(x38)
        x40=self.conv2d12(x39)
        x41=self.batchnorm2d12(x40)
        x42=self.relu10(x41)
        x43=self.conv2d13(x42)
        x44=self.batchnorm2d13(x43)
        x45=self.conv2d14(x36)
        x46=self.batchnorm2d14(x45)
        x47=operator.add(x44, x46)
        x48=self.relu10(x47)
        x49=self.conv2d15(x48)
        x50=self.batchnorm2d15(x49)
        x51=self.relu13(x50)
        x52=self.conv2d16(x51)
        x53=self.batchnorm2d16(x52)
        x54=self.relu13(x53)
        x55=self.conv2d17(x54)
        x56=self.batchnorm2d17(x55)
        x57=operator.add(x56, x48)
        x58=self.relu13(x57)
        x59=self.conv2d18(x58)
        x60=self.batchnorm2d18(x59)
        x61=self.relu16(x60)
        x62=self.conv2d19(x61)
        x63=self.batchnorm2d19(x62)
        x64=self.relu16(x63)
        x65=self.conv2d20(x64)
        x66=self.batchnorm2d20(x65)
        x67=operator.add(x66, x58)
        x68=self.relu16(x67)
        x69=self.conv2d21(x68)
        x70=self.batchnorm2d21(x69)
        x71=self.relu19(x70)
        x72=self.conv2d22(x71)
        x73=self.batchnorm2d22(x72)
        x74=self.relu19(x73)
        x75=self.conv2d23(x74)
        x76=self.batchnorm2d23(x75)
        x77=operator.add(x76, x68)
        x78=self.relu19(x77)
        x79=self.conv2d24(x78)
        x80=self.batchnorm2d24(x79)
        x81=self.relu22(x80)
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        x84=self.relu22(x83)
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        x87=self.conv2d27(x78)
        x88=self.batchnorm2d27(x87)
        x89=operator.add(x86, x88)
        x90=self.relu22(x89)
        x91=self.conv2d28(x90)
        x92=self.batchnorm2d28(x91)
        x93=self.relu25(x92)
        x94=self.conv2d29(x93)
        x95=self.batchnorm2d29(x94)
        x96=self.relu25(x95)
        x97=self.conv2d30(x96)
        x98=self.batchnorm2d30(x97)
        x99=operator.add(x98, x90)
        x100=self.relu25(x99)
        x101=self.conv2d31(x100)
        x102=self.batchnorm2d31(x101)
        x103=self.relu28(x102)
        x104=self.conv2d32(x103)
        x105=self.batchnorm2d32(x104)
        x106=self.relu28(x105)
        x107=self.conv2d33(x106)
        x108=self.batchnorm2d33(x107)
        x109=operator.add(x108, x100)
        x110=self.relu28(x109)
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        x113=self.relu31(x112)
        x114=self.conv2d35(x113)
        x115=self.batchnorm2d35(x114)
        x116=self.relu31(x115)
        x117=self.conv2d36(x116)
        x118=self.batchnorm2d36(x117)
        x119=operator.add(x118, x110)
        x120=self.relu31(x119)
        x121=self.conv2d37(x120)
        x122=self.batchnorm2d37(x121)
        x123=self.relu34(x122)
        x124=self.conv2d38(x123)
        x125=self.batchnorm2d38(x124)
        x126=self.relu34(x125)
        x127=self.conv2d39(x126)
        x128=self.batchnorm2d39(x127)
        x129=operator.add(x128, x120)
        x130=self.relu34(x129)
        x131=self.conv2d40(x130)
        x132=self.batchnorm2d40(x131)
        x133=self.relu37(x132)
        x134=self.conv2d41(x133)
        x135=self.batchnorm2d41(x134)
        x136=self.relu37(x135)
        x137=self.conv2d42(x136)
        x138=self.batchnorm2d42(x137)
        x139=operator.add(x138, x130)
        x140=self.relu37(x139)
        x141=self.conv2d43(x140)
        x142=self.batchnorm2d43(x141)
        x143=self.relu40(x142)
        x144=self.conv2d44(x143)
        x145=self.batchnorm2d44(x144)
        x146=self.relu40(x145)
        x147=self.conv2d45(x146)
        x148=self.batchnorm2d45(x147)
        x149=self.conv2d46(x140)
        x150=self.batchnorm2d46(x149)
        x151=operator.add(x148, x150)
        x152=self.relu40(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu43(x154)
        x156=self.conv2d48(x155)
        x157=self.batchnorm2d48(x156)
        x158=self.relu43(x157)
        x159=self.conv2d49(x158)
        x160=self.batchnorm2d49(x159)
        x161=operator.add(x160, x152)
        x162=self.relu43(x161)
        x163=self.conv2d50(x162)
        x164=self.batchnorm2d50(x163)
        x165=self.relu46(x164)
        x166=self.conv2d51(x165)
        x167=self.batchnorm2d51(x166)
        x168=self.relu46(x167)
        x169=self.conv2d52(x168)
        x170=self.batchnorm2d52(x169)
        x171=operator.add(x170, x162)
        x172=self.relu46(x171)
        x173=self.adaptiveavgpool2d0(x172)
        x174=torch.flatten(x173, 1)
        x175=self.linear0(x174)

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.rand(1, 3, 224, 224)
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
