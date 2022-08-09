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
        self.batchnorm2d0 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.identity0 = Identity()
        self.conv2d3 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d5 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.identity1 = Identity()
        self.conv2d6 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d7 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.identity2 = Identity()
        self.conv2d9 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d11 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.identity3 = Identity()
        self.conv2d12 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d14 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.identity4 = Identity()
        self.conv2d15 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=96, bias=False)
        self.batchnorm2d17 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.identity5 = Identity()
        self.conv2d18 = Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d20 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.identity6 = Identity()
        self.conv2d21 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d23 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.identity7 = Identity()
        self.conv2d24 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d26 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.identity8 = Identity()
        self.conv2d27 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d29 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.identity9 = Identity()
        self.conv2d30 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d32 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.identity10 = Identity()
        self.conv2d33 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d35 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.identity11 = Identity()
        self.conv2d36 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.batchnorm2d38 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.identity12 = Identity()
        self.conv2d39 = Conv2d(384, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d41 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.identity13 = Identity()
        self.conv2d42 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d44 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.identity14 = Identity()
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
        self.batchnorm2d47 = BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.identity15 = Identity()
        self.conv2d48 = Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d50 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.identity16 = Identity()
        self.conv2d51 = Conv2d(672, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d53 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.identity17 = Identity()
        self.conv2d54 = Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d55 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d56 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.identity18 = Identity()
        self.conv2d57 = Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d58 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(1104, 1104, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1104, bias=False)
        self.batchnorm2d59 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.identity19 = Identity()
        self.conv2d60 = Conv2d(1104, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d61 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(1104, 1104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1104, bias=False)
        self.batchnorm2d62 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)
        self.identity20 = Identity()
        self.conv2d63 = Conv2d(1104, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(352, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(352, 1984, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(1984, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear0 = Linear(in_features=1984, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu1(x5)
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        x9=self.relu2(x8)
        x10=self.identity0(x9)
        x11=self.conv2d3(x10)
        x12=self.batchnorm2d3(x11)
        x13=operator.add(x12, x3)
        x14=self.conv2d4(x13)
        x15=self.batchnorm2d4(x14)
        x16=self.relu3(x15)
        x17=self.conv2d5(x16)
        x18=self.batchnorm2d5(x17)
        x19=self.relu4(x18)
        x20=self.identity1(x19)
        x21=self.conv2d6(x20)
        x22=self.batchnorm2d6(x21)
        x23=self.conv2d7(x22)
        x24=self.batchnorm2d7(x23)
        x25=self.relu5(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x28=self.relu6(x27)
        x29=self.identity2(x28)
        x30=self.conv2d9(x29)
        x31=self.batchnorm2d9(x30)
        x32=operator.add(x31, x22)
        x33=self.conv2d10(x32)
        x34=self.batchnorm2d10(x33)
        x35=self.relu7(x34)
        x36=self.conv2d11(x35)
        x37=self.batchnorm2d11(x36)
        x38=self.relu8(x37)
        x39=self.identity3(x38)
        x40=self.conv2d12(x39)
        x41=self.batchnorm2d12(x40)
        x42=operator.add(x41, x32)
        x43=self.conv2d13(x42)
        x44=self.batchnorm2d13(x43)
        x45=self.relu9(x44)
        x46=self.conv2d14(x45)
        x47=self.batchnorm2d14(x46)
        x48=self.relu10(x47)
        x49=self.identity4(x48)
        x50=self.conv2d15(x49)
        x51=self.batchnorm2d15(x50)
        x52=self.conv2d16(x51)
        x53=self.batchnorm2d16(x52)
        x54=self.relu11(x53)
        x55=self.conv2d17(x54)
        x56=self.batchnorm2d17(x55)
        x57=self.relu12(x56)
        x58=self.identity5(x57)
        x59=self.conv2d18(x58)
        x60=self.batchnorm2d18(x59)
        x61=operator.add(x60, x51)
        x62=self.conv2d19(x61)
        x63=self.batchnorm2d19(x62)
        x64=self.relu13(x63)
        x65=self.conv2d20(x64)
        x66=self.batchnorm2d20(x65)
        x67=self.relu14(x66)
        x68=self.identity6(x67)
        x69=self.conv2d21(x68)
        x70=self.batchnorm2d21(x69)
        x71=operator.add(x70, x61)
        x72=self.conv2d22(x71)
        x73=self.batchnorm2d22(x72)
        x74=self.relu15(x73)
        x75=self.conv2d23(x74)
        x76=self.batchnorm2d23(x75)
        x77=self.relu16(x76)
        x78=self.identity7(x77)
        x79=self.conv2d24(x78)
        x80=self.batchnorm2d24(x79)
        x81=operator.add(x80, x71)
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        x84=self.relu17(x83)
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        x87=self.relu18(x86)
        x88=self.identity8(x87)
        x89=self.conv2d27(x88)
        x90=self.batchnorm2d27(x89)
        x91=self.conv2d28(x90)
        x92=self.batchnorm2d28(x91)
        x93=self.relu19(x92)
        x94=self.conv2d29(x93)
        x95=self.batchnorm2d29(x94)
        x96=self.relu20(x95)
        x97=self.identity9(x96)
        x98=self.conv2d30(x97)
        x99=self.batchnorm2d30(x98)
        x100=operator.add(x99, x90)
        x101=self.conv2d31(x100)
        x102=self.batchnorm2d31(x101)
        x103=self.relu21(x102)
        x104=self.conv2d32(x103)
        x105=self.batchnorm2d32(x104)
        x106=self.relu22(x105)
        x107=self.identity10(x106)
        x108=self.conv2d33(x107)
        x109=self.batchnorm2d33(x108)
        x110=operator.add(x109, x100)
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        x113=self.relu23(x112)
        x114=self.conv2d35(x113)
        x115=self.batchnorm2d35(x114)
        x116=self.relu24(x115)
        x117=self.identity11(x116)
        x118=self.conv2d36(x117)
        x119=self.batchnorm2d36(x118)
        x120=operator.add(x119, x110)
        x121=self.conv2d37(x120)
        x122=self.batchnorm2d37(x121)
        x123=self.relu25(x122)
        x124=self.conv2d38(x123)
        x125=self.batchnorm2d38(x124)
        x126=self.relu26(x125)
        x127=self.identity12(x126)
        x128=self.conv2d39(x127)
        x129=self.batchnorm2d39(x128)
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        x132=self.relu27(x131)
        x133=self.conv2d41(x132)
        x134=self.batchnorm2d41(x133)
        x135=self.relu28(x134)
        x136=self.identity13(x135)
        x137=self.conv2d42(x136)
        x138=self.batchnorm2d42(x137)
        x139=operator.add(x138, x129)
        x140=self.conv2d43(x139)
        x141=self.batchnorm2d43(x140)
        x142=self.relu29(x141)
        x143=self.conv2d44(x142)
        x144=self.batchnorm2d44(x143)
        x145=self.relu30(x144)
        x146=self.identity14(x145)
        x147=self.conv2d45(x146)
        x148=self.batchnorm2d45(x147)
        x149=operator.add(x148, x139)
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        x152=self.relu31(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu32(x154)
        x156=self.identity15(x155)
        x157=self.conv2d48(x156)
        x158=self.batchnorm2d48(x157)
        x159=operator.add(x158, x149)
        x160=self.conv2d49(x159)
        x161=self.batchnorm2d49(x160)
        x162=self.relu33(x161)
        x163=self.conv2d50(x162)
        x164=self.batchnorm2d50(x163)
        x165=self.relu34(x164)
        x166=self.identity16(x165)
        x167=self.conv2d51(x166)
        x168=self.batchnorm2d51(x167)
        x169=self.conv2d52(x168)
        x170=self.batchnorm2d52(x169)
        x171=self.relu35(x170)
        x172=self.conv2d53(x171)
        x173=self.batchnorm2d53(x172)
        x174=self.relu36(x173)
        x175=self.identity17(x174)
        x176=self.conv2d54(x175)
        x177=self.batchnorm2d54(x176)
        x178=operator.add(x177, x168)
        x179=self.conv2d55(x178)
        x180=self.batchnorm2d55(x179)
        x181=self.relu37(x180)
        x182=self.conv2d56(x181)
        x183=self.batchnorm2d56(x182)
        x184=self.relu38(x183)
        x185=self.identity18(x184)
        x186=self.conv2d57(x185)
        x187=self.batchnorm2d57(x186)
        x188=operator.add(x187, x178)
        x189=self.conv2d58(x188)
        x190=self.batchnorm2d58(x189)
        x191=self.relu39(x190)
        x192=self.conv2d59(x191)
        x193=self.batchnorm2d59(x192)
        x194=self.relu40(x193)
        x195=self.identity19(x194)
        x196=self.conv2d60(x195)
        x197=self.batchnorm2d60(x196)
        x198=operator.add(x197, x188)
        x199=self.conv2d61(x198)
        x200=self.batchnorm2d61(x199)
        x201=self.relu41(x200)
        x202=self.conv2d62(x201)
        x203=self.batchnorm2d62(x202)
        x204=self.relu42(x203)
        x205=self.identity20(x204)
        x206=self.conv2d63(x205)
        x207=self.batchnorm2d63(x206)
        x208=self.conv2d64(x207)
        x209=self.batchnorm2d64(x208)
        x210=self.relu43(x209)
        x211=self.adaptiveavgpool2d0(x210)
        x212=x211.flatten(1)
        x213=self.linear0(x212)

m = M().eval()
x = torch.rand(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
