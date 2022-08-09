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
        self.conv2d0 = Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu0 = SiLU(inplace=True)
        self.conv2d1 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d1 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu1 = SiLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d2 = Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu2 = SiLU(inplace=True)
        self.conv2d3 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d4 = Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu3 = SiLU(inplace=True)
        self.conv2d6 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d4 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu4 = SiLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d7 = Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
        self.silu5 = SiLU(inplace=True)
        self.conv2d8 = Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d9 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu6 = SiLU(inplace=True)
        self.conv2d11 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu7 = SiLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d12 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu8 = SiLU(inplace=True)
        self.conv2d13 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d14 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d15 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu9 = SiLU(inplace=True)
        self.conv2d16 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu10 = SiLU(inplace=True)
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d17 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu11 = SiLU(inplace=True)
        self.conv2d18 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d19 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d20 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu12 = SiLU(inplace=True)
        self.conv2d21 = Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        self.batchnorm2d13 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu13 = SiLU(inplace=True)
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d22 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu14 = SiLU(inplace=True)
        self.conv2d23 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d24 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu15 = SiLU(inplace=True)
        self.conv2d26 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d16 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu16 = SiLU(inplace=True)
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d27 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu17 = SiLU(inplace=True)
        self.conv2d28 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d29 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d30 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu18 = SiLU(inplace=True)
        self.conv2d31 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d19 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu19 = SiLU(inplace=True)
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d32 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu20 = SiLU(inplace=True)
        self.conv2d33 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d34 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d35 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu21 = SiLU(inplace=True)
        self.conv2d36 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d22 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu22 = SiLU(inplace=True)
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d37 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu23 = SiLU(inplace=True)
        self.conv2d38 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d39 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu24 = SiLU(inplace=True)
        self.conv2d41 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d25 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu25 = SiLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d42 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu26 = SiLU(inplace=True)
        self.conv2d43 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d44 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu27 = SiLU(inplace=True)
        self.conv2d46 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d28 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu28 = SiLU(inplace=True)
        self.adaptiveavgpool2d9 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d47 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu29 = SiLU(inplace=True)
        self.conv2d48 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d49 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d50 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu30 = SiLU(inplace=True)
        self.conv2d51 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d31 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu31 = SiLU(inplace=True)
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d52 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu32 = SiLU(inplace=True)
        self.conv2d53 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d54 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d55 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu33 = SiLU(inplace=True)
        self.conv2d56 = Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d34 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu34 = SiLU(inplace=True)
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d57 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu35 = SiLU(inplace=True)
        self.conv2d58 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d59 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d60 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu36 = SiLU(inplace=True)
        self.conv2d61 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d37 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu37 = SiLU(inplace=True)
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d62 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu38 = SiLU(inplace=True)
        self.conv2d63 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d64 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d65 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu39 = SiLU(inplace=True)
        self.conv2d66 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d40 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu40 = SiLU(inplace=True)
        self.adaptiveavgpool2d13 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d67 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu41 = SiLU(inplace=True)
        self.conv2d68 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d69 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d70 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu42 = SiLU(inplace=True)
        self.conv2d71 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d43 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu43 = SiLU(inplace=True)
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d72 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu44 = SiLU(inplace=True)
        self.conv2d73 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d74 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d75 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu45 = SiLU(inplace=True)
        self.conv2d76 = Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
        self.batchnorm2d46 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu46 = SiLU(inplace=True)
        self.adaptiveavgpool2d15 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d77 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu47 = SiLU(inplace=True)
        self.conv2d78 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d79 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d80 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu48 = SiLU(inplace=True)
        self.adaptiveavgpool2d16 = AdaptiveAvgPool2d(output_size=1)
        self.dropout0 = Dropout(p=0.2, inplace=True)
        self.linear0 = Linear(in_features=1280, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.silu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.silu1(x5)
        x7=self.adaptiveavgpool2d0(x6)
        x8=self.conv2d2(x7)
        x9=self.silu2(x8)
        x10=self.conv2d3(x9)
        x11=self.sigmoid0(x10)
        x12=operator.mul(x11, x6)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d2(x13)
        x15=self.conv2d5(x14)
        x16=self.batchnorm2d3(x15)
        x17=self.silu3(x16)
        x18=self.conv2d6(x17)
        x19=self.batchnorm2d4(x18)
        x20=self.silu4(x19)
        x21=self.adaptiveavgpool2d1(x20)
        x22=self.conv2d7(x21)
        x23=self.silu5(x22)
        x24=self.conv2d8(x23)
        x25=self.sigmoid1(x24)
        x26=operator.mul(x25, x20)
        x27=self.conv2d9(x26)
        x28=self.batchnorm2d5(x27)
        x29=self.conv2d10(x28)
        x30=self.batchnorm2d6(x29)
        x31=self.silu6(x30)
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d7(x32)
        x34=self.silu7(x33)
        x35=self.adaptiveavgpool2d2(x34)
        x36=self.conv2d12(x35)
        x37=self.silu8(x36)
        x38=self.conv2d13(x37)
        x39=self.sigmoid2(x38)
        x40=operator.mul(x39, x34)
        x41=self.conv2d14(x40)
        x42=self.batchnorm2d8(x41)
        x43=stochastic_depth(x42, 0.025, 'row', False)
        x44=operator.add(x43, x28)
        x45=self.conv2d15(x44)
        x46=self.batchnorm2d9(x45)
        x47=self.silu9(x46)
        x48=self.conv2d16(x47)
        x49=self.batchnorm2d10(x48)
        x50=self.silu10(x49)
        x51=self.adaptiveavgpool2d3(x50)
        x52=self.conv2d17(x51)
        x53=self.silu11(x52)
        x54=self.conv2d18(x53)
        x55=self.sigmoid3(x54)
        x56=operator.mul(x55, x50)
        x57=self.conv2d19(x56)
        x58=self.batchnorm2d11(x57)
        x59=self.conv2d20(x58)
        x60=self.batchnorm2d12(x59)
        x61=self.silu12(x60)
        x62=self.conv2d21(x61)
        x63=self.batchnorm2d13(x62)
        x64=self.silu13(x63)
        x65=self.adaptiveavgpool2d4(x64)
        x66=self.conv2d22(x65)
        x67=self.silu14(x66)
        x68=self.conv2d23(x67)
        x69=self.sigmoid4(x68)
        x70=operator.mul(x69, x64)
        x71=self.conv2d24(x70)
        x72=self.batchnorm2d14(x71)
        x73=stochastic_depth(x72, 0.05, 'row', False)
        x74=operator.add(x73, x58)
        x75=self.conv2d25(x74)
        x76=self.batchnorm2d15(x75)
        x77=self.silu15(x76)
        x78=self.conv2d26(x77)
        x79=self.batchnorm2d16(x78)
        x80=self.silu16(x79)
        x81=self.adaptiveavgpool2d5(x80)
        x82=self.conv2d27(x81)
        x83=self.silu17(x82)
        x84=self.conv2d28(x83)
        x85=self.sigmoid5(x84)
        x86=operator.mul(x85, x80)
        x87=self.conv2d29(x86)
        x88=self.batchnorm2d17(x87)
        x89=self.conv2d30(x88)
        x90=self.batchnorm2d18(x89)
        x91=self.silu18(x90)
        x92=self.conv2d31(x91)
        x93=self.batchnorm2d19(x92)
        x94=self.silu19(x93)
        x95=self.adaptiveavgpool2d6(x94)
        x96=self.conv2d32(x95)
        x97=self.silu20(x96)
        x98=self.conv2d33(x97)
        x99=self.sigmoid6(x98)
        x100=operator.mul(x99, x94)
        x101=self.conv2d34(x100)
        x102=self.batchnorm2d20(x101)
        x103=stochastic_depth(x102, 0.07500000000000001, 'row', False)
        x104=operator.add(x103, x88)
        x105=self.conv2d35(x104)
        x106=self.batchnorm2d21(x105)
        x107=self.silu21(x106)
        x108=self.conv2d36(x107)
        x109=self.batchnorm2d22(x108)
        x110=self.silu22(x109)
        x111=self.adaptiveavgpool2d7(x110)
        x112=self.conv2d37(x111)
        x113=self.silu23(x112)
        x114=self.conv2d38(x113)
        x115=self.sigmoid7(x114)
        x116=operator.mul(x115, x110)
        x117=self.conv2d39(x116)
        x118=self.batchnorm2d23(x117)
        x119=stochastic_depth(x118, 0.08750000000000001, 'row', False)
        x120=operator.add(x119, x104)
        x121=self.conv2d40(x120)
        x122=self.batchnorm2d24(x121)
        x123=self.silu24(x122)
        x124=self.conv2d41(x123)
        x125=self.batchnorm2d25(x124)
        x126=self.silu25(x125)
        x127=self.adaptiveavgpool2d8(x126)
        x128=self.conv2d42(x127)
        x129=self.silu26(x128)
        x130=self.conv2d43(x129)
        x131=self.sigmoid8(x130)
        x132=operator.mul(x131, x126)
        x133=self.conv2d44(x132)
        x134=self.batchnorm2d26(x133)
        x135=self.conv2d45(x134)
        x136=self.batchnorm2d27(x135)
        x137=self.silu27(x136)
        x138=self.conv2d46(x137)
        x139=self.batchnorm2d28(x138)
        x140=self.silu28(x139)
        x141=self.adaptiveavgpool2d9(x140)
        x142=self.conv2d47(x141)
        x143=self.silu29(x142)
        x144=self.conv2d48(x143)
        x145=self.sigmoid9(x144)
        x146=operator.mul(x145, x140)
        x147=self.conv2d49(x146)
        x148=self.batchnorm2d29(x147)
        x149=stochastic_depth(x148, 0.1125, 'row', False)
        x150=operator.add(x149, x134)
        x151=self.conv2d50(x150)
        x152=self.batchnorm2d30(x151)
        x153=self.silu30(x152)
        x154=self.conv2d51(x153)
        x155=self.batchnorm2d31(x154)
        x156=self.silu31(x155)
        x157=self.adaptiveavgpool2d10(x156)
        x158=self.conv2d52(x157)
        x159=self.silu32(x158)
        x160=self.conv2d53(x159)
        x161=self.sigmoid10(x160)
        x162=operator.mul(x161, x156)
        x163=self.conv2d54(x162)
        x164=self.batchnorm2d32(x163)
        x165=stochastic_depth(x164, 0.125, 'row', False)
        x166=operator.add(x165, x150)
        x167=self.conv2d55(x166)
        x168=self.batchnorm2d33(x167)
        x169=self.silu33(x168)
        x170=self.conv2d56(x169)
        x171=self.batchnorm2d34(x170)
        x172=self.silu34(x171)
        x173=self.adaptiveavgpool2d11(x172)
        x174=self.conv2d57(x173)
        x175=self.silu35(x174)
        x176=self.conv2d58(x175)
        x177=self.sigmoid11(x176)
        x178=operator.mul(x177, x172)
        x179=self.conv2d59(x178)
        x180=self.batchnorm2d35(x179)
        x181=self.conv2d60(x180)
        x182=self.batchnorm2d36(x181)
        x183=self.silu36(x182)
        x184=self.conv2d61(x183)
        x185=self.batchnorm2d37(x184)
        x186=self.silu37(x185)
        x187=self.adaptiveavgpool2d12(x186)
        x188=self.conv2d62(x187)
        x189=self.silu38(x188)
        x190=self.conv2d63(x189)
        x191=self.sigmoid12(x190)
        x192=operator.mul(x191, x186)
        x193=self.conv2d64(x192)
        x194=self.batchnorm2d38(x193)
        x195=stochastic_depth(x194, 0.15000000000000002, 'row', False)
        x196=operator.add(x195, x180)
        x197=self.conv2d65(x196)
        x198=self.batchnorm2d39(x197)
        x199=self.silu39(x198)
        x200=self.conv2d66(x199)
        x201=self.batchnorm2d40(x200)
        x202=self.silu40(x201)
        x203=self.adaptiveavgpool2d13(x202)
        x204=self.conv2d67(x203)
        x205=self.silu41(x204)
        x206=self.conv2d68(x205)
        x207=self.sigmoid13(x206)
        x208=operator.mul(x207, x202)
        x209=self.conv2d69(x208)
        x210=self.batchnorm2d41(x209)
        x211=stochastic_depth(x210, 0.1625, 'row', False)
        x212=operator.add(x211, x196)
        x213=self.conv2d70(x212)
        x214=self.batchnorm2d42(x213)
        x215=self.silu42(x214)
        x216=self.conv2d71(x215)
        x217=self.batchnorm2d43(x216)
        x218=self.silu43(x217)
        x219=self.adaptiveavgpool2d14(x218)
        x220=self.conv2d72(x219)
        x221=self.silu44(x220)
        x222=self.conv2d73(x221)
        x223=self.sigmoid14(x222)
        x224=operator.mul(x223, x218)
        x225=self.conv2d74(x224)
        x226=self.batchnorm2d44(x225)
        x227=stochastic_depth(x226, 0.17500000000000002, 'row', False)
        x228=operator.add(x227, x212)
        x229=self.conv2d75(x228)
        x230=self.batchnorm2d45(x229)
        x231=self.silu45(x230)
        x232=self.conv2d76(x231)
        x233=self.batchnorm2d46(x232)
        x234=self.silu46(x233)
        x235=self.adaptiveavgpool2d15(x234)
        x236=self.conv2d77(x235)
        x237=self.silu47(x236)
        x238=self.conv2d78(x237)
        x239=self.sigmoid15(x238)
        x240=operator.mul(x239, x234)
        x241=self.conv2d79(x240)
        x242=self.batchnorm2d47(x241)
        x243=self.conv2d80(x242)
        x244=self.batchnorm2d48(x243)
        x245=self.silu48(x244)
        x246=self.adaptiveavgpool2d16(x245)
        x247=torch.flatten(x246, 1)
        x248=self.dropout0(x247)
        x249=self.linear0(x248)

m = M().eval()
x = torch.rand(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
