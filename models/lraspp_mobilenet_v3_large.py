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
        self.conv2d62 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d63 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.sigmoid0 = Sigmoid()
        self.conv2d64 = Conv2d(40, 21, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d65 = Conv2d(128, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, input):
        x0=input
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.hardswish0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu0(x5)
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        x9=operator.add(x8, x3)
        x10=self.conv2d3(x9)
        x11=self.batchnorm2d3(x10)
        x12=self.relu1(x11)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d4(x13)
        x15=self.relu2(x14)
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        x18=self.conv2d6(x17)
        x19=self.batchnorm2d6(x18)
        x20=self.relu3(x19)
        x21=self.conv2d7(x20)
        x22=self.batchnorm2d7(x21)
        x23=self.relu4(x22)
        x24=self.conv2d8(x23)
        x25=self.batchnorm2d8(x24)
        x26=operator.add(x25, x17)
        x27=self.conv2d9(x26)
        x28=self.batchnorm2d9(x27)
        x29=self.relu5(x28)
        x30=self.conv2d10(x29)
        x31=self.batchnorm2d10(x30)
        x32=self.relu6(x31)
        x33=self.adaptiveavgpool2d0(x32)
        x34=self.conv2d11(x33)
        x35=self.relu7(x34)
        x36=self.conv2d12(x35)
        x37=self.hardsigmoid0(x36)
        x38=operator.mul(x37, x32)
        x39=self.conv2d13(x38)
        x40=self.batchnorm2d11(x39)
        x41=self.conv2d14(x40)
        x42=self.batchnorm2d12(x41)
        x43=self.relu8(x42)
        x44=self.conv2d15(x43)
        x45=self.batchnorm2d13(x44)
        x46=self.relu9(x45)
        x47=self.adaptiveavgpool2d1(x46)
        x48=self.conv2d16(x47)
        x49=self.relu10(x48)
        x50=self.conv2d17(x49)
        x51=self.hardsigmoid1(x50)
        x52=operator.mul(x51, x46)
        x53=self.conv2d18(x52)
        x54=self.batchnorm2d14(x53)
        x55=operator.add(x54, x40)
        x56=self.conv2d19(x55)
        x57=self.batchnorm2d15(x56)
        x58=self.relu11(x57)
        x59=self.conv2d20(x58)
        x60=self.batchnorm2d16(x59)
        x61=self.relu12(x60)
        x62=self.adaptiveavgpool2d2(x61)
        x63=self.conv2d21(x62)
        x64=self.relu13(x63)
        x65=self.conv2d22(x64)
        x66=self.hardsigmoid2(x65)
        x67=operator.mul(x66, x61)
        x68=self.conv2d23(x67)
        x69=self.batchnorm2d17(x68)
        x70=operator.add(x69, x55)
        x71=self.conv2d24(x70)
        x72=self.batchnorm2d18(x71)
        x73=self.hardswish1(x72)
        x74=self.conv2d25(x73)
        x75=self.batchnorm2d19(x74)
        x76=self.hardswish2(x75)
        x77=self.conv2d26(x76)
        x78=self.batchnorm2d20(x77)
        x79=self.conv2d27(x78)
        x80=self.batchnorm2d21(x79)
        x81=self.hardswish3(x80)
        x82=self.conv2d28(x81)
        x83=self.batchnorm2d22(x82)
        x84=self.hardswish4(x83)
        x85=self.conv2d29(x84)
        x86=self.batchnorm2d23(x85)
        x87=operator.add(x86, x78)
        x88=self.conv2d30(x87)
        x89=self.batchnorm2d24(x88)
        x90=self.hardswish5(x89)
        x91=self.conv2d31(x90)
        x92=self.batchnorm2d25(x91)
        x93=self.hardswish6(x92)
        x94=self.conv2d32(x93)
        x95=self.batchnorm2d26(x94)
        x96=operator.add(x95, x87)
        x97=self.conv2d33(x96)
        x98=self.batchnorm2d27(x97)
        x99=self.hardswish7(x98)
        x100=self.conv2d34(x99)
        x101=self.batchnorm2d28(x100)
        x102=self.hardswish8(x101)
        x103=self.conv2d35(x102)
        x104=self.batchnorm2d29(x103)
        x105=operator.add(x104, x96)
        x106=self.conv2d36(x105)
        x107=self.batchnorm2d30(x106)
        x108=self.hardswish9(x107)
        x109=self.conv2d37(x108)
        x110=self.batchnorm2d31(x109)
        x111=self.hardswish10(x110)
        x112=self.adaptiveavgpool2d3(x111)
        x113=self.conv2d38(x112)
        x114=self.relu14(x113)
        x115=self.conv2d39(x114)
        x116=self.hardsigmoid3(x115)
        x117=operator.mul(x116, x111)
        x118=self.conv2d40(x117)
        x119=self.batchnorm2d32(x118)
        x120=self.conv2d41(x119)
        x121=self.batchnorm2d33(x120)
        x122=self.hardswish11(x121)
        x123=self.conv2d42(x122)
        x124=self.batchnorm2d34(x123)
        x125=self.hardswish12(x124)
        x126=self.adaptiveavgpool2d4(x125)
        x127=self.conv2d43(x126)
        x128=self.relu15(x127)
        x129=self.conv2d44(x128)
        x130=self.hardsigmoid4(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d45(x131)
        x133=self.batchnorm2d35(x132)
        x134=operator.add(x133, x119)
        x135=self.conv2d46(x134)
        x136=self.batchnorm2d36(x135)
        x137=self.hardswish13(x136)
        x138=self.conv2d47(x137)
        x139=self.batchnorm2d37(x138)
        x140=self.hardswish14(x139)
        x141=self.adaptiveavgpool2d5(x140)
        x142=self.conv2d48(x141)
        x143=self.relu16(x142)
        x144=self.conv2d49(x143)
        x145=self.hardsigmoid5(x144)
        x146=operator.mul(x145, x140)
        x147=self.conv2d50(x146)
        x148=self.batchnorm2d38(x147)
        x149=self.conv2d51(x148)
        x150=self.batchnorm2d39(x149)
        x151=self.hardswish15(x150)
        x152=self.conv2d52(x151)
        x153=self.batchnorm2d40(x152)
        x154=self.hardswish16(x153)
        x155=self.adaptiveavgpool2d6(x154)
        x156=self.conv2d53(x155)
        x157=self.relu17(x156)
        x158=self.conv2d54(x157)
        x159=self.hardsigmoid6(x158)
        x160=operator.mul(x159, x154)
        x161=self.conv2d55(x160)
        x162=self.batchnorm2d41(x161)
        x163=operator.add(x162, x148)
        x164=self.conv2d56(x163)
        x165=self.batchnorm2d42(x164)
        x166=self.hardswish17(x165)
        x167=self.conv2d57(x166)
        x168=self.batchnorm2d43(x167)
        x169=self.hardswish18(x168)
        x170=self.adaptiveavgpool2d7(x169)
        x171=self.conv2d58(x170)
        x172=self.relu18(x171)
        x173=self.conv2d59(x172)
        x174=self.hardsigmoid7(x173)
        x175=operator.mul(x174, x169)
        x176=self.conv2d60(x175)
        x177=self.batchnorm2d44(x176)
        x178=operator.add(x177, x163)
        x179=self.conv2d61(x178)
        x180=self.batchnorm2d45(x179)
        x181=self.hardswish19(x180)
        x182=self.conv2d62(x181)
        x183=self.batchnorm2d46(x182)
        x184=self.relu19(x183)
        x185=self.adaptiveavgpool2d8(x181)
        x186=self.conv2d63(x185)
        x187=self.sigmoid0(x186)
        x188=operator.mul(x184, x187)
        x189=builtins.getattr(x40, 'shape')
        x190=operator.getitem(x189, slice(-2, None, None))
        x191=torch.nn.functional.interpolate(x188,size=x190, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)
        x192=self.conv2d64(x40)
        x193=self.conv2d65(x191)
        x194=operator.add(x192, x193)
        x195=builtins.getattr(x0, 'shape')
        x196=operator.getitem(x195, slice(-2, None, None))
        x197=torch.nn.functional.interpolate(x194,size=x196, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)

m = M().eval()
input = torch.rand(1, 3, 224, 224)
start = time.time()
output = m(input)
end = time.time()
print(end-start)
