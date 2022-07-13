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
        self.conv2d0 = Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d1 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d4 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d7 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d8 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d10 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d13 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d15 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d18 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d21 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d24 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d27 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d30 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d33 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d36 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d39 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d41 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d44 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d47 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d50 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d53 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(192, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.linear0 = Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        x4=self.maxpool2d0(x3)
        x5=self.conv2d1(x4)
        x6=self.batchnorm2d1(x5)
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        x9=self.relu1(x8)
        x10=self.conv2d3(x4)
        x11=self.batchnorm2d3(x10)
        x12=self.relu2(x11)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d4(x13)
        x15=self.conv2d5(x14)
        x16=self.batchnorm2d5(x15)
        x17=self.relu3(x16)
        x18=torch.cat((x9, x17),dim=1)
        x19=x18.size()
        x20=operator.getitem(x19, 0)
        x21=operator.getitem(x19, 1)
        x22=operator.getitem(x19, 2)
        x23=operator.getitem(x19, 3)
        x24=operator.floordiv(x21, 2)
        x25=x18.view(x20, 2, x24, x22, x23)
        x26=torch.transpose(x25, 1, 2)
        x27=x26.contiguous()
        x28=x27.view(x20, -1, x22, x23)
        x29=x28.chunk(2,dim=1)
        x30=operator.getitem(x29, 0)
        x31=operator.getitem(x29, 1)
        x32=self.conv2d6(x31)
        x33=self.batchnorm2d6(x32)
        x34=self.relu4(x33)
        x35=self.conv2d7(x34)
        x36=self.batchnorm2d7(x35)
        x37=self.conv2d8(x36)
        x38=self.batchnorm2d8(x37)
        x39=self.relu5(x38)
        x40=torch.cat((x30, x39),dim=1)
        x41=x40.size()
        x42=operator.getitem(x41, 0)
        x43=operator.getitem(x41, 1)
        x44=operator.getitem(x41, 2)
        x45=operator.getitem(x41, 3)
        x46=operator.floordiv(x43, 2)
        x47=x40.view(x42, 2, x46, x44, x45)
        x48=torch.transpose(x47, 1, 2)
        x49=x48.contiguous()
        x50=x49.view(x42, -1, x44, x45)
        x51=x50.chunk(2,dim=1)
        x52=operator.getitem(x51, 0)
        x53=operator.getitem(x51, 1)
        x54=self.conv2d9(x53)
        x55=self.batchnorm2d9(x54)
        x56=self.relu6(x55)
        x57=self.conv2d10(x56)
        x58=self.batchnorm2d10(x57)
        x59=self.conv2d11(x58)
        x60=self.batchnorm2d11(x59)
        x61=self.relu7(x60)
        x62=torch.cat((x52, x61),dim=1)
        x63=x62.size()
        x64=operator.getitem(x63, 0)
        x65=operator.getitem(x63, 1)
        x66=operator.getitem(x63, 2)
        x67=operator.getitem(x63, 3)
        x68=operator.floordiv(x65, 2)
        x69=x62.view(x64, 2, x68, x66, x67)
        x70=torch.transpose(x69, 1, 2)
        x71=x70.contiguous()
        x72=x71.view(x64, -1, x66, x67)
        x73=x72.chunk(2,dim=1)
        x74=operator.getitem(x73, 0)
        x75=operator.getitem(x73, 1)
        x76=self.conv2d12(x75)
        x77=self.batchnorm2d12(x76)
        x78=self.relu8(x77)
        x79=self.conv2d13(x78)
        x80=self.batchnorm2d13(x79)
        x81=self.conv2d14(x80)
        x82=self.batchnorm2d14(x81)
        x83=self.relu9(x82)
        x84=torch.cat((x74, x83),dim=1)
        x85=x84.size()
        x86=operator.getitem(x85, 0)
        x87=operator.getitem(x85, 1)
        x88=operator.getitem(x85, 2)
        x89=operator.getitem(x85, 3)
        x90=operator.floordiv(x87, 2)
        x91=x84.view(x86, 2, x90, x88, x89)
        x92=torch.transpose(x91, 1, 2)
        x93=x92.contiguous()
        x94=x93.view(x86, -1, x88, x89)
        x95=self.conv2d15(x94)
        x96=self.batchnorm2d15(x95)
        x97=self.conv2d16(x96)
        x98=self.batchnorm2d16(x97)
        x99=self.relu10(x98)
        x100=self.conv2d17(x94)
        x101=self.batchnorm2d17(x100)
        x102=self.relu11(x101)
        x103=self.conv2d18(x102)
        x104=self.batchnorm2d18(x103)
        x105=self.conv2d19(x104)
        x106=self.batchnorm2d19(x105)
        x107=self.relu12(x106)
        x108=torch.cat((x99, x107),dim=1)
        x109=x108.size()
        x110=operator.getitem(x109, 0)
        x111=operator.getitem(x109, 1)
        x112=operator.getitem(x109, 2)
        x113=operator.getitem(x109, 3)
        x114=operator.floordiv(x111, 2)
        x115=x108.view(x110, 2, x114, x112, x113)
        x116=torch.transpose(x115, 1, 2)
        x117=x116.contiguous()
        x118=x117.view(x110, -1, x112, x113)
        x119=x118.chunk(2,dim=1)
        x120=operator.getitem(x119, 0)
        x121=operator.getitem(x119, 1)
        x122=self.conv2d20(x121)
        x123=self.batchnorm2d20(x122)
        x124=self.relu13(x123)
        x125=self.conv2d21(x124)
        x126=self.batchnorm2d21(x125)
        x127=self.conv2d22(x126)
        x128=self.batchnorm2d22(x127)
        x129=self.relu14(x128)
        x130=torch.cat((x120, x129),dim=1)
        x131=x130.size()
        x132=operator.getitem(x131, 0)
        x133=operator.getitem(x131, 1)
        x134=operator.getitem(x131, 2)
        x135=operator.getitem(x131, 3)
        x136=operator.floordiv(x133, 2)
        x137=x130.view(x132, 2, x136, x134, x135)
        x138=torch.transpose(x137, 1, 2)
        x139=x138.contiguous()
        x140=x139.view(x132, -1, x134, x135)
        x141=x140.chunk(2,dim=1)
        x142=operator.getitem(x141, 0)
        x143=operator.getitem(x141, 1)
        x144=self.conv2d23(x143)
        x145=self.batchnorm2d23(x144)
        x146=self.relu15(x145)
        x147=self.conv2d24(x146)
        x148=self.batchnorm2d24(x147)
        x149=self.conv2d25(x148)
        x150=self.batchnorm2d25(x149)
        x151=self.relu16(x150)
        x152=torch.cat((x142, x151),dim=1)
        x153=x152.size()
        x154=operator.getitem(x153, 0)
        x155=operator.getitem(x153, 1)
        x156=operator.getitem(x153, 2)
        x157=operator.getitem(x153, 3)
        x158=operator.floordiv(x155, 2)
        x159=x152.view(x154, 2, x158, x156, x157)
        x160=torch.transpose(x159, 1, 2)
        x161=x160.contiguous()
        x162=x161.view(x154, -1, x156, x157)
        x163=x162.chunk(2,dim=1)
        x164=operator.getitem(x163, 0)
        x165=operator.getitem(x163, 1)
        x166=self.conv2d26(x165)
        x167=self.batchnorm2d26(x166)
        x168=self.relu17(x167)
        x169=self.conv2d27(x168)
        x170=self.batchnorm2d27(x169)
        x171=self.conv2d28(x170)
        x172=self.batchnorm2d28(x171)
        x173=self.relu18(x172)
        x174=torch.cat((x164, x173),dim=1)
        x175=x174.size()
        x176=operator.getitem(x175, 0)
        x177=operator.getitem(x175, 1)
        x178=operator.getitem(x175, 2)
        x179=operator.getitem(x175, 3)
        x180=operator.floordiv(x177, 2)
        x181=x174.view(x176, 2, x180, x178, x179)
        x182=torch.transpose(x181, 1, 2)
        x183=x182.contiguous()
        x184=x183.view(x176, -1, x178, x179)
        x185=x184.chunk(2,dim=1)
        x186=operator.getitem(x185, 0)
        x187=operator.getitem(x185, 1)
        x188=self.conv2d29(x187)
        x189=self.batchnorm2d29(x188)
        x190=self.relu19(x189)
        x191=self.conv2d30(x190)
        x192=self.batchnorm2d30(x191)
        x193=self.conv2d31(x192)
        x194=self.batchnorm2d31(x193)
        x195=self.relu20(x194)
        x196=torch.cat((x186, x195),dim=1)
        x197=x196.size()
        x198=operator.getitem(x197, 0)
        x199=operator.getitem(x197, 1)
        x200=operator.getitem(x197, 2)
        x201=operator.getitem(x197, 3)
        x202=operator.floordiv(x199, 2)
        x203=x196.view(x198, 2, x202, x200, x201)
        x204=torch.transpose(x203, 1, 2)
        x205=x204.contiguous()
        x206=x205.view(x198, -1, x200, x201)
        x207=x206.chunk(2,dim=1)
        x208=operator.getitem(x207, 0)
        x209=operator.getitem(x207, 1)
        x210=self.conv2d32(x209)
        x211=self.batchnorm2d32(x210)
        x212=self.relu21(x211)
        x213=self.conv2d33(x212)
        x214=self.batchnorm2d33(x213)
        x215=self.conv2d34(x214)
        x216=self.batchnorm2d34(x215)
        x217=self.relu22(x216)
        x218=torch.cat((x208, x217),dim=1)
        x219=x218.size()
        x220=operator.getitem(x219, 0)
        x221=operator.getitem(x219, 1)
        x222=operator.getitem(x219, 2)
        x223=operator.getitem(x219, 3)
        x224=operator.floordiv(x221, 2)
        x225=x218.view(x220, 2, x224, x222, x223)
        x226=torch.transpose(x225, 1, 2)
        x227=x226.contiguous()
        x228=x227.view(x220, -1, x222, x223)
        x229=x228.chunk(2,dim=1)
        x230=operator.getitem(x229, 0)
        x231=operator.getitem(x229, 1)
        x232=self.conv2d35(x231)
        x233=self.batchnorm2d35(x232)
        x234=self.relu23(x233)
        x235=self.conv2d36(x234)
        x236=self.batchnorm2d36(x235)
        x237=self.conv2d37(x236)
        x238=self.batchnorm2d37(x237)
        x239=self.relu24(x238)
        x240=torch.cat((x230, x239),dim=1)
        x241=x240.size()
        x242=operator.getitem(x241, 0)
        x243=operator.getitem(x241, 1)
        x244=operator.getitem(x241, 2)
        x245=operator.getitem(x241, 3)
        x246=operator.floordiv(x243, 2)
        x247=x240.view(x242, 2, x246, x244, x245)
        x248=torch.transpose(x247, 1, 2)
        x249=x248.contiguous()
        x250=x249.view(x242, -1, x244, x245)
        x251=x250.chunk(2,dim=1)
        x252=operator.getitem(x251, 0)
        x253=operator.getitem(x251, 1)
        x254=self.conv2d38(x253)
        x255=self.batchnorm2d38(x254)
        x256=self.relu25(x255)
        x257=self.conv2d39(x256)
        x258=self.batchnorm2d39(x257)
        x259=self.conv2d40(x258)
        x260=self.batchnorm2d40(x259)
        x261=self.relu26(x260)
        x262=torch.cat((x252, x261),dim=1)
        x263=x262.size()
        x264=operator.getitem(x263, 0)
        x265=operator.getitem(x263, 1)
        x266=operator.getitem(x263, 2)
        x267=operator.getitem(x263, 3)
        x268=operator.floordiv(x265, 2)
        x269=x262.view(x264, 2, x268, x266, x267)
        x270=torch.transpose(x269, 1, 2)
        x271=x270.contiguous()
        x272=x271.view(x264, -1, x266, x267)
        x273=self.conv2d41(x272)
        x274=self.batchnorm2d41(x273)
        x275=self.conv2d42(x274)
        x276=self.batchnorm2d42(x275)
        x277=self.relu27(x276)
        x278=self.conv2d43(x272)
        x279=self.batchnorm2d43(x278)
        x280=self.relu28(x279)
        x281=self.conv2d44(x280)
        x282=self.batchnorm2d44(x281)
        x283=self.conv2d45(x282)
        x284=self.batchnorm2d45(x283)
        x285=self.relu29(x284)
        x286=torch.cat((x277, x285),dim=1)
        x287=x286.size()
        x288=operator.getitem(x287, 0)
        x289=operator.getitem(x287, 1)
        x290=operator.getitem(x287, 2)
        x291=operator.getitem(x287, 3)
        x292=operator.floordiv(x289, 2)
        x293=x286.view(x288, 2, x292, x290, x291)
        x294=torch.transpose(x293, 1, 2)
        x295=x294.contiguous()
        x296=x295.view(x288, -1, x290, x291)
        x297=x296.chunk(2,dim=1)
        x298=operator.getitem(x297, 0)
        x299=operator.getitem(x297, 1)
        x300=self.conv2d46(x299)
        x301=self.batchnorm2d46(x300)
        x302=self.relu30(x301)
        x303=self.conv2d47(x302)
        x304=self.batchnorm2d47(x303)
        x305=self.conv2d48(x304)
        x306=self.batchnorm2d48(x305)
        x307=self.relu31(x306)
        x308=torch.cat((x298, x307),dim=1)
        x309=x308.size()
        x310=operator.getitem(x309, 0)
        x311=operator.getitem(x309, 1)
        x312=operator.getitem(x309, 2)
        x313=operator.getitem(x309, 3)
        x314=operator.floordiv(x311, 2)
        x315=x308.view(x310, 2, x314, x312, x313)
        x316=torch.transpose(x315, 1, 2)
        x317=x316.contiguous()
        x318=x317.view(x310, -1, x312, x313)
        x319=x318.chunk(2,dim=1)
        x320=operator.getitem(x319, 0)
        x321=operator.getitem(x319, 1)
        x322=self.conv2d49(x321)
        x323=self.batchnorm2d49(x322)
        x324=self.relu32(x323)
        x325=self.conv2d50(x324)
        x326=self.batchnorm2d50(x325)
        x327=self.conv2d51(x326)
        x328=self.batchnorm2d51(x327)
        x329=self.relu33(x328)
        x330=torch.cat((x320, x329),dim=1)
        x331=x330.size()
        x332=operator.getitem(x331, 0)
        x333=operator.getitem(x331, 1)
        x334=operator.getitem(x331, 2)
        x335=operator.getitem(x331, 3)
        x336=operator.floordiv(x333, 2)
        x337=x330.view(x332, 2, x336, x334, x335)
        x338=torch.transpose(x337, 1, 2)
        x339=x338.contiguous()
        x340=x339.view(x332, -1, x334, x335)
        x341=x340.chunk(2,dim=1)
        x342=operator.getitem(x341, 0)
        x343=operator.getitem(x341, 1)
        x344=self.conv2d52(x343)
        x345=self.batchnorm2d52(x344)
        x346=self.relu34(x345)
        x347=self.conv2d53(x346)
        x348=self.batchnorm2d53(x347)
        x349=self.conv2d54(x348)
        x350=self.batchnorm2d54(x349)
        x351=self.relu35(x350)
        x352=torch.cat((x342, x351),dim=1)
        x353=x352.size()
        x354=operator.getitem(x353, 0)
        x355=operator.getitem(x353, 1)
        x356=operator.getitem(x353, 2)
        x357=operator.getitem(x353, 3)
        x358=operator.floordiv(x355, 2)
        x359=x352.view(x354, 2, x358, x356, x357)
        x360=torch.transpose(x359, 1, 2)
        x361=x360.contiguous()
        x362=x361.view(x354, -1, x356, x357)
        x363=self.conv2d55(x362)
        x364=self.batchnorm2d55(x363)
        x365=self.relu36(x364)
        x366=x365.mean([2, 3])
        x367=self.linear0(x366)
        return [x367]

m = M().eval()
x = torch.randn(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
