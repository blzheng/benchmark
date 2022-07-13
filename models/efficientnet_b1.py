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
        self.conv2d5 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d3 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu3 = SiLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d6 = Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))
        self.silu4 = SiLU(inplace=True)
        self.conv2d7 = Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d8 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu5 = SiLU(inplace=True)
        self.conv2d10 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d6 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu6 = SiLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d11 = Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
        self.silu7 = SiLU(inplace=True)
        self.conv2d12 = Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d13 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu8 = SiLU(inplace=True)
        self.conv2d15 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu9 = SiLU(inplace=True)
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d16 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu10 = SiLU(inplace=True)
        self.conv2d17 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d18 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu11 = SiLU(inplace=True)
        self.conv2d20 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d12 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu12 = SiLU(inplace=True)
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d21 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu13 = SiLU(inplace=True)
        self.conv2d22 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d23 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu14 = SiLU(inplace=True)
        self.conv2d25 = Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
        self.batchnorm2d15 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu15 = SiLU(inplace=True)
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d26 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu16 = SiLU(inplace=True)
        self.conv2d27 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d28 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu17 = SiLU(inplace=True)
        self.conv2d30 = Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        self.batchnorm2d18 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu18 = SiLU(inplace=True)
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d31 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu19 = SiLU(inplace=True)
        self.conv2d32 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d33 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu20 = SiLU(inplace=True)
        self.conv2d35 = Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
        self.batchnorm2d21 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu21 = SiLU(inplace=True)
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d36 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu22 = SiLU(inplace=True)
        self.conv2d37 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d38 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu23 = SiLU(inplace=True)
        self.conv2d40 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d24 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu24 = SiLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d41 = Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu25 = SiLU(inplace=True)
        self.conv2d42 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d43 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu26 = SiLU(inplace=True)
        self.conv2d45 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d27 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu27 = SiLU(inplace=True)
        self.adaptiveavgpool2d9 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d46 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu28 = SiLU(inplace=True)
        self.conv2d47 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d48 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu29 = SiLU(inplace=True)
        self.conv2d50 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d30 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu30 = SiLU(inplace=True)
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d51 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu31 = SiLU(inplace=True)
        self.conv2d52 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d53 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu32 = SiLU(inplace=True)
        self.conv2d55 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d33 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu33 = SiLU(inplace=True)
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d56 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu34 = SiLU(inplace=True)
        self.conv2d57 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d58 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d59 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu35 = SiLU(inplace=True)
        self.conv2d60 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d36 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu36 = SiLU(inplace=True)
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d61 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu37 = SiLU(inplace=True)
        self.conv2d62 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d63 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu38 = SiLU(inplace=True)
        self.conv2d65 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d39 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu39 = SiLU(inplace=True)
        self.adaptiveavgpool2d13 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d66 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu40 = SiLU(inplace=True)
        self.conv2d67 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d68 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d69 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu41 = SiLU(inplace=True)
        self.conv2d70 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d42 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu42 = SiLU(inplace=True)
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d71 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu43 = SiLU(inplace=True)
        self.conv2d72 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d73 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d74 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu44 = SiLU(inplace=True)
        self.conv2d75 = Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d45 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu45 = SiLU(inplace=True)
        self.adaptiveavgpool2d15 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d76 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu46 = SiLU(inplace=True)
        self.conv2d77 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d78 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d79 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu47 = SiLU(inplace=True)
        self.conv2d80 = Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        self.batchnorm2d48 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu48 = SiLU(inplace=True)
        self.adaptiveavgpool2d16 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d81 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
        self.silu49 = SiLU(inplace=True)
        self.conv2d82 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d83 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu50 = SiLU(inplace=True)
        self.conv2d85 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d51 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu51 = SiLU(inplace=True)
        self.adaptiveavgpool2d17 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d86 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu52 = SiLU(inplace=True)
        self.conv2d87 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d88 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d89 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu53 = SiLU(inplace=True)
        self.conv2d90 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d54 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu54 = SiLU(inplace=True)
        self.adaptiveavgpool2d18 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d91 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu55 = SiLU(inplace=True)
        self.conv2d92 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d93 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu56 = SiLU(inplace=True)
        self.conv2d95 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d57 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu57 = SiLU(inplace=True)
        self.adaptiveavgpool2d19 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d96 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu58 = SiLU(inplace=True)
        self.conv2d97 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d98 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d99 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu59 = SiLU(inplace=True)
        self.conv2d100 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
        self.batchnorm2d60 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu60 = SiLU(inplace=True)
        self.adaptiveavgpool2d20 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d101 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu61 = SiLU(inplace=True)
        self.conv2d102 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d103 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d104 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu62 = SiLU(inplace=True)
        self.conv2d105 = Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
        self.batchnorm2d63 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu63 = SiLU(inplace=True)
        self.adaptiveavgpool2d21 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d106 = Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
        self.silu64 = SiLU(inplace=True)
        self.conv2d107 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d108 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d109 = Conv2d(320, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu65 = SiLU(inplace=True)
        self.conv2d110 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1920, bias=False)
        self.batchnorm2d66 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu66 = SiLU(inplace=True)
        self.adaptiveavgpool2d22 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d111 = Conv2d(1920, 80, kernel_size=(1, 1), stride=(1, 1))
        self.silu67 = SiLU(inplace=True)
        self.conv2d112 = Conv2d(80, 1920, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d113 = Conv2d(1920, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu68 = SiLU(inplace=True)
        self.adaptiveavgpool2d23 = AdaptiveAvgPool2d(output_size=1)
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
        x18=self.adaptiveavgpool2d1(x17)
        x19=self.conv2d6(x18)
        x20=self.silu4(x19)
        x21=self.conv2d7(x20)
        x22=self.sigmoid1(x21)
        x23=operator.mul(x22, x17)
        x24=self.conv2d8(x23)
        x25=self.batchnorm2d4(x24)
        x26=stochastic_depth(x25, 0.008695652173913044, 'row', False)
        x27=operator.add(x26, x14)
        x28=self.conv2d9(x27)
        x29=self.batchnorm2d5(x28)
        x30=self.silu5(x29)
        x31=self.conv2d10(x30)
        x32=self.batchnorm2d6(x31)
        x33=self.silu6(x32)
        x34=self.adaptiveavgpool2d2(x33)
        x35=self.conv2d11(x34)
        x36=self.silu7(x35)
        x37=self.conv2d12(x36)
        x38=self.sigmoid2(x37)
        x39=operator.mul(x38, x33)
        x40=self.conv2d13(x39)
        x41=self.batchnorm2d7(x40)
        x42=self.conv2d14(x41)
        x43=self.batchnorm2d8(x42)
        x44=self.silu8(x43)
        x45=self.conv2d15(x44)
        x46=self.batchnorm2d9(x45)
        x47=self.silu9(x46)
        x48=self.adaptiveavgpool2d3(x47)
        x49=self.conv2d16(x48)
        x50=self.silu10(x49)
        x51=self.conv2d17(x50)
        x52=self.sigmoid3(x51)
        x53=operator.mul(x52, x47)
        x54=self.conv2d18(x53)
        x55=self.batchnorm2d10(x54)
        x56=stochastic_depth(x55, 0.026086956521739136, 'row', False)
        x57=operator.add(x56, x41)
        x58=self.conv2d19(x57)
        x59=self.batchnorm2d11(x58)
        x60=self.silu11(x59)
        x61=self.conv2d20(x60)
        x62=self.batchnorm2d12(x61)
        x63=self.silu12(x62)
        x64=self.adaptiveavgpool2d4(x63)
        x65=self.conv2d21(x64)
        x66=self.silu13(x65)
        x67=self.conv2d22(x66)
        x68=self.sigmoid4(x67)
        x69=operator.mul(x68, x63)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d13(x70)
        x72=stochastic_depth(x71, 0.034782608695652174, 'row', False)
        x73=operator.add(x72, x57)
        x74=self.conv2d24(x73)
        x75=self.batchnorm2d14(x74)
        x76=self.silu14(x75)
        x77=self.conv2d25(x76)
        x78=self.batchnorm2d15(x77)
        x79=self.silu15(x78)
        x80=self.adaptiveavgpool2d5(x79)
        x81=self.conv2d26(x80)
        x82=self.silu16(x81)
        x83=self.conv2d27(x82)
        x84=self.sigmoid5(x83)
        x85=operator.mul(x84, x79)
        x86=self.conv2d28(x85)
        x87=self.batchnorm2d16(x86)
        x88=self.conv2d29(x87)
        x89=self.batchnorm2d17(x88)
        x90=self.silu17(x89)
        x91=self.conv2d30(x90)
        x92=self.batchnorm2d18(x91)
        x93=self.silu18(x92)
        x94=self.adaptiveavgpool2d6(x93)
        x95=self.conv2d31(x94)
        x96=self.silu19(x95)
        x97=self.conv2d32(x96)
        x98=self.sigmoid6(x97)
        x99=operator.mul(x98, x93)
        x100=self.conv2d33(x99)
        x101=self.batchnorm2d19(x100)
        x102=stochastic_depth(x101, 0.05217391304347827, 'row', False)
        x103=operator.add(x102, x87)
        x104=self.conv2d34(x103)
        x105=self.batchnorm2d20(x104)
        x106=self.silu20(x105)
        x107=self.conv2d35(x106)
        x108=self.batchnorm2d21(x107)
        x109=self.silu21(x108)
        x110=self.adaptiveavgpool2d7(x109)
        x111=self.conv2d36(x110)
        x112=self.silu22(x111)
        x113=self.conv2d37(x112)
        x114=self.sigmoid7(x113)
        x115=operator.mul(x114, x109)
        x116=self.conv2d38(x115)
        x117=self.batchnorm2d22(x116)
        x118=stochastic_depth(x117, 0.06086956521739131, 'row', False)
        x119=operator.add(x118, x103)
        x120=self.conv2d39(x119)
        x121=self.batchnorm2d23(x120)
        x122=self.silu23(x121)
        x123=self.conv2d40(x122)
        x124=self.batchnorm2d24(x123)
        x125=self.silu24(x124)
        x126=self.adaptiveavgpool2d8(x125)
        x127=self.conv2d41(x126)
        x128=self.silu25(x127)
        x129=self.conv2d42(x128)
        x130=self.sigmoid8(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d43(x131)
        x133=self.batchnorm2d25(x132)
        x134=self.conv2d44(x133)
        x135=self.batchnorm2d26(x134)
        x136=self.silu26(x135)
        x137=self.conv2d45(x136)
        x138=self.batchnorm2d27(x137)
        x139=self.silu27(x138)
        x140=self.adaptiveavgpool2d9(x139)
        x141=self.conv2d46(x140)
        x142=self.silu28(x141)
        x143=self.conv2d47(x142)
        x144=self.sigmoid9(x143)
        x145=operator.mul(x144, x139)
        x146=self.conv2d48(x145)
        x147=self.batchnorm2d28(x146)
        x148=stochastic_depth(x147, 0.0782608695652174, 'row', False)
        x149=operator.add(x148, x133)
        x150=self.conv2d49(x149)
        x151=self.batchnorm2d29(x150)
        x152=self.silu29(x151)
        x153=self.conv2d50(x152)
        x154=self.batchnorm2d30(x153)
        x155=self.silu30(x154)
        x156=self.adaptiveavgpool2d10(x155)
        x157=self.conv2d51(x156)
        x158=self.silu31(x157)
        x159=self.conv2d52(x158)
        x160=self.sigmoid10(x159)
        x161=operator.mul(x160, x155)
        x162=self.conv2d53(x161)
        x163=self.batchnorm2d31(x162)
        x164=stochastic_depth(x163, 0.08695652173913043, 'row', False)
        x165=operator.add(x164, x149)
        x166=self.conv2d54(x165)
        x167=self.batchnorm2d32(x166)
        x168=self.silu32(x167)
        x169=self.conv2d55(x168)
        x170=self.batchnorm2d33(x169)
        x171=self.silu33(x170)
        x172=self.adaptiveavgpool2d11(x171)
        x173=self.conv2d56(x172)
        x174=self.silu34(x173)
        x175=self.conv2d57(x174)
        x176=self.sigmoid11(x175)
        x177=operator.mul(x176, x171)
        x178=self.conv2d58(x177)
        x179=self.batchnorm2d34(x178)
        x180=stochastic_depth(x179, 0.09565217391304348, 'row', False)
        x181=operator.add(x180, x165)
        x182=self.conv2d59(x181)
        x183=self.batchnorm2d35(x182)
        x184=self.silu35(x183)
        x185=self.conv2d60(x184)
        x186=self.batchnorm2d36(x185)
        x187=self.silu36(x186)
        x188=self.adaptiveavgpool2d12(x187)
        x189=self.conv2d61(x188)
        x190=self.silu37(x189)
        x191=self.conv2d62(x190)
        x192=self.sigmoid12(x191)
        x193=operator.mul(x192, x187)
        x194=self.conv2d63(x193)
        x195=self.batchnorm2d37(x194)
        x196=self.conv2d64(x195)
        x197=self.batchnorm2d38(x196)
        x198=self.silu38(x197)
        x199=self.conv2d65(x198)
        x200=self.batchnorm2d39(x199)
        x201=self.silu39(x200)
        x202=self.adaptiveavgpool2d13(x201)
        x203=self.conv2d66(x202)
        x204=self.silu40(x203)
        x205=self.conv2d67(x204)
        x206=self.sigmoid13(x205)
        x207=operator.mul(x206, x201)
        x208=self.conv2d68(x207)
        x209=self.batchnorm2d40(x208)
        x210=stochastic_depth(x209, 0.11304347826086956, 'row', False)
        x211=operator.add(x210, x195)
        x212=self.conv2d69(x211)
        x213=self.batchnorm2d41(x212)
        x214=self.silu41(x213)
        x215=self.conv2d70(x214)
        x216=self.batchnorm2d42(x215)
        x217=self.silu42(x216)
        x218=self.adaptiveavgpool2d14(x217)
        x219=self.conv2d71(x218)
        x220=self.silu43(x219)
        x221=self.conv2d72(x220)
        x222=self.sigmoid14(x221)
        x223=operator.mul(x222, x217)
        x224=self.conv2d73(x223)
        x225=self.batchnorm2d43(x224)
        x226=stochastic_depth(x225, 0.12173913043478261, 'row', False)
        x227=operator.add(x226, x211)
        x228=self.conv2d74(x227)
        x229=self.batchnorm2d44(x228)
        x230=self.silu44(x229)
        x231=self.conv2d75(x230)
        x232=self.batchnorm2d45(x231)
        x233=self.silu45(x232)
        x234=self.adaptiveavgpool2d15(x233)
        x235=self.conv2d76(x234)
        x236=self.silu46(x235)
        x237=self.conv2d77(x236)
        x238=self.sigmoid15(x237)
        x239=operator.mul(x238, x233)
        x240=self.conv2d78(x239)
        x241=self.batchnorm2d46(x240)
        x242=stochastic_depth(x241, 0.13043478260869565, 'row', False)
        x243=operator.add(x242, x227)
        x244=self.conv2d79(x243)
        x245=self.batchnorm2d47(x244)
        x246=self.silu47(x245)
        x247=self.conv2d80(x246)
        x248=self.batchnorm2d48(x247)
        x249=self.silu48(x248)
        x250=self.adaptiveavgpool2d16(x249)
        x251=self.conv2d81(x250)
        x252=self.silu49(x251)
        x253=self.conv2d82(x252)
        x254=self.sigmoid16(x253)
        x255=operator.mul(x254, x249)
        x256=self.conv2d83(x255)
        x257=self.batchnorm2d49(x256)
        x258=self.conv2d84(x257)
        x259=self.batchnorm2d50(x258)
        x260=self.silu50(x259)
        x261=self.conv2d85(x260)
        x262=self.batchnorm2d51(x261)
        x263=self.silu51(x262)
        x264=self.adaptiveavgpool2d17(x263)
        x265=self.conv2d86(x264)
        x266=self.silu52(x265)
        x267=self.conv2d87(x266)
        x268=self.sigmoid17(x267)
        x269=operator.mul(x268, x263)
        x270=self.conv2d88(x269)
        x271=self.batchnorm2d52(x270)
        x272=stochastic_depth(x271, 0.14782608695652175, 'row', False)
        x273=operator.add(x272, x257)
        x274=self.conv2d89(x273)
        x275=self.batchnorm2d53(x274)
        x276=self.silu53(x275)
        x277=self.conv2d90(x276)
        x278=self.batchnorm2d54(x277)
        x279=self.silu54(x278)
        x280=self.adaptiveavgpool2d18(x279)
        x281=self.conv2d91(x280)
        x282=self.silu55(x281)
        x283=self.conv2d92(x282)
        x284=self.sigmoid18(x283)
        x285=operator.mul(x284, x279)
        x286=self.conv2d93(x285)
        x287=self.batchnorm2d55(x286)
        x288=stochastic_depth(x287, 0.1565217391304348, 'row', False)
        x289=operator.add(x288, x273)
        x290=self.conv2d94(x289)
        x291=self.batchnorm2d56(x290)
        x292=self.silu56(x291)
        x293=self.conv2d95(x292)
        x294=self.batchnorm2d57(x293)
        x295=self.silu57(x294)
        x296=self.adaptiveavgpool2d19(x295)
        x297=self.conv2d96(x296)
        x298=self.silu58(x297)
        x299=self.conv2d97(x298)
        x300=self.sigmoid19(x299)
        x301=operator.mul(x300, x295)
        x302=self.conv2d98(x301)
        x303=self.batchnorm2d58(x302)
        x304=stochastic_depth(x303, 0.16521739130434784, 'row', False)
        x305=operator.add(x304, x289)
        x306=self.conv2d99(x305)
        x307=self.batchnorm2d59(x306)
        x308=self.silu59(x307)
        x309=self.conv2d100(x308)
        x310=self.batchnorm2d60(x309)
        x311=self.silu60(x310)
        x312=self.adaptiveavgpool2d20(x311)
        x313=self.conv2d101(x312)
        x314=self.silu61(x313)
        x315=self.conv2d102(x314)
        x316=self.sigmoid20(x315)
        x317=operator.mul(x316, x311)
        x318=self.conv2d103(x317)
        x319=self.batchnorm2d61(x318)
        x320=stochastic_depth(x319, 0.17391304347826086, 'row', False)
        x321=operator.add(x320, x305)
        x322=self.conv2d104(x321)
        x323=self.batchnorm2d62(x322)
        x324=self.silu62(x323)
        x325=self.conv2d105(x324)
        x326=self.batchnorm2d63(x325)
        x327=self.silu63(x326)
        x328=self.adaptiveavgpool2d21(x327)
        x329=self.conv2d106(x328)
        x330=self.silu64(x329)
        x331=self.conv2d107(x330)
        x332=self.sigmoid21(x331)
        x333=operator.mul(x332, x327)
        x334=self.conv2d108(x333)
        x335=self.batchnorm2d64(x334)
        x336=self.conv2d109(x335)
        x337=self.batchnorm2d65(x336)
        x338=self.silu65(x337)
        x339=self.conv2d110(x338)
        x340=self.batchnorm2d66(x339)
        x341=self.silu66(x340)
        x342=self.adaptiveavgpool2d22(x341)
        x343=self.conv2d111(x342)
        x344=self.silu67(x343)
        x345=self.conv2d112(x344)
        x346=self.sigmoid22(x345)
        x347=operator.mul(x346, x341)
        x348=self.conv2d113(x347)
        x349=self.batchnorm2d67(x348)
        x350=stochastic_depth(x349, 0.19130434782608696, 'row', False)
        x351=operator.add(x350, x335)
        x352=self.conv2d114(x351)
        x353=self.batchnorm2d68(x352)
        x354=self.silu68(x353)
        x355=self.adaptiveavgpool2d23(x354)
        x356=torch.flatten(x355, 1)
        x357=self.dropout0(x356)
        x358=self.linear0(x357)
        return [x358]

m = M().eval()
x = torch.randn(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
