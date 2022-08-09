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
        self.conv2d0 = Conv2d(3, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu0 = SiLU(inplace=True)
        self.conv2d1 = Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
        self.batchnorm2d1 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu1 = SiLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d2 = Conv2d(40, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu2 = SiLU(inplace=True)
        self.conv2d3 = Conv2d(10, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d4 = Conv2d(40, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d3 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu3 = SiLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d6 = Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu4 = SiLU(inplace=True)
        self.conv2d7 = Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d8 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu5 = SiLU(inplace=True)
        self.conv2d10 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu6 = SiLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d11 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu7 = SiLU(inplace=True)
        self.conv2d12 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d13 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu8 = SiLU(inplace=True)
        self.conv2d15 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d9 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu9 = SiLU(inplace=True)
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d16 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu10 = SiLU(inplace=True)
        self.conv2d17 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d18 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu11 = SiLU(inplace=True)
        self.conv2d20 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d12 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu12 = SiLU(inplace=True)
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d21 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu13 = SiLU(inplace=True)
        self.conv2d22 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d23 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu14 = SiLU(inplace=True)
        self.conv2d25 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu15 = SiLU(inplace=True)
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d26 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu16 = SiLU(inplace=True)
        self.conv2d27 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d28 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu17 = SiLU(inplace=True)
        self.conv2d30 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d18 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu18 = SiLU(inplace=True)
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d31 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu19 = SiLU(inplace=True)
        self.conv2d32 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d33 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu20 = SiLU(inplace=True)
        self.conv2d35 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d21 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu21 = SiLU(inplace=True)
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d36 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu22 = SiLU(inplace=True)
        self.conv2d37 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d38 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu23 = SiLU(inplace=True)
        self.conv2d40 = Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu24 = SiLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d41 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu25 = SiLU(inplace=True)
        self.conv2d42 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d43 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu26 = SiLU(inplace=True)
        self.conv2d45 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d27 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu27 = SiLU(inplace=True)
        self.adaptiveavgpool2d9 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d46 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu28 = SiLU(inplace=True)
        self.conv2d47 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d48 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu29 = SiLU(inplace=True)
        self.conv2d50 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d30 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu30 = SiLU(inplace=True)
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d51 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu31 = SiLU(inplace=True)
        self.conv2d52 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d53 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu32 = SiLU(inplace=True)
        self.conv2d55 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d33 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu33 = SiLU(inplace=True)
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d56 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu34 = SiLU(inplace=True)
        self.conv2d57 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d58 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d59 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu35 = SiLU(inplace=True)
        self.conv2d60 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d36 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu36 = SiLU(inplace=True)
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d61 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu37 = SiLU(inplace=True)
        self.conv2d62 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d63 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu38 = SiLU(inplace=True)
        self.conv2d65 = Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d39 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu39 = SiLU(inplace=True)
        self.adaptiveavgpool2d13 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d66 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu40 = SiLU(inplace=True)
        self.conv2d67 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d68 = Conv2d(576, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d69 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu41 = SiLU(inplace=True)
        self.conv2d70 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d42 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu42 = SiLU(inplace=True)
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d71 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu43 = SiLU(inplace=True)
        self.conv2d72 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d73 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d74 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu44 = SiLU(inplace=True)
        self.conv2d75 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d45 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu45 = SiLU(inplace=True)
        self.adaptiveavgpool2d15 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d76 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu46 = SiLU(inplace=True)
        self.conv2d77 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d78 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d79 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu47 = SiLU(inplace=True)
        self.conv2d80 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d48 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu48 = SiLU(inplace=True)
        self.adaptiveavgpool2d16 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d81 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu49 = SiLU(inplace=True)
        self.conv2d82 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d83 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu50 = SiLU(inplace=True)
        self.conv2d85 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d51 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu51 = SiLU(inplace=True)
        self.adaptiveavgpool2d17 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d86 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu52 = SiLU(inplace=True)
        self.conv2d87 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d88 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d89 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu53 = SiLU(inplace=True)
        self.conv2d90 = Conv2d(816, 816, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d54 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu54 = SiLU(inplace=True)
        self.adaptiveavgpool2d18 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d91 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu55 = SiLU(inplace=True)
        self.conv2d92 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu56 = SiLU(inplace=True)
        self.conv2d95 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d57 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu57 = SiLU(inplace=True)
        self.adaptiveavgpool2d19 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d96 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu58 = SiLU(inplace=True)
        self.conv2d97 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d98 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d99 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu59 = SiLU(inplace=True)
        self.conv2d100 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d60 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu60 = SiLU(inplace=True)
        self.adaptiveavgpool2d20 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d101 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu61 = SiLU(inplace=True)
        self.conv2d102 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d103 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d104 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu62 = SiLU(inplace=True)
        self.conv2d105 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d63 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu63 = SiLU(inplace=True)
        self.adaptiveavgpool2d21 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d106 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu64 = SiLU(inplace=True)
        self.conv2d107 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d108 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d109 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu65 = SiLU(inplace=True)
        self.conv2d110 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d66 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu66 = SiLU(inplace=True)
        self.adaptiveavgpool2d22 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d111 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu67 = SiLU(inplace=True)
        self.conv2d112 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d113 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu68 = SiLU(inplace=True)
        self.conv2d115 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d69 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu69 = SiLU(inplace=True)
        self.adaptiveavgpool2d23 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d116 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu70 = SiLU(inplace=True)
        self.conv2d117 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()
        self.conv2d118 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d119 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu71 = SiLU(inplace=True)
        self.conv2d120 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1392, bias=False)
        self.batchnorm2d72 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu72 = SiLU(inplace=True)
        self.adaptiveavgpool2d24 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d121 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu73 = SiLU(inplace=True)
        self.conv2d122 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d124 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu74 = SiLU(inplace=True)
        self.conv2d125 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
        self.batchnorm2d75 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu75 = SiLU(inplace=True)
        self.adaptiveavgpool2d25 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d126 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu76 = SiLU(inplace=True)
        self.conv2d127 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d128 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d129 = Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu77 = SiLU(inplace=True)
        self.adaptiveavgpool2d26 = AdaptiveAvgPool2d(output_size=1)
        self.dropout0 = Dropout(p=0.3, inplace=True)
        self.linear0 = Linear(in_features=1536, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        if x0 is None:
            print('x0: {}'.format(x0))
        elif isinstance(x0, torch.Tensor):
            print('x0: {}'.format(x0.shape))
        elif isinstance(x0, tuple):
            tuple_shapes = '('
            for item in x0:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x0: {}'.format(tuple_shapes))
        else:
            print('x0: {}'.format(x0))
        x1=self.conv2d0(x0)
        if x1 is None:
            print('x1: {}'.format(x1))
        elif isinstance(x1, torch.Tensor):
            print('x1: {}'.format(x1.shape))
        elif isinstance(x1, tuple):
            tuple_shapes = '('
            for item in x1:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x1: {}'.format(tuple_shapes))
        else:
            print('x1: {}'.format(x1))
        x2=self.batchnorm2d0(x1)
        if x2 is None:
            print('x2: {}'.format(x2))
        elif isinstance(x2, torch.Tensor):
            print('x2: {}'.format(x2.shape))
        elif isinstance(x2, tuple):
            tuple_shapes = '('
            for item in x2:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x2: {}'.format(tuple_shapes))
        else:
            print('x2: {}'.format(x2))
        x3=self.silu0(x2)
        if x3 is None:
            print('x3: {}'.format(x3))
        elif isinstance(x3, torch.Tensor):
            print('x3: {}'.format(x3.shape))
        elif isinstance(x3, tuple):
            tuple_shapes = '('
            for item in x3:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x3: {}'.format(tuple_shapes))
        else:
            print('x3: {}'.format(x3))
        x4=self.conv2d1(x3)
        if x4 is None:
            print('x4: {}'.format(x4))
        elif isinstance(x4, torch.Tensor):
            print('x4: {}'.format(x4.shape))
        elif isinstance(x4, tuple):
            tuple_shapes = '('
            for item in x4:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x4: {}'.format(tuple_shapes))
        else:
            print('x4: {}'.format(x4))
        x5=self.batchnorm2d1(x4)
        if x5 is None:
            print('x5: {}'.format(x5))
        elif isinstance(x5, torch.Tensor):
            print('x5: {}'.format(x5.shape))
        elif isinstance(x5, tuple):
            tuple_shapes = '('
            for item in x5:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x5: {}'.format(tuple_shapes))
        else:
            print('x5: {}'.format(x5))
        x6=self.silu1(x5)
        if x6 is None:
            print('x6: {}'.format(x6))
        elif isinstance(x6, torch.Tensor):
            print('x6: {}'.format(x6.shape))
        elif isinstance(x6, tuple):
            tuple_shapes = '('
            for item in x6:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x6: {}'.format(tuple_shapes))
        else:
            print('x6: {}'.format(x6))
        x7=self.adaptiveavgpool2d0(x6)
        if x7 is None:
            print('x7: {}'.format(x7))
        elif isinstance(x7, torch.Tensor):
            print('x7: {}'.format(x7.shape))
        elif isinstance(x7, tuple):
            tuple_shapes = '('
            for item in x7:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x7: {}'.format(tuple_shapes))
        else:
            print('x7: {}'.format(x7))
        x8=self.conv2d2(x7)
        if x8 is None:
            print('x8: {}'.format(x8))
        elif isinstance(x8, torch.Tensor):
            print('x8: {}'.format(x8.shape))
        elif isinstance(x8, tuple):
            tuple_shapes = '('
            for item in x8:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x8: {}'.format(tuple_shapes))
        else:
            print('x8: {}'.format(x8))
        x9=self.silu2(x8)
        if x9 is None:
            print('x9: {}'.format(x9))
        elif isinstance(x9, torch.Tensor):
            print('x9: {}'.format(x9.shape))
        elif isinstance(x9, tuple):
            tuple_shapes = '('
            for item in x9:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x9: {}'.format(tuple_shapes))
        else:
            print('x9: {}'.format(x9))
        x10=self.conv2d3(x9)
        if x10 is None:
            print('x10: {}'.format(x10))
        elif isinstance(x10, torch.Tensor):
            print('x10: {}'.format(x10.shape))
        elif isinstance(x10, tuple):
            tuple_shapes = '('
            for item in x10:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x10: {}'.format(tuple_shapes))
        else:
            print('x10: {}'.format(x10))
        x11=self.sigmoid0(x10)
        if x11 is None:
            print('x11: {}'.format(x11))
        elif isinstance(x11, torch.Tensor):
            print('x11: {}'.format(x11.shape))
        elif isinstance(x11, tuple):
            tuple_shapes = '('
            for item in x11:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x11: {}'.format(tuple_shapes))
        else:
            print('x11: {}'.format(x11))
        x12=operator.mul(x11, x6)
        if x12 is None:
            print('x12: {}'.format(x12))
        elif isinstance(x12, torch.Tensor):
            print('x12: {}'.format(x12.shape))
        elif isinstance(x12, tuple):
            tuple_shapes = '('
            for item in x12:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x12: {}'.format(tuple_shapes))
        else:
            print('x12: {}'.format(x12))
        x13=self.conv2d4(x12)
        if x13 is None:
            print('x13: {}'.format(x13))
        elif isinstance(x13, torch.Tensor):
            print('x13: {}'.format(x13.shape))
        elif isinstance(x13, tuple):
            tuple_shapes = '('
            for item in x13:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x13: {}'.format(tuple_shapes))
        else:
            print('x13: {}'.format(x13))
        x14=self.batchnorm2d2(x13)
        if x14 is None:
            print('x14: {}'.format(x14))
        elif isinstance(x14, torch.Tensor):
            print('x14: {}'.format(x14.shape))
        elif isinstance(x14, tuple):
            tuple_shapes = '('
            for item in x14:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x14: {}'.format(tuple_shapes))
        else:
            print('x14: {}'.format(x14))
        x15=self.conv2d5(x14)
        if x15 is None:
            print('x15: {}'.format(x15))
        elif isinstance(x15, torch.Tensor):
            print('x15: {}'.format(x15.shape))
        elif isinstance(x15, tuple):
            tuple_shapes = '('
            for item in x15:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x15: {}'.format(tuple_shapes))
        else:
            print('x15: {}'.format(x15))
        x16=self.batchnorm2d3(x15)
        if x16 is None:
            print('x16: {}'.format(x16))
        elif isinstance(x16, torch.Tensor):
            print('x16: {}'.format(x16.shape))
        elif isinstance(x16, tuple):
            tuple_shapes = '('
            for item in x16:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x16: {}'.format(tuple_shapes))
        else:
            print('x16: {}'.format(x16))
        x17=self.silu3(x16)
        if x17 is None:
            print('x17: {}'.format(x17))
        elif isinstance(x17, torch.Tensor):
            print('x17: {}'.format(x17.shape))
        elif isinstance(x17, tuple):
            tuple_shapes = '('
            for item in x17:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x17: {}'.format(tuple_shapes))
        else:
            print('x17: {}'.format(x17))
        x18=self.adaptiveavgpool2d1(x17)
        if x18 is None:
            print('x18: {}'.format(x18))
        elif isinstance(x18, torch.Tensor):
            print('x18: {}'.format(x18.shape))
        elif isinstance(x18, tuple):
            tuple_shapes = '('
            for item in x18:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x18: {}'.format(tuple_shapes))
        else:
            print('x18: {}'.format(x18))
        x19=self.conv2d6(x18)
        if x19 is None:
            print('x19: {}'.format(x19))
        elif isinstance(x19, torch.Tensor):
            print('x19: {}'.format(x19.shape))
        elif isinstance(x19, tuple):
            tuple_shapes = '('
            for item in x19:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x19: {}'.format(tuple_shapes))
        else:
            print('x19: {}'.format(x19))
        x20=self.silu4(x19)
        if x20 is None:
            print('x20: {}'.format(x20))
        elif isinstance(x20, torch.Tensor):
            print('x20: {}'.format(x20.shape))
        elif isinstance(x20, tuple):
            tuple_shapes = '('
            for item in x20:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x20: {}'.format(tuple_shapes))
        else:
            print('x20: {}'.format(x20))
        x21=self.conv2d7(x20)
        if x21 is None:
            print('x21: {}'.format(x21))
        elif isinstance(x21, torch.Tensor):
            print('x21: {}'.format(x21.shape))
        elif isinstance(x21, tuple):
            tuple_shapes = '('
            for item in x21:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x21: {}'.format(tuple_shapes))
        else:
            print('x21: {}'.format(x21))
        x22=self.sigmoid1(x21)
        if x22 is None:
            print('x22: {}'.format(x22))
        elif isinstance(x22, torch.Tensor):
            print('x22: {}'.format(x22.shape))
        elif isinstance(x22, tuple):
            tuple_shapes = '('
            for item in x22:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x22: {}'.format(tuple_shapes))
        else:
            print('x22: {}'.format(x22))
        x23=operator.mul(x22, x17)
        if x23 is None:
            print('x23: {}'.format(x23))
        elif isinstance(x23, torch.Tensor):
            print('x23: {}'.format(x23.shape))
        elif isinstance(x23, tuple):
            tuple_shapes = '('
            for item in x23:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x23: {}'.format(tuple_shapes))
        else:
            print('x23: {}'.format(x23))
        x24=self.conv2d8(x23)
        if x24 is None:
            print('x24: {}'.format(x24))
        elif isinstance(x24, torch.Tensor):
            print('x24: {}'.format(x24.shape))
        elif isinstance(x24, tuple):
            tuple_shapes = '('
            for item in x24:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x24: {}'.format(tuple_shapes))
        else:
            print('x24: {}'.format(x24))
        x25=self.batchnorm2d4(x24)
        if x25 is None:
            print('x25: {}'.format(x25))
        elif isinstance(x25, torch.Tensor):
            print('x25: {}'.format(x25.shape))
        elif isinstance(x25, tuple):
            tuple_shapes = '('
            for item in x25:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x25: {}'.format(tuple_shapes))
        else:
            print('x25: {}'.format(x25))
        x26=stochastic_depth(x25, 0.007692307692307693, 'row', False)
        if x26 is None:
            print('x26: {}'.format(x26))
        elif isinstance(x26, torch.Tensor):
            print('x26: {}'.format(x26.shape))
        elif isinstance(x26, tuple):
            tuple_shapes = '('
            for item in x26:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x26: {}'.format(tuple_shapes))
        else:
            print('x26: {}'.format(x26))
        x27=operator.add(x26, x14)
        if x27 is None:
            print('x27: {}'.format(x27))
        elif isinstance(x27, torch.Tensor):
            print('x27: {}'.format(x27.shape))
        elif isinstance(x27, tuple):
            tuple_shapes = '('
            for item in x27:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x27: {}'.format(tuple_shapes))
        else:
            print('x27: {}'.format(x27))
        x28=self.conv2d9(x27)
        if x28 is None:
            print('x28: {}'.format(x28))
        elif isinstance(x28, torch.Tensor):
            print('x28: {}'.format(x28.shape))
        elif isinstance(x28, tuple):
            tuple_shapes = '('
            for item in x28:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x28: {}'.format(tuple_shapes))
        else:
            print('x28: {}'.format(x28))
        x29=self.batchnorm2d5(x28)
        if x29 is None:
            print('x29: {}'.format(x29))
        elif isinstance(x29, torch.Tensor):
            print('x29: {}'.format(x29.shape))
        elif isinstance(x29, tuple):
            tuple_shapes = '('
            for item in x29:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x29: {}'.format(tuple_shapes))
        else:
            print('x29: {}'.format(x29))
        x30=self.silu5(x29)
        if x30 is None:
            print('x30: {}'.format(x30))
        elif isinstance(x30, torch.Tensor):
            print('x30: {}'.format(x30.shape))
        elif isinstance(x30, tuple):
            tuple_shapes = '('
            for item in x30:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x30: {}'.format(tuple_shapes))
        else:
            print('x30: {}'.format(x30))
        x31=self.conv2d10(x30)
        if x31 is None:
            print('x31: {}'.format(x31))
        elif isinstance(x31, torch.Tensor):
            print('x31: {}'.format(x31.shape))
        elif isinstance(x31, tuple):
            tuple_shapes = '('
            for item in x31:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x31: {}'.format(tuple_shapes))
        else:
            print('x31: {}'.format(x31))
        x32=self.batchnorm2d6(x31)
        if x32 is None:
            print('x32: {}'.format(x32))
        elif isinstance(x32, torch.Tensor):
            print('x32: {}'.format(x32.shape))
        elif isinstance(x32, tuple):
            tuple_shapes = '('
            for item in x32:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x32: {}'.format(tuple_shapes))
        else:
            print('x32: {}'.format(x32))
        x33=self.silu6(x32)
        if x33 is None:
            print('x33: {}'.format(x33))
        elif isinstance(x33, torch.Tensor):
            print('x33: {}'.format(x33.shape))
        elif isinstance(x33, tuple):
            tuple_shapes = '('
            for item in x33:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x33: {}'.format(tuple_shapes))
        else:
            print('x33: {}'.format(x33))
        x34=self.adaptiveavgpool2d2(x33)
        if x34 is None:
            print('x34: {}'.format(x34))
        elif isinstance(x34, torch.Tensor):
            print('x34: {}'.format(x34.shape))
        elif isinstance(x34, tuple):
            tuple_shapes = '('
            for item in x34:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x34: {}'.format(tuple_shapes))
        else:
            print('x34: {}'.format(x34))
        x35=self.conv2d11(x34)
        if x35 is None:
            print('x35: {}'.format(x35))
        elif isinstance(x35, torch.Tensor):
            print('x35: {}'.format(x35.shape))
        elif isinstance(x35, tuple):
            tuple_shapes = '('
            for item in x35:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x35: {}'.format(tuple_shapes))
        else:
            print('x35: {}'.format(x35))
        x36=self.silu7(x35)
        if x36 is None:
            print('x36: {}'.format(x36))
        elif isinstance(x36, torch.Tensor):
            print('x36: {}'.format(x36.shape))
        elif isinstance(x36, tuple):
            tuple_shapes = '('
            for item in x36:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x36: {}'.format(tuple_shapes))
        else:
            print('x36: {}'.format(x36))
        x37=self.conv2d12(x36)
        if x37 is None:
            print('x37: {}'.format(x37))
        elif isinstance(x37, torch.Tensor):
            print('x37: {}'.format(x37.shape))
        elif isinstance(x37, tuple):
            tuple_shapes = '('
            for item in x37:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x37: {}'.format(tuple_shapes))
        else:
            print('x37: {}'.format(x37))
        x38=self.sigmoid2(x37)
        if x38 is None:
            print('x38: {}'.format(x38))
        elif isinstance(x38, torch.Tensor):
            print('x38: {}'.format(x38.shape))
        elif isinstance(x38, tuple):
            tuple_shapes = '('
            for item in x38:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x38: {}'.format(tuple_shapes))
        else:
            print('x38: {}'.format(x38))
        x39=operator.mul(x38, x33)
        if x39 is None:
            print('x39: {}'.format(x39))
        elif isinstance(x39, torch.Tensor):
            print('x39: {}'.format(x39.shape))
        elif isinstance(x39, tuple):
            tuple_shapes = '('
            for item in x39:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x39: {}'.format(tuple_shapes))
        else:
            print('x39: {}'.format(x39))
        x40=self.conv2d13(x39)
        if x40 is None:
            print('x40: {}'.format(x40))
        elif isinstance(x40, torch.Tensor):
            print('x40: {}'.format(x40.shape))
        elif isinstance(x40, tuple):
            tuple_shapes = '('
            for item in x40:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x40: {}'.format(tuple_shapes))
        else:
            print('x40: {}'.format(x40))
        x41=self.batchnorm2d7(x40)
        if x41 is None:
            print('x41: {}'.format(x41))
        elif isinstance(x41, torch.Tensor):
            print('x41: {}'.format(x41.shape))
        elif isinstance(x41, tuple):
            tuple_shapes = '('
            for item in x41:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x41: {}'.format(tuple_shapes))
        else:
            print('x41: {}'.format(x41))
        x42=self.conv2d14(x41)
        if x42 is None:
            print('x42: {}'.format(x42))
        elif isinstance(x42, torch.Tensor):
            print('x42: {}'.format(x42.shape))
        elif isinstance(x42, tuple):
            tuple_shapes = '('
            for item in x42:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x42: {}'.format(tuple_shapes))
        else:
            print('x42: {}'.format(x42))
        x43=self.batchnorm2d8(x42)
        if x43 is None:
            print('x43: {}'.format(x43))
        elif isinstance(x43, torch.Tensor):
            print('x43: {}'.format(x43.shape))
        elif isinstance(x43, tuple):
            tuple_shapes = '('
            for item in x43:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x43: {}'.format(tuple_shapes))
        else:
            print('x43: {}'.format(x43))
        x44=self.silu8(x43)
        if x44 is None:
            print('x44: {}'.format(x44))
        elif isinstance(x44, torch.Tensor):
            print('x44: {}'.format(x44.shape))
        elif isinstance(x44, tuple):
            tuple_shapes = '('
            for item in x44:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x44: {}'.format(tuple_shapes))
        else:
            print('x44: {}'.format(x44))
        x45=self.conv2d15(x44)
        if x45 is None:
            print('x45: {}'.format(x45))
        elif isinstance(x45, torch.Tensor):
            print('x45: {}'.format(x45.shape))
        elif isinstance(x45, tuple):
            tuple_shapes = '('
            for item in x45:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x45: {}'.format(tuple_shapes))
        else:
            print('x45: {}'.format(x45))
        x46=self.batchnorm2d9(x45)
        if x46 is None:
            print('x46: {}'.format(x46))
        elif isinstance(x46, torch.Tensor):
            print('x46: {}'.format(x46.shape))
        elif isinstance(x46, tuple):
            tuple_shapes = '('
            for item in x46:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x46: {}'.format(tuple_shapes))
        else:
            print('x46: {}'.format(x46))
        x47=self.silu9(x46)
        if x47 is None:
            print('x47: {}'.format(x47))
        elif isinstance(x47, torch.Tensor):
            print('x47: {}'.format(x47.shape))
        elif isinstance(x47, tuple):
            tuple_shapes = '('
            for item in x47:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x47: {}'.format(tuple_shapes))
        else:
            print('x47: {}'.format(x47))
        x48=self.adaptiveavgpool2d3(x47)
        if x48 is None:
            print('x48: {}'.format(x48))
        elif isinstance(x48, torch.Tensor):
            print('x48: {}'.format(x48.shape))
        elif isinstance(x48, tuple):
            tuple_shapes = '('
            for item in x48:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x48: {}'.format(tuple_shapes))
        else:
            print('x48: {}'.format(x48))
        x49=self.conv2d16(x48)
        if x49 is None:
            print('x49: {}'.format(x49))
        elif isinstance(x49, torch.Tensor):
            print('x49: {}'.format(x49.shape))
        elif isinstance(x49, tuple):
            tuple_shapes = '('
            for item in x49:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x49: {}'.format(tuple_shapes))
        else:
            print('x49: {}'.format(x49))
        x50=self.silu10(x49)
        if x50 is None:
            print('x50: {}'.format(x50))
        elif isinstance(x50, torch.Tensor):
            print('x50: {}'.format(x50.shape))
        elif isinstance(x50, tuple):
            tuple_shapes = '('
            for item in x50:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x50: {}'.format(tuple_shapes))
        else:
            print('x50: {}'.format(x50))
        x51=self.conv2d17(x50)
        if x51 is None:
            print('x51: {}'.format(x51))
        elif isinstance(x51, torch.Tensor):
            print('x51: {}'.format(x51.shape))
        elif isinstance(x51, tuple):
            tuple_shapes = '('
            for item in x51:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x51: {}'.format(tuple_shapes))
        else:
            print('x51: {}'.format(x51))
        x52=self.sigmoid3(x51)
        if x52 is None:
            print('x52: {}'.format(x52))
        elif isinstance(x52, torch.Tensor):
            print('x52: {}'.format(x52.shape))
        elif isinstance(x52, tuple):
            tuple_shapes = '('
            for item in x52:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x52: {}'.format(tuple_shapes))
        else:
            print('x52: {}'.format(x52))
        x53=operator.mul(x52, x47)
        if x53 is None:
            print('x53: {}'.format(x53))
        elif isinstance(x53, torch.Tensor):
            print('x53: {}'.format(x53.shape))
        elif isinstance(x53, tuple):
            tuple_shapes = '('
            for item in x53:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x53: {}'.format(tuple_shapes))
        else:
            print('x53: {}'.format(x53))
        x54=self.conv2d18(x53)
        if x54 is None:
            print('x54: {}'.format(x54))
        elif isinstance(x54, torch.Tensor):
            print('x54: {}'.format(x54.shape))
        elif isinstance(x54, tuple):
            tuple_shapes = '('
            for item in x54:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x54: {}'.format(tuple_shapes))
        else:
            print('x54: {}'.format(x54))
        x55=self.batchnorm2d10(x54)
        if x55 is None:
            print('x55: {}'.format(x55))
        elif isinstance(x55, torch.Tensor):
            print('x55: {}'.format(x55.shape))
        elif isinstance(x55, tuple):
            tuple_shapes = '('
            for item in x55:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x55: {}'.format(tuple_shapes))
        else:
            print('x55: {}'.format(x55))
        x56=stochastic_depth(x55, 0.02307692307692308, 'row', False)
        if x56 is None:
            print('x56: {}'.format(x56))
        elif isinstance(x56, torch.Tensor):
            print('x56: {}'.format(x56.shape))
        elif isinstance(x56, tuple):
            tuple_shapes = '('
            for item in x56:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x56: {}'.format(tuple_shapes))
        else:
            print('x56: {}'.format(x56))
        x57=operator.add(x56, x41)
        if x57 is None:
            print('x57: {}'.format(x57))
        elif isinstance(x57, torch.Tensor):
            print('x57: {}'.format(x57.shape))
        elif isinstance(x57, tuple):
            tuple_shapes = '('
            for item in x57:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x57: {}'.format(tuple_shapes))
        else:
            print('x57: {}'.format(x57))
        x58=self.conv2d19(x57)
        if x58 is None:
            print('x58: {}'.format(x58))
        elif isinstance(x58, torch.Tensor):
            print('x58: {}'.format(x58.shape))
        elif isinstance(x58, tuple):
            tuple_shapes = '('
            for item in x58:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x58: {}'.format(tuple_shapes))
        else:
            print('x58: {}'.format(x58))
        x59=self.batchnorm2d11(x58)
        if x59 is None:
            print('x59: {}'.format(x59))
        elif isinstance(x59, torch.Tensor):
            print('x59: {}'.format(x59.shape))
        elif isinstance(x59, tuple):
            tuple_shapes = '('
            for item in x59:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x59: {}'.format(tuple_shapes))
        else:
            print('x59: {}'.format(x59))
        x60=self.silu11(x59)
        if x60 is None:
            print('x60: {}'.format(x60))
        elif isinstance(x60, torch.Tensor):
            print('x60: {}'.format(x60.shape))
        elif isinstance(x60, tuple):
            tuple_shapes = '('
            for item in x60:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x60: {}'.format(tuple_shapes))
        else:
            print('x60: {}'.format(x60))
        x61=self.conv2d20(x60)
        if x61 is None:
            print('x61: {}'.format(x61))
        elif isinstance(x61, torch.Tensor):
            print('x61: {}'.format(x61.shape))
        elif isinstance(x61, tuple):
            tuple_shapes = '('
            for item in x61:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x61: {}'.format(tuple_shapes))
        else:
            print('x61: {}'.format(x61))
        x62=self.batchnorm2d12(x61)
        if x62 is None:
            print('x62: {}'.format(x62))
        elif isinstance(x62, torch.Tensor):
            print('x62: {}'.format(x62.shape))
        elif isinstance(x62, tuple):
            tuple_shapes = '('
            for item in x62:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x62: {}'.format(tuple_shapes))
        else:
            print('x62: {}'.format(x62))
        x63=self.silu12(x62)
        if x63 is None:
            print('x63: {}'.format(x63))
        elif isinstance(x63, torch.Tensor):
            print('x63: {}'.format(x63.shape))
        elif isinstance(x63, tuple):
            tuple_shapes = '('
            for item in x63:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x63: {}'.format(tuple_shapes))
        else:
            print('x63: {}'.format(x63))
        x64=self.adaptiveavgpool2d4(x63)
        if x64 is None:
            print('x64: {}'.format(x64))
        elif isinstance(x64, torch.Tensor):
            print('x64: {}'.format(x64.shape))
        elif isinstance(x64, tuple):
            tuple_shapes = '('
            for item in x64:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x64: {}'.format(tuple_shapes))
        else:
            print('x64: {}'.format(x64))
        x65=self.conv2d21(x64)
        if x65 is None:
            print('x65: {}'.format(x65))
        elif isinstance(x65, torch.Tensor):
            print('x65: {}'.format(x65.shape))
        elif isinstance(x65, tuple):
            tuple_shapes = '('
            for item in x65:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x65: {}'.format(tuple_shapes))
        else:
            print('x65: {}'.format(x65))
        x66=self.silu13(x65)
        if x66 is None:
            print('x66: {}'.format(x66))
        elif isinstance(x66, torch.Tensor):
            print('x66: {}'.format(x66.shape))
        elif isinstance(x66, tuple):
            tuple_shapes = '('
            for item in x66:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x66: {}'.format(tuple_shapes))
        else:
            print('x66: {}'.format(x66))
        x67=self.conv2d22(x66)
        if x67 is None:
            print('x67: {}'.format(x67))
        elif isinstance(x67, torch.Tensor):
            print('x67: {}'.format(x67.shape))
        elif isinstance(x67, tuple):
            tuple_shapes = '('
            for item in x67:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x67: {}'.format(tuple_shapes))
        else:
            print('x67: {}'.format(x67))
        x68=self.sigmoid4(x67)
        if x68 is None:
            print('x68: {}'.format(x68))
        elif isinstance(x68, torch.Tensor):
            print('x68: {}'.format(x68.shape))
        elif isinstance(x68, tuple):
            tuple_shapes = '('
            for item in x68:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x68: {}'.format(tuple_shapes))
        else:
            print('x68: {}'.format(x68))
        x69=operator.mul(x68, x63)
        if x69 is None:
            print('x69: {}'.format(x69))
        elif isinstance(x69, torch.Tensor):
            print('x69: {}'.format(x69.shape))
        elif isinstance(x69, tuple):
            tuple_shapes = '('
            for item in x69:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x69: {}'.format(tuple_shapes))
        else:
            print('x69: {}'.format(x69))
        x70=self.conv2d23(x69)
        if x70 is None:
            print('x70: {}'.format(x70))
        elif isinstance(x70, torch.Tensor):
            print('x70: {}'.format(x70.shape))
        elif isinstance(x70, tuple):
            tuple_shapes = '('
            for item in x70:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x70: {}'.format(tuple_shapes))
        else:
            print('x70: {}'.format(x70))
        x71=self.batchnorm2d13(x70)
        if x71 is None:
            print('x71: {}'.format(x71))
        elif isinstance(x71, torch.Tensor):
            print('x71: {}'.format(x71.shape))
        elif isinstance(x71, tuple):
            tuple_shapes = '('
            for item in x71:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x71: {}'.format(tuple_shapes))
        else:
            print('x71: {}'.format(x71))
        x72=stochastic_depth(x71, 0.03076923076923077, 'row', False)
        if x72 is None:
            print('x72: {}'.format(x72))
        elif isinstance(x72, torch.Tensor):
            print('x72: {}'.format(x72.shape))
        elif isinstance(x72, tuple):
            tuple_shapes = '('
            for item in x72:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x72: {}'.format(tuple_shapes))
        else:
            print('x72: {}'.format(x72))
        x73=operator.add(x72, x57)
        if x73 is None:
            print('x73: {}'.format(x73))
        elif isinstance(x73, torch.Tensor):
            print('x73: {}'.format(x73.shape))
        elif isinstance(x73, tuple):
            tuple_shapes = '('
            for item in x73:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x73: {}'.format(tuple_shapes))
        else:
            print('x73: {}'.format(x73))
        x74=self.conv2d24(x73)
        if x74 is None:
            print('x74: {}'.format(x74))
        elif isinstance(x74, torch.Tensor):
            print('x74: {}'.format(x74.shape))
        elif isinstance(x74, tuple):
            tuple_shapes = '('
            for item in x74:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x74: {}'.format(tuple_shapes))
        else:
            print('x74: {}'.format(x74))
        x75=self.batchnorm2d14(x74)
        if x75 is None:
            print('x75: {}'.format(x75))
        elif isinstance(x75, torch.Tensor):
            print('x75: {}'.format(x75.shape))
        elif isinstance(x75, tuple):
            tuple_shapes = '('
            for item in x75:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x75: {}'.format(tuple_shapes))
        else:
            print('x75: {}'.format(x75))
        x76=self.silu14(x75)
        if x76 is None:
            print('x76: {}'.format(x76))
        elif isinstance(x76, torch.Tensor):
            print('x76: {}'.format(x76.shape))
        elif isinstance(x76, tuple):
            tuple_shapes = '('
            for item in x76:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x76: {}'.format(tuple_shapes))
        else:
            print('x76: {}'.format(x76))
        x77=self.conv2d25(x76)
        if x77 is None:
            print('x77: {}'.format(x77))
        elif isinstance(x77, torch.Tensor):
            print('x77: {}'.format(x77.shape))
        elif isinstance(x77, tuple):
            tuple_shapes = '('
            for item in x77:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x77: {}'.format(tuple_shapes))
        else:
            print('x77: {}'.format(x77))
        x78=self.batchnorm2d15(x77)
        if x78 is None:
            print('x78: {}'.format(x78))
        elif isinstance(x78, torch.Tensor):
            print('x78: {}'.format(x78.shape))
        elif isinstance(x78, tuple):
            tuple_shapes = '('
            for item in x78:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x78: {}'.format(tuple_shapes))
        else:
            print('x78: {}'.format(x78))
        x79=self.silu15(x78)
        if x79 is None:
            print('x79: {}'.format(x79))
        elif isinstance(x79, torch.Tensor):
            print('x79: {}'.format(x79.shape))
        elif isinstance(x79, tuple):
            tuple_shapes = '('
            for item in x79:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x79: {}'.format(tuple_shapes))
        else:
            print('x79: {}'.format(x79))
        x80=self.adaptiveavgpool2d5(x79)
        if x80 is None:
            print('x80: {}'.format(x80))
        elif isinstance(x80, torch.Tensor):
            print('x80: {}'.format(x80.shape))
        elif isinstance(x80, tuple):
            tuple_shapes = '('
            for item in x80:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x80: {}'.format(tuple_shapes))
        else:
            print('x80: {}'.format(x80))
        x81=self.conv2d26(x80)
        if x81 is None:
            print('x81: {}'.format(x81))
        elif isinstance(x81, torch.Tensor):
            print('x81: {}'.format(x81.shape))
        elif isinstance(x81, tuple):
            tuple_shapes = '('
            for item in x81:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x81: {}'.format(tuple_shapes))
        else:
            print('x81: {}'.format(x81))
        x82=self.silu16(x81)
        if x82 is None:
            print('x82: {}'.format(x82))
        elif isinstance(x82, torch.Tensor):
            print('x82: {}'.format(x82.shape))
        elif isinstance(x82, tuple):
            tuple_shapes = '('
            for item in x82:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x82: {}'.format(tuple_shapes))
        else:
            print('x82: {}'.format(x82))
        x83=self.conv2d27(x82)
        if x83 is None:
            print('x83: {}'.format(x83))
        elif isinstance(x83, torch.Tensor):
            print('x83: {}'.format(x83.shape))
        elif isinstance(x83, tuple):
            tuple_shapes = '('
            for item in x83:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x83: {}'.format(tuple_shapes))
        else:
            print('x83: {}'.format(x83))
        x84=self.sigmoid5(x83)
        if x84 is None:
            print('x84: {}'.format(x84))
        elif isinstance(x84, torch.Tensor):
            print('x84: {}'.format(x84.shape))
        elif isinstance(x84, tuple):
            tuple_shapes = '('
            for item in x84:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x84: {}'.format(tuple_shapes))
        else:
            print('x84: {}'.format(x84))
        x85=operator.mul(x84, x79)
        if x85 is None:
            print('x85: {}'.format(x85))
        elif isinstance(x85, torch.Tensor):
            print('x85: {}'.format(x85.shape))
        elif isinstance(x85, tuple):
            tuple_shapes = '('
            for item in x85:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x85: {}'.format(tuple_shapes))
        else:
            print('x85: {}'.format(x85))
        x86=self.conv2d28(x85)
        if x86 is None:
            print('x86: {}'.format(x86))
        elif isinstance(x86, torch.Tensor):
            print('x86: {}'.format(x86.shape))
        elif isinstance(x86, tuple):
            tuple_shapes = '('
            for item in x86:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x86: {}'.format(tuple_shapes))
        else:
            print('x86: {}'.format(x86))
        x87=self.batchnorm2d16(x86)
        if x87 is None:
            print('x87: {}'.format(x87))
        elif isinstance(x87, torch.Tensor):
            print('x87: {}'.format(x87.shape))
        elif isinstance(x87, tuple):
            tuple_shapes = '('
            for item in x87:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x87: {}'.format(tuple_shapes))
        else:
            print('x87: {}'.format(x87))
        x88=self.conv2d29(x87)
        if x88 is None:
            print('x88: {}'.format(x88))
        elif isinstance(x88, torch.Tensor):
            print('x88: {}'.format(x88.shape))
        elif isinstance(x88, tuple):
            tuple_shapes = '('
            for item in x88:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x88: {}'.format(tuple_shapes))
        else:
            print('x88: {}'.format(x88))
        x89=self.batchnorm2d17(x88)
        if x89 is None:
            print('x89: {}'.format(x89))
        elif isinstance(x89, torch.Tensor):
            print('x89: {}'.format(x89.shape))
        elif isinstance(x89, tuple):
            tuple_shapes = '('
            for item in x89:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x89: {}'.format(tuple_shapes))
        else:
            print('x89: {}'.format(x89))
        x90=self.silu17(x89)
        if x90 is None:
            print('x90: {}'.format(x90))
        elif isinstance(x90, torch.Tensor):
            print('x90: {}'.format(x90.shape))
        elif isinstance(x90, tuple):
            tuple_shapes = '('
            for item in x90:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x90: {}'.format(tuple_shapes))
        else:
            print('x90: {}'.format(x90))
        x91=self.conv2d30(x90)
        if x91 is None:
            print('x91: {}'.format(x91))
        elif isinstance(x91, torch.Tensor):
            print('x91: {}'.format(x91.shape))
        elif isinstance(x91, tuple):
            tuple_shapes = '('
            for item in x91:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x91: {}'.format(tuple_shapes))
        else:
            print('x91: {}'.format(x91))
        x92=self.batchnorm2d18(x91)
        if x92 is None:
            print('x92: {}'.format(x92))
        elif isinstance(x92, torch.Tensor):
            print('x92: {}'.format(x92.shape))
        elif isinstance(x92, tuple):
            tuple_shapes = '('
            for item in x92:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x92: {}'.format(tuple_shapes))
        else:
            print('x92: {}'.format(x92))
        x93=self.silu18(x92)
        if x93 is None:
            print('x93: {}'.format(x93))
        elif isinstance(x93, torch.Tensor):
            print('x93: {}'.format(x93.shape))
        elif isinstance(x93, tuple):
            tuple_shapes = '('
            for item in x93:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x93: {}'.format(tuple_shapes))
        else:
            print('x93: {}'.format(x93))
        x94=self.adaptiveavgpool2d6(x93)
        if x94 is None:
            print('x94: {}'.format(x94))
        elif isinstance(x94, torch.Tensor):
            print('x94: {}'.format(x94.shape))
        elif isinstance(x94, tuple):
            tuple_shapes = '('
            for item in x94:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x94: {}'.format(tuple_shapes))
        else:
            print('x94: {}'.format(x94))
        x95=self.conv2d31(x94)
        if x95 is None:
            print('x95: {}'.format(x95))
        elif isinstance(x95, torch.Tensor):
            print('x95: {}'.format(x95.shape))
        elif isinstance(x95, tuple):
            tuple_shapes = '('
            for item in x95:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x95: {}'.format(tuple_shapes))
        else:
            print('x95: {}'.format(x95))
        x96=self.silu19(x95)
        if x96 is None:
            print('x96: {}'.format(x96))
        elif isinstance(x96, torch.Tensor):
            print('x96: {}'.format(x96.shape))
        elif isinstance(x96, tuple):
            tuple_shapes = '('
            for item in x96:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x96: {}'.format(tuple_shapes))
        else:
            print('x96: {}'.format(x96))
        x97=self.conv2d32(x96)
        if x97 is None:
            print('x97: {}'.format(x97))
        elif isinstance(x97, torch.Tensor):
            print('x97: {}'.format(x97.shape))
        elif isinstance(x97, tuple):
            tuple_shapes = '('
            for item in x97:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x97: {}'.format(tuple_shapes))
        else:
            print('x97: {}'.format(x97))
        x98=self.sigmoid6(x97)
        if x98 is None:
            print('x98: {}'.format(x98))
        elif isinstance(x98, torch.Tensor):
            print('x98: {}'.format(x98.shape))
        elif isinstance(x98, tuple):
            tuple_shapes = '('
            for item in x98:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x98: {}'.format(tuple_shapes))
        else:
            print('x98: {}'.format(x98))
        x99=operator.mul(x98, x93)
        if x99 is None:
            print('x99: {}'.format(x99))
        elif isinstance(x99, torch.Tensor):
            print('x99: {}'.format(x99.shape))
        elif isinstance(x99, tuple):
            tuple_shapes = '('
            for item in x99:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x99: {}'.format(tuple_shapes))
        else:
            print('x99: {}'.format(x99))
        x100=self.conv2d33(x99)
        if x100 is None:
            print('x100: {}'.format(x100))
        elif isinstance(x100, torch.Tensor):
            print('x100: {}'.format(x100.shape))
        elif isinstance(x100, tuple):
            tuple_shapes = '('
            for item in x100:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x100: {}'.format(tuple_shapes))
        else:
            print('x100: {}'.format(x100))
        x101=self.batchnorm2d19(x100)
        if x101 is None:
            print('x101: {}'.format(x101))
        elif isinstance(x101, torch.Tensor):
            print('x101: {}'.format(x101.shape))
        elif isinstance(x101, tuple):
            tuple_shapes = '('
            for item in x101:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x101: {}'.format(tuple_shapes))
        else:
            print('x101: {}'.format(x101))
        x102=stochastic_depth(x101, 0.04615384615384616, 'row', False)
        if x102 is None:
            print('x102: {}'.format(x102))
        elif isinstance(x102, torch.Tensor):
            print('x102: {}'.format(x102.shape))
        elif isinstance(x102, tuple):
            tuple_shapes = '('
            for item in x102:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x102: {}'.format(tuple_shapes))
        else:
            print('x102: {}'.format(x102))
        x103=operator.add(x102, x87)
        if x103 is None:
            print('x103: {}'.format(x103))
        elif isinstance(x103, torch.Tensor):
            print('x103: {}'.format(x103.shape))
        elif isinstance(x103, tuple):
            tuple_shapes = '('
            for item in x103:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x103: {}'.format(tuple_shapes))
        else:
            print('x103: {}'.format(x103))
        x104=self.conv2d34(x103)
        if x104 is None:
            print('x104: {}'.format(x104))
        elif isinstance(x104, torch.Tensor):
            print('x104: {}'.format(x104.shape))
        elif isinstance(x104, tuple):
            tuple_shapes = '('
            for item in x104:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x104: {}'.format(tuple_shapes))
        else:
            print('x104: {}'.format(x104))
        x105=self.batchnorm2d20(x104)
        if x105 is None:
            print('x105: {}'.format(x105))
        elif isinstance(x105, torch.Tensor):
            print('x105: {}'.format(x105.shape))
        elif isinstance(x105, tuple):
            tuple_shapes = '('
            for item in x105:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x105: {}'.format(tuple_shapes))
        else:
            print('x105: {}'.format(x105))
        x106=self.silu20(x105)
        if x106 is None:
            print('x106: {}'.format(x106))
        elif isinstance(x106, torch.Tensor):
            print('x106: {}'.format(x106.shape))
        elif isinstance(x106, tuple):
            tuple_shapes = '('
            for item in x106:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x106: {}'.format(tuple_shapes))
        else:
            print('x106: {}'.format(x106))
        x107=self.conv2d35(x106)
        if x107 is None:
            print('x107: {}'.format(x107))
        elif isinstance(x107, torch.Tensor):
            print('x107: {}'.format(x107.shape))
        elif isinstance(x107, tuple):
            tuple_shapes = '('
            for item in x107:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x107: {}'.format(tuple_shapes))
        else:
            print('x107: {}'.format(x107))
        x108=self.batchnorm2d21(x107)
        if x108 is None:
            print('x108: {}'.format(x108))
        elif isinstance(x108, torch.Tensor):
            print('x108: {}'.format(x108.shape))
        elif isinstance(x108, tuple):
            tuple_shapes = '('
            for item in x108:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x108: {}'.format(tuple_shapes))
        else:
            print('x108: {}'.format(x108))
        x109=self.silu21(x108)
        if x109 is None:
            print('x109: {}'.format(x109))
        elif isinstance(x109, torch.Tensor):
            print('x109: {}'.format(x109.shape))
        elif isinstance(x109, tuple):
            tuple_shapes = '('
            for item in x109:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x109: {}'.format(tuple_shapes))
        else:
            print('x109: {}'.format(x109))
        x110=self.adaptiveavgpool2d7(x109)
        if x110 is None:
            print('x110: {}'.format(x110))
        elif isinstance(x110, torch.Tensor):
            print('x110: {}'.format(x110.shape))
        elif isinstance(x110, tuple):
            tuple_shapes = '('
            for item in x110:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x110: {}'.format(tuple_shapes))
        else:
            print('x110: {}'.format(x110))
        x111=self.conv2d36(x110)
        if x111 is None:
            print('x111: {}'.format(x111))
        elif isinstance(x111, torch.Tensor):
            print('x111: {}'.format(x111.shape))
        elif isinstance(x111, tuple):
            tuple_shapes = '('
            for item in x111:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x111: {}'.format(tuple_shapes))
        else:
            print('x111: {}'.format(x111))
        x112=self.silu22(x111)
        if x112 is None:
            print('x112: {}'.format(x112))
        elif isinstance(x112, torch.Tensor):
            print('x112: {}'.format(x112.shape))
        elif isinstance(x112, tuple):
            tuple_shapes = '('
            for item in x112:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x112: {}'.format(tuple_shapes))
        else:
            print('x112: {}'.format(x112))
        x113=self.conv2d37(x112)
        if x113 is None:
            print('x113: {}'.format(x113))
        elif isinstance(x113, torch.Tensor):
            print('x113: {}'.format(x113.shape))
        elif isinstance(x113, tuple):
            tuple_shapes = '('
            for item in x113:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x113: {}'.format(tuple_shapes))
        else:
            print('x113: {}'.format(x113))
        x114=self.sigmoid7(x113)
        if x114 is None:
            print('x114: {}'.format(x114))
        elif isinstance(x114, torch.Tensor):
            print('x114: {}'.format(x114.shape))
        elif isinstance(x114, tuple):
            tuple_shapes = '('
            for item in x114:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x114: {}'.format(tuple_shapes))
        else:
            print('x114: {}'.format(x114))
        x115=operator.mul(x114, x109)
        if x115 is None:
            print('x115: {}'.format(x115))
        elif isinstance(x115, torch.Tensor):
            print('x115: {}'.format(x115.shape))
        elif isinstance(x115, tuple):
            tuple_shapes = '('
            for item in x115:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x115: {}'.format(tuple_shapes))
        else:
            print('x115: {}'.format(x115))
        x116=self.conv2d38(x115)
        if x116 is None:
            print('x116: {}'.format(x116))
        elif isinstance(x116, torch.Tensor):
            print('x116: {}'.format(x116.shape))
        elif isinstance(x116, tuple):
            tuple_shapes = '('
            for item in x116:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x116: {}'.format(tuple_shapes))
        else:
            print('x116: {}'.format(x116))
        x117=self.batchnorm2d22(x116)
        if x117 is None:
            print('x117: {}'.format(x117))
        elif isinstance(x117, torch.Tensor):
            print('x117: {}'.format(x117.shape))
        elif isinstance(x117, tuple):
            tuple_shapes = '('
            for item in x117:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x117: {}'.format(tuple_shapes))
        else:
            print('x117: {}'.format(x117))
        x118=stochastic_depth(x117, 0.05384615384615385, 'row', False)
        if x118 is None:
            print('x118: {}'.format(x118))
        elif isinstance(x118, torch.Tensor):
            print('x118: {}'.format(x118.shape))
        elif isinstance(x118, tuple):
            tuple_shapes = '('
            for item in x118:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x118: {}'.format(tuple_shapes))
        else:
            print('x118: {}'.format(x118))
        x119=operator.add(x118, x103)
        if x119 is None:
            print('x119: {}'.format(x119))
        elif isinstance(x119, torch.Tensor):
            print('x119: {}'.format(x119.shape))
        elif isinstance(x119, tuple):
            tuple_shapes = '('
            for item in x119:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x119: {}'.format(tuple_shapes))
        else:
            print('x119: {}'.format(x119))
        x120=self.conv2d39(x119)
        if x120 is None:
            print('x120: {}'.format(x120))
        elif isinstance(x120, torch.Tensor):
            print('x120: {}'.format(x120.shape))
        elif isinstance(x120, tuple):
            tuple_shapes = '('
            for item in x120:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x120: {}'.format(tuple_shapes))
        else:
            print('x120: {}'.format(x120))
        x121=self.batchnorm2d23(x120)
        if x121 is None:
            print('x121: {}'.format(x121))
        elif isinstance(x121, torch.Tensor):
            print('x121: {}'.format(x121.shape))
        elif isinstance(x121, tuple):
            tuple_shapes = '('
            for item in x121:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x121: {}'.format(tuple_shapes))
        else:
            print('x121: {}'.format(x121))
        x122=self.silu23(x121)
        if x122 is None:
            print('x122: {}'.format(x122))
        elif isinstance(x122, torch.Tensor):
            print('x122: {}'.format(x122.shape))
        elif isinstance(x122, tuple):
            tuple_shapes = '('
            for item in x122:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x122: {}'.format(tuple_shapes))
        else:
            print('x122: {}'.format(x122))
        x123=self.conv2d40(x122)
        if x123 is None:
            print('x123: {}'.format(x123))
        elif isinstance(x123, torch.Tensor):
            print('x123: {}'.format(x123.shape))
        elif isinstance(x123, tuple):
            tuple_shapes = '('
            for item in x123:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x123: {}'.format(tuple_shapes))
        else:
            print('x123: {}'.format(x123))
        x124=self.batchnorm2d24(x123)
        if x124 is None:
            print('x124: {}'.format(x124))
        elif isinstance(x124, torch.Tensor):
            print('x124: {}'.format(x124.shape))
        elif isinstance(x124, tuple):
            tuple_shapes = '('
            for item in x124:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x124: {}'.format(tuple_shapes))
        else:
            print('x124: {}'.format(x124))
        x125=self.silu24(x124)
        if x125 is None:
            print('x125: {}'.format(x125))
        elif isinstance(x125, torch.Tensor):
            print('x125: {}'.format(x125.shape))
        elif isinstance(x125, tuple):
            tuple_shapes = '('
            for item in x125:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x125: {}'.format(tuple_shapes))
        else:
            print('x125: {}'.format(x125))
        x126=self.adaptiveavgpool2d8(x125)
        if x126 is None:
            print('x126: {}'.format(x126))
        elif isinstance(x126, torch.Tensor):
            print('x126: {}'.format(x126.shape))
        elif isinstance(x126, tuple):
            tuple_shapes = '('
            for item in x126:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x126: {}'.format(tuple_shapes))
        else:
            print('x126: {}'.format(x126))
        x127=self.conv2d41(x126)
        if x127 is None:
            print('x127: {}'.format(x127))
        elif isinstance(x127, torch.Tensor):
            print('x127: {}'.format(x127.shape))
        elif isinstance(x127, tuple):
            tuple_shapes = '('
            for item in x127:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x127: {}'.format(tuple_shapes))
        else:
            print('x127: {}'.format(x127))
        x128=self.silu25(x127)
        if x128 is None:
            print('x128: {}'.format(x128))
        elif isinstance(x128, torch.Tensor):
            print('x128: {}'.format(x128.shape))
        elif isinstance(x128, tuple):
            tuple_shapes = '('
            for item in x128:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x128: {}'.format(tuple_shapes))
        else:
            print('x128: {}'.format(x128))
        x129=self.conv2d42(x128)
        if x129 is None:
            print('x129: {}'.format(x129))
        elif isinstance(x129, torch.Tensor):
            print('x129: {}'.format(x129.shape))
        elif isinstance(x129, tuple):
            tuple_shapes = '('
            for item in x129:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x129: {}'.format(tuple_shapes))
        else:
            print('x129: {}'.format(x129))
        x130=self.sigmoid8(x129)
        if x130 is None:
            print('x130: {}'.format(x130))
        elif isinstance(x130, torch.Tensor):
            print('x130: {}'.format(x130.shape))
        elif isinstance(x130, tuple):
            tuple_shapes = '('
            for item in x130:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x130: {}'.format(tuple_shapes))
        else:
            print('x130: {}'.format(x130))
        x131=operator.mul(x130, x125)
        if x131 is None:
            print('x131: {}'.format(x131))
        elif isinstance(x131, torch.Tensor):
            print('x131: {}'.format(x131.shape))
        elif isinstance(x131, tuple):
            tuple_shapes = '('
            for item in x131:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x131: {}'.format(tuple_shapes))
        else:
            print('x131: {}'.format(x131))
        x132=self.conv2d43(x131)
        if x132 is None:
            print('x132: {}'.format(x132))
        elif isinstance(x132, torch.Tensor):
            print('x132: {}'.format(x132.shape))
        elif isinstance(x132, tuple):
            tuple_shapes = '('
            for item in x132:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x132: {}'.format(tuple_shapes))
        else:
            print('x132: {}'.format(x132))
        x133=self.batchnorm2d25(x132)
        if x133 is None:
            print('x133: {}'.format(x133))
        elif isinstance(x133, torch.Tensor):
            print('x133: {}'.format(x133.shape))
        elif isinstance(x133, tuple):
            tuple_shapes = '('
            for item in x133:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x133: {}'.format(tuple_shapes))
        else:
            print('x133: {}'.format(x133))
        x134=self.conv2d44(x133)
        if x134 is None:
            print('x134: {}'.format(x134))
        elif isinstance(x134, torch.Tensor):
            print('x134: {}'.format(x134.shape))
        elif isinstance(x134, tuple):
            tuple_shapes = '('
            for item in x134:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x134: {}'.format(tuple_shapes))
        else:
            print('x134: {}'.format(x134))
        x135=self.batchnorm2d26(x134)
        if x135 is None:
            print('x135: {}'.format(x135))
        elif isinstance(x135, torch.Tensor):
            print('x135: {}'.format(x135.shape))
        elif isinstance(x135, tuple):
            tuple_shapes = '('
            for item in x135:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x135: {}'.format(tuple_shapes))
        else:
            print('x135: {}'.format(x135))
        x136=self.silu26(x135)
        if x136 is None:
            print('x136: {}'.format(x136))
        elif isinstance(x136, torch.Tensor):
            print('x136: {}'.format(x136.shape))
        elif isinstance(x136, tuple):
            tuple_shapes = '('
            for item in x136:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x136: {}'.format(tuple_shapes))
        else:
            print('x136: {}'.format(x136))
        x137=self.conv2d45(x136)
        if x137 is None:
            print('x137: {}'.format(x137))
        elif isinstance(x137, torch.Tensor):
            print('x137: {}'.format(x137.shape))
        elif isinstance(x137, tuple):
            tuple_shapes = '('
            for item in x137:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x137: {}'.format(tuple_shapes))
        else:
            print('x137: {}'.format(x137))
        x138=self.batchnorm2d27(x137)
        if x138 is None:
            print('x138: {}'.format(x138))
        elif isinstance(x138, torch.Tensor):
            print('x138: {}'.format(x138.shape))
        elif isinstance(x138, tuple):
            tuple_shapes = '('
            for item in x138:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x138: {}'.format(tuple_shapes))
        else:
            print('x138: {}'.format(x138))
        x139=self.silu27(x138)
        if x139 is None:
            print('x139: {}'.format(x139))
        elif isinstance(x139, torch.Tensor):
            print('x139: {}'.format(x139.shape))
        elif isinstance(x139, tuple):
            tuple_shapes = '('
            for item in x139:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x139: {}'.format(tuple_shapes))
        else:
            print('x139: {}'.format(x139))
        x140=self.adaptiveavgpool2d9(x139)
        if x140 is None:
            print('x140: {}'.format(x140))
        elif isinstance(x140, torch.Tensor):
            print('x140: {}'.format(x140.shape))
        elif isinstance(x140, tuple):
            tuple_shapes = '('
            for item in x140:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x140: {}'.format(tuple_shapes))
        else:
            print('x140: {}'.format(x140))
        x141=self.conv2d46(x140)
        if x141 is None:
            print('x141: {}'.format(x141))
        elif isinstance(x141, torch.Tensor):
            print('x141: {}'.format(x141.shape))
        elif isinstance(x141, tuple):
            tuple_shapes = '('
            for item in x141:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x141: {}'.format(tuple_shapes))
        else:
            print('x141: {}'.format(x141))
        x142=self.silu28(x141)
        if x142 is None:
            print('x142: {}'.format(x142))
        elif isinstance(x142, torch.Tensor):
            print('x142: {}'.format(x142.shape))
        elif isinstance(x142, tuple):
            tuple_shapes = '('
            for item in x142:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x142: {}'.format(tuple_shapes))
        else:
            print('x142: {}'.format(x142))
        x143=self.conv2d47(x142)
        if x143 is None:
            print('x143: {}'.format(x143))
        elif isinstance(x143, torch.Tensor):
            print('x143: {}'.format(x143.shape))
        elif isinstance(x143, tuple):
            tuple_shapes = '('
            for item in x143:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x143: {}'.format(tuple_shapes))
        else:
            print('x143: {}'.format(x143))
        x144=self.sigmoid9(x143)
        if x144 is None:
            print('x144: {}'.format(x144))
        elif isinstance(x144, torch.Tensor):
            print('x144: {}'.format(x144.shape))
        elif isinstance(x144, tuple):
            tuple_shapes = '('
            for item in x144:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x144: {}'.format(tuple_shapes))
        else:
            print('x144: {}'.format(x144))
        x145=operator.mul(x144, x139)
        if x145 is None:
            print('x145: {}'.format(x145))
        elif isinstance(x145, torch.Tensor):
            print('x145: {}'.format(x145.shape))
        elif isinstance(x145, tuple):
            tuple_shapes = '('
            for item in x145:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x145: {}'.format(tuple_shapes))
        else:
            print('x145: {}'.format(x145))
        x146=self.conv2d48(x145)
        if x146 is None:
            print('x146: {}'.format(x146))
        elif isinstance(x146, torch.Tensor):
            print('x146: {}'.format(x146.shape))
        elif isinstance(x146, tuple):
            tuple_shapes = '('
            for item in x146:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x146: {}'.format(tuple_shapes))
        else:
            print('x146: {}'.format(x146))
        x147=self.batchnorm2d28(x146)
        if x147 is None:
            print('x147: {}'.format(x147))
        elif isinstance(x147, torch.Tensor):
            print('x147: {}'.format(x147.shape))
        elif isinstance(x147, tuple):
            tuple_shapes = '('
            for item in x147:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x147: {}'.format(tuple_shapes))
        else:
            print('x147: {}'.format(x147))
        x148=stochastic_depth(x147, 0.06923076923076923, 'row', False)
        if x148 is None:
            print('x148: {}'.format(x148))
        elif isinstance(x148, torch.Tensor):
            print('x148: {}'.format(x148.shape))
        elif isinstance(x148, tuple):
            tuple_shapes = '('
            for item in x148:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x148: {}'.format(tuple_shapes))
        else:
            print('x148: {}'.format(x148))
        x149=operator.add(x148, x133)
        if x149 is None:
            print('x149: {}'.format(x149))
        elif isinstance(x149, torch.Tensor):
            print('x149: {}'.format(x149.shape))
        elif isinstance(x149, tuple):
            tuple_shapes = '('
            for item in x149:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x149: {}'.format(tuple_shapes))
        else:
            print('x149: {}'.format(x149))
        x150=self.conv2d49(x149)
        if x150 is None:
            print('x150: {}'.format(x150))
        elif isinstance(x150, torch.Tensor):
            print('x150: {}'.format(x150.shape))
        elif isinstance(x150, tuple):
            tuple_shapes = '('
            for item in x150:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x150: {}'.format(tuple_shapes))
        else:
            print('x150: {}'.format(x150))
        x151=self.batchnorm2d29(x150)
        if x151 is None:
            print('x151: {}'.format(x151))
        elif isinstance(x151, torch.Tensor):
            print('x151: {}'.format(x151.shape))
        elif isinstance(x151, tuple):
            tuple_shapes = '('
            for item in x151:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x151: {}'.format(tuple_shapes))
        else:
            print('x151: {}'.format(x151))
        x152=self.silu29(x151)
        if x152 is None:
            print('x152: {}'.format(x152))
        elif isinstance(x152, torch.Tensor):
            print('x152: {}'.format(x152.shape))
        elif isinstance(x152, tuple):
            tuple_shapes = '('
            for item in x152:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x152: {}'.format(tuple_shapes))
        else:
            print('x152: {}'.format(x152))
        x153=self.conv2d50(x152)
        if x153 is None:
            print('x153: {}'.format(x153))
        elif isinstance(x153, torch.Tensor):
            print('x153: {}'.format(x153.shape))
        elif isinstance(x153, tuple):
            tuple_shapes = '('
            for item in x153:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x153: {}'.format(tuple_shapes))
        else:
            print('x153: {}'.format(x153))
        x154=self.batchnorm2d30(x153)
        if x154 is None:
            print('x154: {}'.format(x154))
        elif isinstance(x154, torch.Tensor):
            print('x154: {}'.format(x154.shape))
        elif isinstance(x154, tuple):
            tuple_shapes = '('
            for item in x154:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x154: {}'.format(tuple_shapes))
        else:
            print('x154: {}'.format(x154))
        x155=self.silu30(x154)
        if x155 is None:
            print('x155: {}'.format(x155))
        elif isinstance(x155, torch.Tensor):
            print('x155: {}'.format(x155.shape))
        elif isinstance(x155, tuple):
            tuple_shapes = '('
            for item in x155:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x155: {}'.format(tuple_shapes))
        else:
            print('x155: {}'.format(x155))
        x156=self.adaptiveavgpool2d10(x155)
        if x156 is None:
            print('x156: {}'.format(x156))
        elif isinstance(x156, torch.Tensor):
            print('x156: {}'.format(x156.shape))
        elif isinstance(x156, tuple):
            tuple_shapes = '('
            for item in x156:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x156: {}'.format(tuple_shapes))
        else:
            print('x156: {}'.format(x156))
        x157=self.conv2d51(x156)
        if x157 is None:
            print('x157: {}'.format(x157))
        elif isinstance(x157, torch.Tensor):
            print('x157: {}'.format(x157.shape))
        elif isinstance(x157, tuple):
            tuple_shapes = '('
            for item in x157:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x157: {}'.format(tuple_shapes))
        else:
            print('x157: {}'.format(x157))
        x158=self.silu31(x157)
        if x158 is None:
            print('x158: {}'.format(x158))
        elif isinstance(x158, torch.Tensor):
            print('x158: {}'.format(x158.shape))
        elif isinstance(x158, tuple):
            tuple_shapes = '('
            for item in x158:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x158: {}'.format(tuple_shapes))
        else:
            print('x158: {}'.format(x158))
        x159=self.conv2d52(x158)
        if x159 is None:
            print('x159: {}'.format(x159))
        elif isinstance(x159, torch.Tensor):
            print('x159: {}'.format(x159.shape))
        elif isinstance(x159, tuple):
            tuple_shapes = '('
            for item in x159:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x159: {}'.format(tuple_shapes))
        else:
            print('x159: {}'.format(x159))
        x160=self.sigmoid10(x159)
        if x160 is None:
            print('x160: {}'.format(x160))
        elif isinstance(x160, torch.Tensor):
            print('x160: {}'.format(x160.shape))
        elif isinstance(x160, tuple):
            tuple_shapes = '('
            for item in x160:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x160: {}'.format(tuple_shapes))
        else:
            print('x160: {}'.format(x160))
        x161=operator.mul(x160, x155)
        if x161 is None:
            print('x161: {}'.format(x161))
        elif isinstance(x161, torch.Tensor):
            print('x161: {}'.format(x161.shape))
        elif isinstance(x161, tuple):
            tuple_shapes = '('
            for item in x161:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x161: {}'.format(tuple_shapes))
        else:
            print('x161: {}'.format(x161))
        x162=self.conv2d53(x161)
        if x162 is None:
            print('x162: {}'.format(x162))
        elif isinstance(x162, torch.Tensor):
            print('x162: {}'.format(x162.shape))
        elif isinstance(x162, tuple):
            tuple_shapes = '('
            for item in x162:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x162: {}'.format(tuple_shapes))
        else:
            print('x162: {}'.format(x162))
        x163=self.batchnorm2d31(x162)
        if x163 is None:
            print('x163: {}'.format(x163))
        elif isinstance(x163, torch.Tensor):
            print('x163: {}'.format(x163.shape))
        elif isinstance(x163, tuple):
            tuple_shapes = '('
            for item in x163:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x163: {}'.format(tuple_shapes))
        else:
            print('x163: {}'.format(x163))
        x164=stochastic_depth(x163, 0.07692307692307693, 'row', False)
        if x164 is None:
            print('x164: {}'.format(x164))
        elif isinstance(x164, torch.Tensor):
            print('x164: {}'.format(x164.shape))
        elif isinstance(x164, tuple):
            tuple_shapes = '('
            for item in x164:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x164: {}'.format(tuple_shapes))
        else:
            print('x164: {}'.format(x164))
        x165=operator.add(x164, x149)
        if x165 is None:
            print('x165: {}'.format(x165))
        elif isinstance(x165, torch.Tensor):
            print('x165: {}'.format(x165.shape))
        elif isinstance(x165, tuple):
            tuple_shapes = '('
            for item in x165:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x165: {}'.format(tuple_shapes))
        else:
            print('x165: {}'.format(x165))
        x166=self.conv2d54(x165)
        if x166 is None:
            print('x166: {}'.format(x166))
        elif isinstance(x166, torch.Tensor):
            print('x166: {}'.format(x166.shape))
        elif isinstance(x166, tuple):
            tuple_shapes = '('
            for item in x166:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x166: {}'.format(tuple_shapes))
        else:
            print('x166: {}'.format(x166))
        x167=self.batchnorm2d32(x166)
        if x167 is None:
            print('x167: {}'.format(x167))
        elif isinstance(x167, torch.Tensor):
            print('x167: {}'.format(x167.shape))
        elif isinstance(x167, tuple):
            tuple_shapes = '('
            for item in x167:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x167: {}'.format(tuple_shapes))
        else:
            print('x167: {}'.format(x167))
        x168=self.silu32(x167)
        if x168 is None:
            print('x168: {}'.format(x168))
        elif isinstance(x168, torch.Tensor):
            print('x168: {}'.format(x168.shape))
        elif isinstance(x168, tuple):
            tuple_shapes = '('
            for item in x168:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x168: {}'.format(tuple_shapes))
        else:
            print('x168: {}'.format(x168))
        x169=self.conv2d55(x168)
        if x169 is None:
            print('x169: {}'.format(x169))
        elif isinstance(x169, torch.Tensor):
            print('x169: {}'.format(x169.shape))
        elif isinstance(x169, tuple):
            tuple_shapes = '('
            for item in x169:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x169: {}'.format(tuple_shapes))
        else:
            print('x169: {}'.format(x169))
        x170=self.batchnorm2d33(x169)
        if x170 is None:
            print('x170: {}'.format(x170))
        elif isinstance(x170, torch.Tensor):
            print('x170: {}'.format(x170.shape))
        elif isinstance(x170, tuple):
            tuple_shapes = '('
            for item in x170:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x170: {}'.format(tuple_shapes))
        else:
            print('x170: {}'.format(x170))
        x171=self.silu33(x170)
        if x171 is None:
            print('x171: {}'.format(x171))
        elif isinstance(x171, torch.Tensor):
            print('x171: {}'.format(x171.shape))
        elif isinstance(x171, tuple):
            tuple_shapes = '('
            for item in x171:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x171: {}'.format(tuple_shapes))
        else:
            print('x171: {}'.format(x171))
        x172=self.adaptiveavgpool2d11(x171)
        if x172 is None:
            print('x172: {}'.format(x172))
        elif isinstance(x172, torch.Tensor):
            print('x172: {}'.format(x172.shape))
        elif isinstance(x172, tuple):
            tuple_shapes = '('
            for item in x172:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x172: {}'.format(tuple_shapes))
        else:
            print('x172: {}'.format(x172))
        x173=self.conv2d56(x172)
        if x173 is None:
            print('x173: {}'.format(x173))
        elif isinstance(x173, torch.Tensor):
            print('x173: {}'.format(x173.shape))
        elif isinstance(x173, tuple):
            tuple_shapes = '('
            for item in x173:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x173: {}'.format(tuple_shapes))
        else:
            print('x173: {}'.format(x173))
        x174=self.silu34(x173)
        if x174 is None:
            print('x174: {}'.format(x174))
        elif isinstance(x174, torch.Tensor):
            print('x174: {}'.format(x174.shape))
        elif isinstance(x174, tuple):
            tuple_shapes = '('
            for item in x174:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x174: {}'.format(tuple_shapes))
        else:
            print('x174: {}'.format(x174))
        x175=self.conv2d57(x174)
        if x175 is None:
            print('x175: {}'.format(x175))
        elif isinstance(x175, torch.Tensor):
            print('x175: {}'.format(x175.shape))
        elif isinstance(x175, tuple):
            tuple_shapes = '('
            for item in x175:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x175: {}'.format(tuple_shapes))
        else:
            print('x175: {}'.format(x175))
        x176=self.sigmoid11(x175)
        if x176 is None:
            print('x176: {}'.format(x176))
        elif isinstance(x176, torch.Tensor):
            print('x176: {}'.format(x176.shape))
        elif isinstance(x176, tuple):
            tuple_shapes = '('
            for item in x176:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x176: {}'.format(tuple_shapes))
        else:
            print('x176: {}'.format(x176))
        x177=operator.mul(x176, x171)
        if x177 is None:
            print('x177: {}'.format(x177))
        elif isinstance(x177, torch.Tensor):
            print('x177: {}'.format(x177.shape))
        elif isinstance(x177, tuple):
            tuple_shapes = '('
            for item in x177:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x177: {}'.format(tuple_shapes))
        else:
            print('x177: {}'.format(x177))
        x178=self.conv2d58(x177)
        if x178 is None:
            print('x178: {}'.format(x178))
        elif isinstance(x178, torch.Tensor):
            print('x178: {}'.format(x178.shape))
        elif isinstance(x178, tuple):
            tuple_shapes = '('
            for item in x178:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x178: {}'.format(tuple_shapes))
        else:
            print('x178: {}'.format(x178))
        x179=self.batchnorm2d34(x178)
        if x179 is None:
            print('x179: {}'.format(x179))
        elif isinstance(x179, torch.Tensor):
            print('x179: {}'.format(x179.shape))
        elif isinstance(x179, tuple):
            tuple_shapes = '('
            for item in x179:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x179: {}'.format(tuple_shapes))
        else:
            print('x179: {}'.format(x179))
        x180=stochastic_depth(x179, 0.08461538461538462, 'row', False)
        if x180 is None:
            print('x180: {}'.format(x180))
        elif isinstance(x180, torch.Tensor):
            print('x180: {}'.format(x180.shape))
        elif isinstance(x180, tuple):
            tuple_shapes = '('
            for item in x180:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x180: {}'.format(tuple_shapes))
        else:
            print('x180: {}'.format(x180))
        x181=operator.add(x180, x165)
        if x181 is None:
            print('x181: {}'.format(x181))
        elif isinstance(x181, torch.Tensor):
            print('x181: {}'.format(x181.shape))
        elif isinstance(x181, tuple):
            tuple_shapes = '('
            for item in x181:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x181: {}'.format(tuple_shapes))
        else:
            print('x181: {}'.format(x181))
        x182=self.conv2d59(x181)
        if x182 is None:
            print('x182: {}'.format(x182))
        elif isinstance(x182, torch.Tensor):
            print('x182: {}'.format(x182.shape))
        elif isinstance(x182, tuple):
            tuple_shapes = '('
            for item in x182:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x182: {}'.format(tuple_shapes))
        else:
            print('x182: {}'.format(x182))
        x183=self.batchnorm2d35(x182)
        if x183 is None:
            print('x183: {}'.format(x183))
        elif isinstance(x183, torch.Tensor):
            print('x183: {}'.format(x183.shape))
        elif isinstance(x183, tuple):
            tuple_shapes = '('
            for item in x183:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x183: {}'.format(tuple_shapes))
        else:
            print('x183: {}'.format(x183))
        x184=self.silu35(x183)
        if x184 is None:
            print('x184: {}'.format(x184))
        elif isinstance(x184, torch.Tensor):
            print('x184: {}'.format(x184.shape))
        elif isinstance(x184, tuple):
            tuple_shapes = '('
            for item in x184:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x184: {}'.format(tuple_shapes))
        else:
            print('x184: {}'.format(x184))
        x185=self.conv2d60(x184)
        if x185 is None:
            print('x185: {}'.format(x185))
        elif isinstance(x185, torch.Tensor):
            print('x185: {}'.format(x185.shape))
        elif isinstance(x185, tuple):
            tuple_shapes = '('
            for item in x185:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x185: {}'.format(tuple_shapes))
        else:
            print('x185: {}'.format(x185))
        x186=self.batchnorm2d36(x185)
        if x186 is None:
            print('x186: {}'.format(x186))
        elif isinstance(x186, torch.Tensor):
            print('x186: {}'.format(x186.shape))
        elif isinstance(x186, tuple):
            tuple_shapes = '('
            for item in x186:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x186: {}'.format(tuple_shapes))
        else:
            print('x186: {}'.format(x186))
        x187=self.silu36(x186)
        if x187 is None:
            print('x187: {}'.format(x187))
        elif isinstance(x187, torch.Tensor):
            print('x187: {}'.format(x187.shape))
        elif isinstance(x187, tuple):
            tuple_shapes = '('
            for item in x187:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x187: {}'.format(tuple_shapes))
        else:
            print('x187: {}'.format(x187))
        x188=self.adaptiveavgpool2d12(x187)
        if x188 is None:
            print('x188: {}'.format(x188))
        elif isinstance(x188, torch.Tensor):
            print('x188: {}'.format(x188.shape))
        elif isinstance(x188, tuple):
            tuple_shapes = '('
            for item in x188:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x188: {}'.format(tuple_shapes))
        else:
            print('x188: {}'.format(x188))
        x189=self.conv2d61(x188)
        if x189 is None:
            print('x189: {}'.format(x189))
        elif isinstance(x189, torch.Tensor):
            print('x189: {}'.format(x189.shape))
        elif isinstance(x189, tuple):
            tuple_shapes = '('
            for item in x189:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x189: {}'.format(tuple_shapes))
        else:
            print('x189: {}'.format(x189))
        x190=self.silu37(x189)
        if x190 is None:
            print('x190: {}'.format(x190))
        elif isinstance(x190, torch.Tensor):
            print('x190: {}'.format(x190.shape))
        elif isinstance(x190, tuple):
            tuple_shapes = '('
            for item in x190:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x190: {}'.format(tuple_shapes))
        else:
            print('x190: {}'.format(x190))
        x191=self.conv2d62(x190)
        if x191 is None:
            print('x191: {}'.format(x191))
        elif isinstance(x191, torch.Tensor):
            print('x191: {}'.format(x191.shape))
        elif isinstance(x191, tuple):
            tuple_shapes = '('
            for item in x191:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x191: {}'.format(tuple_shapes))
        else:
            print('x191: {}'.format(x191))
        x192=self.sigmoid12(x191)
        if x192 is None:
            print('x192: {}'.format(x192))
        elif isinstance(x192, torch.Tensor):
            print('x192: {}'.format(x192.shape))
        elif isinstance(x192, tuple):
            tuple_shapes = '('
            for item in x192:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x192: {}'.format(tuple_shapes))
        else:
            print('x192: {}'.format(x192))
        x193=operator.mul(x192, x187)
        if x193 is None:
            print('x193: {}'.format(x193))
        elif isinstance(x193, torch.Tensor):
            print('x193: {}'.format(x193.shape))
        elif isinstance(x193, tuple):
            tuple_shapes = '('
            for item in x193:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x193: {}'.format(tuple_shapes))
        else:
            print('x193: {}'.format(x193))
        x194=self.conv2d63(x193)
        if x194 is None:
            print('x194: {}'.format(x194))
        elif isinstance(x194, torch.Tensor):
            print('x194: {}'.format(x194.shape))
        elif isinstance(x194, tuple):
            tuple_shapes = '('
            for item in x194:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x194: {}'.format(tuple_shapes))
        else:
            print('x194: {}'.format(x194))
        x195=self.batchnorm2d37(x194)
        if x195 is None:
            print('x195: {}'.format(x195))
        elif isinstance(x195, torch.Tensor):
            print('x195: {}'.format(x195.shape))
        elif isinstance(x195, tuple):
            tuple_shapes = '('
            for item in x195:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x195: {}'.format(tuple_shapes))
        else:
            print('x195: {}'.format(x195))
        x196=stochastic_depth(x195, 0.09230769230769233, 'row', False)
        if x196 is None:
            print('x196: {}'.format(x196))
        elif isinstance(x196, torch.Tensor):
            print('x196: {}'.format(x196.shape))
        elif isinstance(x196, tuple):
            tuple_shapes = '('
            for item in x196:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x196: {}'.format(tuple_shapes))
        else:
            print('x196: {}'.format(x196))
        x197=operator.add(x196, x181)
        if x197 is None:
            print('x197: {}'.format(x197))
        elif isinstance(x197, torch.Tensor):
            print('x197: {}'.format(x197.shape))
        elif isinstance(x197, tuple):
            tuple_shapes = '('
            for item in x197:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x197: {}'.format(tuple_shapes))
        else:
            print('x197: {}'.format(x197))
        x198=self.conv2d64(x197)
        if x198 is None:
            print('x198: {}'.format(x198))
        elif isinstance(x198, torch.Tensor):
            print('x198: {}'.format(x198.shape))
        elif isinstance(x198, tuple):
            tuple_shapes = '('
            for item in x198:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x198: {}'.format(tuple_shapes))
        else:
            print('x198: {}'.format(x198))
        x199=self.batchnorm2d38(x198)
        if x199 is None:
            print('x199: {}'.format(x199))
        elif isinstance(x199, torch.Tensor):
            print('x199: {}'.format(x199.shape))
        elif isinstance(x199, tuple):
            tuple_shapes = '('
            for item in x199:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x199: {}'.format(tuple_shapes))
        else:
            print('x199: {}'.format(x199))
        x200=self.silu38(x199)
        if x200 is None:
            print('x200: {}'.format(x200))
        elif isinstance(x200, torch.Tensor):
            print('x200: {}'.format(x200.shape))
        elif isinstance(x200, tuple):
            tuple_shapes = '('
            for item in x200:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x200: {}'.format(tuple_shapes))
        else:
            print('x200: {}'.format(x200))
        x201=self.conv2d65(x200)
        if x201 is None:
            print('x201: {}'.format(x201))
        elif isinstance(x201, torch.Tensor):
            print('x201: {}'.format(x201.shape))
        elif isinstance(x201, tuple):
            tuple_shapes = '('
            for item in x201:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x201: {}'.format(tuple_shapes))
        else:
            print('x201: {}'.format(x201))
        x202=self.batchnorm2d39(x201)
        if x202 is None:
            print('x202: {}'.format(x202))
        elif isinstance(x202, torch.Tensor):
            print('x202: {}'.format(x202.shape))
        elif isinstance(x202, tuple):
            tuple_shapes = '('
            for item in x202:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x202: {}'.format(tuple_shapes))
        else:
            print('x202: {}'.format(x202))
        x203=self.silu39(x202)
        if x203 is None:
            print('x203: {}'.format(x203))
        elif isinstance(x203, torch.Tensor):
            print('x203: {}'.format(x203.shape))
        elif isinstance(x203, tuple):
            tuple_shapes = '('
            for item in x203:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x203: {}'.format(tuple_shapes))
        else:
            print('x203: {}'.format(x203))
        x204=self.adaptiveavgpool2d13(x203)
        if x204 is None:
            print('x204: {}'.format(x204))
        elif isinstance(x204, torch.Tensor):
            print('x204: {}'.format(x204.shape))
        elif isinstance(x204, tuple):
            tuple_shapes = '('
            for item in x204:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x204: {}'.format(tuple_shapes))
        else:
            print('x204: {}'.format(x204))
        x205=self.conv2d66(x204)
        if x205 is None:
            print('x205: {}'.format(x205))
        elif isinstance(x205, torch.Tensor):
            print('x205: {}'.format(x205.shape))
        elif isinstance(x205, tuple):
            tuple_shapes = '('
            for item in x205:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x205: {}'.format(tuple_shapes))
        else:
            print('x205: {}'.format(x205))
        x206=self.silu40(x205)
        if x206 is None:
            print('x206: {}'.format(x206))
        elif isinstance(x206, torch.Tensor):
            print('x206: {}'.format(x206.shape))
        elif isinstance(x206, tuple):
            tuple_shapes = '('
            for item in x206:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x206: {}'.format(tuple_shapes))
        else:
            print('x206: {}'.format(x206))
        x207=self.conv2d67(x206)
        if x207 is None:
            print('x207: {}'.format(x207))
        elif isinstance(x207, torch.Tensor):
            print('x207: {}'.format(x207.shape))
        elif isinstance(x207, tuple):
            tuple_shapes = '('
            for item in x207:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x207: {}'.format(tuple_shapes))
        else:
            print('x207: {}'.format(x207))
        x208=self.sigmoid13(x207)
        if x208 is None:
            print('x208: {}'.format(x208))
        elif isinstance(x208, torch.Tensor):
            print('x208: {}'.format(x208.shape))
        elif isinstance(x208, tuple):
            tuple_shapes = '('
            for item in x208:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x208: {}'.format(tuple_shapes))
        else:
            print('x208: {}'.format(x208))
        x209=operator.mul(x208, x203)
        if x209 is None:
            print('x209: {}'.format(x209))
        elif isinstance(x209, torch.Tensor):
            print('x209: {}'.format(x209.shape))
        elif isinstance(x209, tuple):
            tuple_shapes = '('
            for item in x209:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x209: {}'.format(tuple_shapes))
        else:
            print('x209: {}'.format(x209))
        x210=self.conv2d68(x209)
        if x210 is None:
            print('x210: {}'.format(x210))
        elif isinstance(x210, torch.Tensor):
            print('x210: {}'.format(x210.shape))
        elif isinstance(x210, tuple):
            tuple_shapes = '('
            for item in x210:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x210: {}'.format(tuple_shapes))
        else:
            print('x210: {}'.format(x210))
        x211=self.batchnorm2d40(x210)
        if x211 is None:
            print('x211: {}'.format(x211))
        elif isinstance(x211, torch.Tensor):
            print('x211: {}'.format(x211.shape))
        elif isinstance(x211, tuple):
            tuple_shapes = '('
            for item in x211:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x211: {}'.format(tuple_shapes))
        else:
            print('x211: {}'.format(x211))
        x212=self.conv2d69(x211)
        if x212 is None:
            print('x212: {}'.format(x212))
        elif isinstance(x212, torch.Tensor):
            print('x212: {}'.format(x212.shape))
        elif isinstance(x212, tuple):
            tuple_shapes = '('
            for item in x212:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x212: {}'.format(tuple_shapes))
        else:
            print('x212: {}'.format(x212))
        x213=self.batchnorm2d41(x212)
        if x213 is None:
            print('x213: {}'.format(x213))
        elif isinstance(x213, torch.Tensor):
            print('x213: {}'.format(x213.shape))
        elif isinstance(x213, tuple):
            tuple_shapes = '('
            for item in x213:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x213: {}'.format(tuple_shapes))
        else:
            print('x213: {}'.format(x213))
        x214=self.silu41(x213)
        if x214 is None:
            print('x214: {}'.format(x214))
        elif isinstance(x214, torch.Tensor):
            print('x214: {}'.format(x214.shape))
        elif isinstance(x214, tuple):
            tuple_shapes = '('
            for item in x214:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x214: {}'.format(tuple_shapes))
        else:
            print('x214: {}'.format(x214))
        x215=self.conv2d70(x214)
        if x215 is None:
            print('x215: {}'.format(x215))
        elif isinstance(x215, torch.Tensor):
            print('x215: {}'.format(x215.shape))
        elif isinstance(x215, tuple):
            tuple_shapes = '('
            for item in x215:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x215: {}'.format(tuple_shapes))
        else:
            print('x215: {}'.format(x215))
        x216=self.batchnorm2d42(x215)
        if x216 is None:
            print('x216: {}'.format(x216))
        elif isinstance(x216, torch.Tensor):
            print('x216: {}'.format(x216.shape))
        elif isinstance(x216, tuple):
            tuple_shapes = '('
            for item in x216:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x216: {}'.format(tuple_shapes))
        else:
            print('x216: {}'.format(x216))
        x217=self.silu42(x216)
        if x217 is None:
            print('x217: {}'.format(x217))
        elif isinstance(x217, torch.Tensor):
            print('x217: {}'.format(x217.shape))
        elif isinstance(x217, tuple):
            tuple_shapes = '('
            for item in x217:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x217: {}'.format(tuple_shapes))
        else:
            print('x217: {}'.format(x217))
        x218=self.adaptiveavgpool2d14(x217)
        if x218 is None:
            print('x218: {}'.format(x218))
        elif isinstance(x218, torch.Tensor):
            print('x218: {}'.format(x218.shape))
        elif isinstance(x218, tuple):
            tuple_shapes = '('
            for item in x218:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x218: {}'.format(tuple_shapes))
        else:
            print('x218: {}'.format(x218))
        x219=self.conv2d71(x218)
        if x219 is None:
            print('x219: {}'.format(x219))
        elif isinstance(x219, torch.Tensor):
            print('x219: {}'.format(x219.shape))
        elif isinstance(x219, tuple):
            tuple_shapes = '('
            for item in x219:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x219: {}'.format(tuple_shapes))
        else:
            print('x219: {}'.format(x219))
        x220=self.silu43(x219)
        if x220 is None:
            print('x220: {}'.format(x220))
        elif isinstance(x220, torch.Tensor):
            print('x220: {}'.format(x220.shape))
        elif isinstance(x220, tuple):
            tuple_shapes = '('
            for item in x220:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x220: {}'.format(tuple_shapes))
        else:
            print('x220: {}'.format(x220))
        x221=self.conv2d72(x220)
        if x221 is None:
            print('x221: {}'.format(x221))
        elif isinstance(x221, torch.Tensor):
            print('x221: {}'.format(x221.shape))
        elif isinstance(x221, tuple):
            tuple_shapes = '('
            for item in x221:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x221: {}'.format(tuple_shapes))
        else:
            print('x221: {}'.format(x221))
        x222=self.sigmoid14(x221)
        if x222 is None:
            print('x222: {}'.format(x222))
        elif isinstance(x222, torch.Tensor):
            print('x222: {}'.format(x222.shape))
        elif isinstance(x222, tuple):
            tuple_shapes = '('
            for item in x222:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x222: {}'.format(tuple_shapes))
        else:
            print('x222: {}'.format(x222))
        x223=operator.mul(x222, x217)
        if x223 is None:
            print('x223: {}'.format(x223))
        elif isinstance(x223, torch.Tensor):
            print('x223: {}'.format(x223.shape))
        elif isinstance(x223, tuple):
            tuple_shapes = '('
            for item in x223:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x223: {}'.format(tuple_shapes))
        else:
            print('x223: {}'.format(x223))
        x224=self.conv2d73(x223)
        if x224 is None:
            print('x224: {}'.format(x224))
        elif isinstance(x224, torch.Tensor):
            print('x224: {}'.format(x224.shape))
        elif isinstance(x224, tuple):
            tuple_shapes = '('
            for item in x224:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x224: {}'.format(tuple_shapes))
        else:
            print('x224: {}'.format(x224))
        x225=self.batchnorm2d43(x224)
        if x225 is None:
            print('x225: {}'.format(x225))
        elif isinstance(x225, torch.Tensor):
            print('x225: {}'.format(x225.shape))
        elif isinstance(x225, tuple):
            tuple_shapes = '('
            for item in x225:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x225: {}'.format(tuple_shapes))
        else:
            print('x225: {}'.format(x225))
        x226=stochastic_depth(x225, 0.1076923076923077, 'row', False)
        if x226 is None:
            print('x226: {}'.format(x226))
        elif isinstance(x226, torch.Tensor):
            print('x226: {}'.format(x226.shape))
        elif isinstance(x226, tuple):
            tuple_shapes = '('
            for item in x226:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x226: {}'.format(tuple_shapes))
        else:
            print('x226: {}'.format(x226))
        x227=operator.add(x226, x211)
        if x227 is None:
            print('x227: {}'.format(x227))
        elif isinstance(x227, torch.Tensor):
            print('x227: {}'.format(x227.shape))
        elif isinstance(x227, tuple):
            tuple_shapes = '('
            for item in x227:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x227: {}'.format(tuple_shapes))
        else:
            print('x227: {}'.format(x227))
        x228=self.conv2d74(x227)
        if x228 is None:
            print('x228: {}'.format(x228))
        elif isinstance(x228, torch.Tensor):
            print('x228: {}'.format(x228.shape))
        elif isinstance(x228, tuple):
            tuple_shapes = '('
            for item in x228:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x228: {}'.format(tuple_shapes))
        else:
            print('x228: {}'.format(x228))
        x229=self.batchnorm2d44(x228)
        if x229 is None:
            print('x229: {}'.format(x229))
        elif isinstance(x229, torch.Tensor):
            print('x229: {}'.format(x229.shape))
        elif isinstance(x229, tuple):
            tuple_shapes = '('
            for item in x229:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x229: {}'.format(tuple_shapes))
        else:
            print('x229: {}'.format(x229))
        x230=self.silu44(x229)
        if x230 is None:
            print('x230: {}'.format(x230))
        elif isinstance(x230, torch.Tensor):
            print('x230: {}'.format(x230.shape))
        elif isinstance(x230, tuple):
            tuple_shapes = '('
            for item in x230:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x230: {}'.format(tuple_shapes))
        else:
            print('x230: {}'.format(x230))
        x231=self.conv2d75(x230)
        if x231 is None:
            print('x231: {}'.format(x231))
        elif isinstance(x231, torch.Tensor):
            print('x231: {}'.format(x231.shape))
        elif isinstance(x231, tuple):
            tuple_shapes = '('
            for item in x231:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x231: {}'.format(tuple_shapes))
        else:
            print('x231: {}'.format(x231))
        x232=self.batchnorm2d45(x231)
        if x232 is None:
            print('x232: {}'.format(x232))
        elif isinstance(x232, torch.Tensor):
            print('x232: {}'.format(x232.shape))
        elif isinstance(x232, tuple):
            tuple_shapes = '('
            for item in x232:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x232: {}'.format(tuple_shapes))
        else:
            print('x232: {}'.format(x232))
        x233=self.silu45(x232)
        if x233 is None:
            print('x233: {}'.format(x233))
        elif isinstance(x233, torch.Tensor):
            print('x233: {}'.format(x233.shape))
        elif isinstance(x233, tuple):
            tuple_shapes = '('
            for item in x233:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x233: {}'.format(tuple_shapes))
        else:
            print('x233: {}'.format(x233))
        x234=self.adaptiveavgpool2d15(x233)
        if x234 is None:
            print('x234: {}'.format(x234))
        elif isinstance(x234, torch.Tensor):
            print('x234: {}'.format(x234.shape))
        elif isinstance(x234, tuple):
            tuple_shapes = '('
            for item in x234:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x234: {}'.format(tuple_shapes))
        else:
            print('x234: {}'.format(x234))
        x235=self.conv2d76(x234)
        if x235 is None:
            print('x235: {}'.format(x235))
        elif isinstance(x235, torch.Tensor):
            print('x235: {}'.format(x235.shape))
        elif isinstance(x235, tuple):
            tuple_shapes = '('
            for item in x235:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x235: {}'.format(tuple_shapes))
        else:
            print('x235: {}'.format(x235))
        x236=self.silu46(x235)
        if x236 is None:
            print('x236: {}'.format(x236))
        elif isinstance(x236, torch.Tensor):
            print('x236: {}'.format(x236.shape))
        elif isinstance(x236, tuple):
            tuple_shapes = '('
            for item in x236:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x236: {}'.format(tuple_shapes))
        else:
            print('x236: {}'.format(x236))
        x237=self.conv2d77(x236)
        if x237 is None:
            print('x237: {}'.format(x237))
        elif isinstance(x237, torch.Tensor):
            print('x237: {}'.format(x237.shape))
        elif isinstance(x237, tuple):
            tuple_shapes = '('
            for item in x237:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x237: {}'.format(tuple_shapes))
        else:
            print('x237: {}'.format(x237))
        x238=self.sigmoid15(x237)
        if x238 is None:
            print('x238: {}'.format(x238))
        elif isinstance(x238, torch.Tensor):
            print('x238: {}'.format(x238.shape))
        elif isinstance(x238, tuple):
            tuple_shapes = '('
            for item in x238:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x238: {}'.format(tuple_shapes))
        else:
            print('x238: {}'.format(x238))
        x239=operator.mul(x238, x233)
        if x239 is None:
            print('x239: {}'.format(x239))
        elif isinstance(x239, torch.Tensor):
            print('x239: {}'.format(x239.shape))
        elif isinstance(x239, tuple):
            tuple_shapes = '('
            for item in x239:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x239: {}'.format(tuple_shapes))
        else:
            print('x239: {}'.format(x239))
        x240=self.conv2d78(x239)
        if x240 is None:
            print('x240: {}'.format(x240))
        elif isinstance(x240, torch.Tensor):
            print('x240: {}'.format(x240.shape))
        elif isinstance(x240, tuple):
            tuple_shapes = '('
            for item in x240:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x240: {}'.format(tuple_shapes))
        else:
            print('x240: {}'.format(x240))
        x241=self.batchnorm2d46(x240)
        if x241 is None:
            print('x241: {}'.format(x241))
        elif isinstance(x241, torch.Tensor):
            print('x241: {}'.format(x241.shape))
        elif isinstance(x241, tuple):
            tuple_shapes = '('
            for item in x241:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x241: {}'.format(tuple_shapes))
        else:
            print('x241: {}'.format(x241))
        x242=stochastic_depth(x241, 0.11538461538461539, 'row', False)
        if x242 is None:
            print('x242: {}'.format(x242))
        elif isinstance(x242, torch.Tensor):
            print('x242: {}'.format(x242.shape))
        elif isinstance(x242, tuple):
            tuple_shapes = '('
            for item in x242:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x242: {}'.format(tuple_shapes))
        else:
            print('x242: {}'.format(x242))
        x243=operator.add(x242, x227)
        if x243 is None:
            print('x243: {}'.format(x243))
        elif isinstance(x243, torch.Tensor):
            print('x243: {}'.format(x243.shape))
        elif isinstance(x243, tuple):
            tuple_shapes = '('
            for item in x243:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x243: {}'.format(tuple_shapes))
        else:
            print('x243: {}'.format(x243))
        x244=self.conv2d79(x243)
        if x244 is None:
            print('x244: {}'.format(x244))
        elif isinstance(x244, torch.Tensor):
            print('x244: {}'.format(x244.shape))
        elif isinstance(x244, tuple):
            tuple_shapes = '('
            for item in x244:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x244: {}'.format(tuple_shapes))
        else:
            print('x244: {}'.format(x244))
        x245=self.batchnorm2d47(x244)
        if x245 is None:
            print('x245: {}'.format(x245))
        elif isinstance(x245, torch.Tensor):
            print('x245: {}'.format(x245.shape))
        elif isinstance(x245, tuple):
            tuple_shapes = '('
            for item in x245:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x245: {}'.format(tuple_shapes))
        else:
            print('x245: {}'.format(x245))
        x246=self.silu47(x245)
        if x246 is None:
            print('x246: {}'.format(x246))
        elif isinstance(x246, torch.Tensor):
            print('x246: {}'.format(x246.shape))
        elif isinstance(x246, tuple):
            tuple_shapes = '('
            for item in x246:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x246: {}'.format(tuple_shapes))
        else:
            print('x246: {}'.format(x246))
        x247=self.conv2d80(x246)
        if x247 is None:
            print('x247: {}'.format(x247))
        elif isinstance(x247, torch.Tensor):
            print('x247: {}'.format(x247.shape))
        elif isinstance(x247, tuple):
            tuple_shapes = '('
            for item in x247:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x247: {}'.format(tuple_shapes))
        else:
            print('x247: {}'.format(x247))
        x248=self.batchnorm2d48(x247)
        if x248 is None:
            print('x248: {}'.format(x248))
        elif isinstance(x248, torch.Tensor):
            print('x248: {}'.format(x248.shape))
        elif isinstance(x248, tuple):
            tuple_shapes = '('
            for item in x248:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x248: {}'.format(tuple_shapes))
        else:
            print('x248: {}'.format(x248))
        x249=self.silu48(x248)
        if x249 is None:
            print('x249: {}'.format(x249))
        elif isinstance(x249, torch.Tensor):
            print('x249: {}'.format(x249.shape))
        elif isinstance(x249, tuple):
            tuple_shapes = '('
            for item in x249:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x249: {}'.format(tuple_shapes))
        else:
            print('x249: {}'.format(x249))
        x250=self.adaptiveavgpool2d16(x249)
        if x250 is None:
            print('x250: {}'.format(x250))
        elif isinstance(x250, torch.Tensor):
            print('x250: {}'.format(x250.shape))
        elif isinstance(x250, tuple):
            tuple_shapes = '('
            for item in x250:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x250: {}'.format(tuple_shapes))
        else:
            print('x250: {}'.format(x250))
        x251=self.conv2d81(x250)
        if x251 is None:
            print('x251: {}'.format(x251))
        elif isinstance(x251, torch.Tensor):
            print('x251: {}'.format(x251.shape))
        elif isinstance(x251, tuple):
            tuple_shapes = '('
            for item in x251:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x251: {}'.format(tuple_shapes))
        else:
            print('x251: {}'.format(x251))
        x252=self.silu49(x251)
        if x252 is None:
            print('x252: {}'.format(x252))
        elif isinstance(x252, torch.Tensor):
            print('x252: {}'.format(x252.shape))
        elif isinstance(x252, tuple):
            tuple_shapes = '('
            for item in x252:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x252: {}'.format(tuple_shapes))
        else:
            print('x252: {}'.format(x252))
        x253=self.conv2d82(x252)
        if x253 is None:
            print('x253: {}'.format(x253))
        elif isinstance(x253, torch.Tensor):
            print('x253: {}'.format(x253.shape))
        elif isinstance(x253, tuple):
            tuple_shapes = '('
            for item in x253:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x253: {}'.format(tuple_shapes))
        else:
            print('x253: {}'.format(x253))
        x254=self.sigmoid16(x253)
        if x254 is None:
            print('x254: {}'.format(x254))
        elif isinstance(x254, torch.Tensor):
            print('x254: {}'.format(x254.shape))
        elif isinstance(x254, tuple):
            tuple_shapes = '('
            for item in x254:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x254: {}'.format(tuple_shapes))
        else:
            print('x254: {}'.format(x254))
        x255=operator.mul(x254, x249)
        if x255 is None:
            print('x255: {}'.format(x255))
        elif isinstance(x255, torch.Tensor):
            print('x255: {}'.format(x255.shape))
        elif isinstance(x255, tuple):
            tuple_shapes = '('
            for item in x255:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x255: {}'.format(tuple_shapes))
        else:
            print('x255: {}'.format(x255))
        x256=self.conv2d83(x255)
        if x256 is None:
            print('x256: {}'.format(x256))
        elif isinstance(x256, torch.Tensor):
            print('x256: {}'.format(x256.shape))
        elif isinstance(x256, tuple):
            tuple_shapes = '('
            for item in x256:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x256: {}'.format(tuple_shapes))
        else:
            print('x256: {}'.format(x256))
        x257=self.batchnorm2d49(x256)
        if x257 is None:
            print('x257: {}'.format(x257))
        elif isinstance(x257, torch.Tensor):
            print('x257: {}'.format(x257.shape))
        elif isinstance(x257, tuple):
            tuple_shapes = '('
            for item in x257:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x257: {}'.format(tuple_shapes))
        else:
            print('x257: {}'.format(x257))
        x258=stochastic_depth(x257, 0.12307692307692308, 'row', False)
        if x258 is None:
            print('x258: {}'.format(x258))
        elif isinstance(x258, torch.Tensor):
            print('x258: {}'.format(x258.shape))
        elif isinstance(x258, tuple):
            tuple_shapes = '('
            for item in x258:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x258: {}'.format(tuple_shapes))
        else:
            print('x258: {}'.format(x258))
        x259=operator.add(x258, x243)
        if x259 is None:
            print('x259: {}'.format(x259))
        elif isinstance(x259, torch.Tensor):
            print('x259: {}'.format(x259.shape))
        elif isinstance(x259, tuple):
            tuple_shapes = '('
            for item in x259:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x259: {}'.format(tuple_shapes))
        else:
            print('x259: {}'.format(x259))
        x260=self.conv2d84(x259)
        if x260 is None:
            print('x260: {}'.format(x260))
        elif isinstance(x260, torch.Tensor):
            print('x260: {}'.format(x260.shape))
        elif isinstance(x260, tuple):
            tuple_shapes = '('
            for item in x260:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x260: {}'.format(tuple_shapes))
        else:
            print('x260: {}'.format(x260))
        x261=self.batchnorm2d50(x260)
        if x261 is None:
            print('x261: {}'.format(x261))
        elif isinstance(x261, torch.Tensor):
            print('x261: {}'.format(x261.shape))
        elif isinstance(x261, tuple):
            tuple_shapes = '('
            for item in x261:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x261: {}'.format(tuple_shapes))
        else:
            print('x261: {}'.format(x261))
        x262=self.silu50(x261)
        if x262 is None:
            print('x262: {}'.format(x262))
        elif isinstance(x262, torch.Tensor):
            print('x262: {}'.format(x262.shape))
        elif isinstance(x262, tuple):
            tuple_shapes = '('
            for item in x262:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x262: {}'.format(tuple_shapes))
        else:
            print('x262: {}'.format(x262))
        x263=self.conv2d85(x262)
        if x263 is None:
            print('x263: {}'.format(x263))
        elif isinstance(x263, torch.Tensor):
            print('x263: {}'.format(x263.shape))
        elif isinstance(x263, tuple):
            tuple_shapes = '('
            for item in x263:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x263: {}'.format(tuple_shapes))
        else:
            print('x263: {}'.format(x263))
        x264=self.batchnorm2d51(x263)
        if x264 is None:
            print('x264: {}'.format(x264))
        elif isinstance(x264, torch.Tensor):
            print('x264: {}'.format(x264.shape))
        elif isinstance(x264, tuple):
            tuple_shapes = '('
            for item in x264:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x264: {}'.format(tuple_shapes))
        else:
            print('x264: {}'.format(x264))
        x265=self.silu51(x264)
        if x265 is None:
            print('x265: {}'.format(x265))
        elif isinstance(x265, torch.Tensor):
            print('x265: {}'.format(x265.shape))
        elif isinstance(x265, tuple):
            tuple_shapes = '('
            for item in x265:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x265: {}'.format(tuple_shapes))
        else:
            print('x265: {}'.format(x265))
        x266=self.adaptiveavgpool2d17(x265)
        if x266 is None:
            print('x266: {}'.format(x266))
        elif isinstance(x266, torch.Tensor):
            print('x266: {}'.format(x266.shape))
        elif isinstance(x266, tuple):
            tuple_shapes = '('
            for item in x266:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x266: {}'.format(tuple_shapes))
        else:
            print('x266: {}'.format(x266))
        x267=self.conv2d86(x266)
        if x267 is None:
            print('x267: {}'.format(x267))
        elif isinstance(x267, torch.Tensor):
            print('x267: {}'.format(x267.shape))
        elif isinstance(x267, tuple):
            tuple_shapes = '('
            for item in x267:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x267: {}'.format(tuple_shapes))
        else:
            print('x267: {}'.format(x267))
        x268=self.silu52(x267)
        if x268 is None:
            print('x268: {}'.format(x268))
        elif isinstance(x268, torch.Tensor):
            print('x268: {}'.format(x268.shape))
        elif isinstance(x268, tuple):
            tuple_shapes = '('
            for item in x268:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x268: {}'.format(tuple_shapes))
        else:
            print('x268: {}'.format(x268))
        x269=self.conv2d87(x268)
        if x269 is None:
            print('x269: {}'.format(x269))
        elif isinstance(x269, torch.Tensor):
            print('x269: {}'.format(x269.shape))
        elif isinstance(x269, tuple):
            tuple_shapes = '('
            for item in x269:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x269: {}'.format(tuple_shapes))
        else:
            print('x269: {}'.format(x269))
        x270=self.sigmoid17(x269)
        if x270 is None:
            print('x270: {}'.format(x270))
        elif isinstance(x270, torch.Tensor):
            print('x270: {}'.format(x270.shape))
        elif isinstance(x270, tuple):
            tuple_shapes = '('
            for item in x270:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x270: {}'.format(tuple_shapes))
        else:
            print('x270: {}'.format(x270))
        x271=operator.mul(x270, x265)
        if x271 is None:
            print('x271: {}'.format(x271))
        elif isinstance(x271, torch.Tensor):
            print('x271: {}'.format(x271.shape))
        elif isinstance(x271, tuple):
            tuple_shapes = '('
            for item in x271:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x271: {}'.format(tuple_shapes))
        else:
            print('x271: {}'.format(x271))
        x272=self.conv2d88(x271)
        if x272 is None:
            print('x272: {}'.format(x272))
        elif isinstance(x272, torch.Tensor):
            print('x272: {}'.format(x272.shape))
        elif isinstance(x272, tuple):
            tuple_shapes = '('
            for item in x272:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x272: {}'.format(tuple_shapes))
        else:
            print('x272: {}'.format(x272))
        x273=self.batchnorm2d52(x272)
        if x273 is None:
            print('x273: {}'.format(x273))
        elif isinstance(x273, torch.Tensor):
            print('x273: {}'.format(x273.shape))
        elif isinstance(x273, tuple):
            tuple_shapes = '('
            for item in x273:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x273: {}'.format(tuple_shapes))
        else:
            print('x273: {}'.format(x273))
        x274=stochastic_depth(x273, 0.13076923076923078, 'row', False)
        if x274 is None:
            print('x274: {}'.format(x274))
        elif isinstance(x274, torch.Tensor):
            print('x274: {}'.format(x274.shape))
        elif isinstance(x274, tuple):
            tuple_shapes = '('
            for item in x274:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x274: {}'.format(tuple_shapes))
        else:
            print('x274: {}'.format(x274))
        x275=operator.add(x274, x259)
        if x275 is None:
            print('x275: {}'.format(x275))
        elif isinstance(x275, torch.Tensor):
            print('x275: {}'.format(x275.shape))
        elif isinstance(x275, tuple):
            tuple_shapes = '('
            for item in x275:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x275: {}'.format(tuple_shapes))
        else:
            print('x275: {}'.format(x275))
        x276=self.conv2d89(x275)
        if x276 is None:
            print('x276: {}'.format(x276))
        elif isinstance(x276, torch.Tensor):
            print('x276: {}'.format(x276.shape))
        elif isinstance(x276, tuple):
            tuple_shapes = '('
            for item in x276:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x276: {}'.format(tuple_shapes))
        else:
            print('x276: {}'.format(x276))
        x277=self.batchnorm2d53(x276)
        if x277 is None:
            print('x277: {}'.format(x277))
        elif isinstance(x277, torch.Tensor):
            print('x277: {}'.format(x277.shape))
        elif isinstance(x277, tuple):
            tuple_shapes = '('
            for item in x277:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x277: {}'.format(tuple_shapes))
        else:
            print('x277: {}'.format(x277))
        x278=self.silu53(x277)
        if x278 is None:
            print('x278: {}'.format(x278))
        elif isinstance(x278, torch.Tensor):
            print('x278: {}'.format(x278.shape))
        elif isinstance(x278, tuple):
            tuple_shapes = '('
            for item in x278:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x278: {}'.format(tuple_shapes))
        else:
            print('x278: {}'.format(x278))
        x279=self.conv2d90(x278)
        if x279 is None:
            print('x279: {}'.format(x279))
        elif isinstance(x279, torch.Tensor):
            print('x279: {}'.format(x279.shape))
        elif isinstance(x279, tuple):
            tuple_shapes = '('
            for item in x279:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x279: {}'.format(tuple_shapes))
        else:
            print('x279: {}'.format(x279))
        x280=self.batchnorm2d54(x279)
        if x280 is None:
            print('x280: {}'.format(x280))
        elif isinstance(x280, torch.Tensor):
            print('x280: {}'.format(x280.shape))
        elif isinstance(x280, tuple):
            tuple_shapes = '('
            for item in x280:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x280: {}'.format(tuple_shapes))
        else:
            print('x280: {}'.format(x280))
        x281=self.silu54(x280)
        if x281 is None:
            print('x281: {}'.format(x281))
        elif isinstance(x281, torch.Tensor):
            print('x281: {}'.format(x281.shape))
        elif isinstance(x281, tuple):
            tuple_shapes = '('
            for item in x281:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x281: {}'.format(tuple_shapes))
        else:
            print('x281: {}'.format(x281))
        x282=self.adaptiveavgpool2d18(x281)
        if x282 is None:
            print('x282: {}'.format(x282))
        elif isinstance(x282, torch.Tensor):
            print('x282: {}'.format(x282.shape))
        elif isinstance(x282, tuple):
            tuple_shapes = '('
            for item in x282:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x282: {}'.format(tuple_shapes))
        else:
            print('x282: {}'.format(x282))
        x283=self.conv2d91(x282)
        if x283 is None:
            print('x283: {}'.format(x283))
        elif isinstance(x283, torch.Tensor):
            print('x283: {}'.format(x283.shape))
        elif isinstance(x283, tuple):
            tuple_shapes = '('
            for item in x283:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x283: {}'.format(tuple_shapes))
        else:
            print('x283: {}'.format(x283))
        x284=self.silu55(x283)
        if x284 is None:
            print('x284: {}'.format(x284))
        elif isinstance(x284, torch.Tensor):
            print('x284: {}'.format(x284.shape))
        elif isinstance(x284, tuple):
            tuple_shapes = '('
            for item in x284:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x284: {}'.format(tuple_shapes))
        else:
            print('x284: {}'.format(x284))
        x285=self.conv2d92(x284)
        if x285 is None:
            print('x285: {}'.format(x285))
        elif isinstance(x285, torch.Tensor):
            print('x285: {}'.format(x285.shape))
        elif isinstance(x285, tuple):
            tuple_shapes = '('
            for item in x285:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x285: {}'.format(tuple_shapes))
        else:
            print('x285: {}'.format(x285))
        x286=self.sigmoid18(x285)
        if x286 is None:
            print('x286: {}'.format(x286))
        elif isinstance(x286, torch.Tensor):
            print('x286: {}'.format(x286.shape))
        elif isinstance(x286, tuple):
            tuple_shapes = '('
            for item in x286:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x286: {}'.format(tuple_shapes))
        else:
            print('x286: {}'.format(x286))
        x287=operator.mul(x286, x281)
        if x287 is None:
            print('x287: {}'.format(x287))
        elif isinstance(x287, torch.Tensor):
            print('x287: {}'.format(x287.shape))
        elif isinstance(x287, tuple):
            tuple_shapes = '('
            for item in x287:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x287: {}'.format(tuple_shapes))
        else:
            print('x287: {}'.format(x287))
        x288=self.conv2d93(x287)
        if x288 is None:
            print('x288: {}'.format(x288))
        elif isinstance(x288, torch.Tensor):
            print('x288: {}'.format(x288.shape))
        elif isinstance(x288, tuple):
            tuple_shapes = '('
            for item in x288:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x288: {}'.format(tuple_shapes))
        else:
            print('x288: {}'.format(x288))
        x289=self.batchnorm2d55(x288)
        if x289 is None:
            print('x289: {}'.format(x289))
        elif isinstance(x289, torch.Tensor):
            print('x289: {}'.format(x289.shape))
        elif isinstance(x289, tuple):
            tuple_shapes = '('
            for item in x289:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x289: {}'.format(tuple_shapes))
        else:
            print('x289: {}'.format(x289))
        x290=self.conv2d94(x289)
        if x290 is None:
            print('x290: {}'.format(x290))
        elif isinstance(x290, torch.Tensor):
            print('x290: {}'.format(x290.shape))
        elif isinstance(x290, tuple):
            tuple_shapes = '('
            for item in x290:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x290: {}'.format(tuple_shapes))
        else:
            print('x290: {}'.format(x290))
        x291=self.batchnorm2d56(x290)
        if x291 is None:
            print('x291: {}'.format(x291))
        elif isinstance(x291, torch.Tensor):
            print('x291: {}'.format(x291.shape))
        elif isinstance(x291, tuple):
            tuple_shapes = '('
            for item in x291:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x291: {}'.format(tuple_shapes))
        else:
            print('x291: {}'.format(x291))
        x292=self.silu56(x291)
        if x292 is None:
            print('x292: {}'.format(x292))
        elif isinstance(x292, torch.Tensor):
            print('x292: {}'.format(x292.shape))
        elif isinstance(x292, tuple):
            tuple_shapes = '('
            for item in x292:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x292: {}'.format(tuple_shapes))
        else:
            print('x292: {}'.format(x292))
        x293=self.conv2d95(x292)
        if x293 is None:
            print('x293: {}'.format(x293))
        elif isinstance(x293, torch.Tensor):
            print('x293: {}'.format(x293.shape))
        elif isinstance(x293, tuple):
            tuple_shapes = '('
            for item in x293:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x293: {}'.format(tuple_shapes))
        else:
            print('x293: {}'.format(x293))
        x294=self.batchnorm2d57(x293)
        if x294 is None:
            print('x294: {}'.format(x294))
        elif isinstance(x294, torch.Tensor):
            print('x294: {}'.format(x294.shape))
        elif isinstance(x294, tuple):
            tuple_shapes = '('
            for item in x294:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x294: {}'.format(tuple_shapes))
        else:
            print('x294: {}'.format(x294))
        x295=self.silu57(x294)
        if x295 is None:
            print('x295: {}'.format(x295))
        elif isinstance(x295, torch.Tensor):
            print('x295: {}'.format(x295.shape))
        elif isinstance(x295, tuple):
            tuple_shapes = '('
            for item in x295:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x295: {}'.format(tuple_shapes))
        else:
            print('x295: {}'.format(x295))
        x296=self.adaptiveavgpool2d19(x295)
        if x296 is None:
            print('x296: {}'.format(x296))
        elif isinstance(x296, torch.Tensor):
            print('x296: {}'.format(x296.shape))
        elif isinstance(x296, tuple):
            tuple_shapes = '('
            for item in x296:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x296: {}'.format(tuple_shapes))
        else:
            print('x296: {}'.format(x296))
        x297=self.conv2d96(x296)
        if x297 is None:
            print('x297: {}'.format(x297))
        elif isinstance(x297, torch.Tensor):
            print('x297: {}'.format(x297.shape))
        elif isinstance(x297, tuple):
            tuple_shapes = '('
            for item in x297:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x297: {}'.format(tuple_shapes))
        else:
            print('x297: {}'.format(x297))
        x298=self.silu58(x297)
        if x298 is None:
            print('x298: {}'.format(x298))
        elif isinstance(x298, torch.Tensor):
            print('x298: {}'.format(x298.shape))
        elif isinstance(x298, tuple):
            tuple_shapes = '('
            for item in x298:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x298: {}'.format(tuple_shapes))
        else:
            print('x298: {}'.format(x298))
        x299=self.conv2d97(x298)
        if x299 is None:
            print('x299: {}'.format(x299))
        elif isinstance(x299, torch.Tensor):
            print('x299: {}'.format(x299.shape))
        elif isinstance(x299, tuple):
            tuple_shapes = '('
            for item in x299:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x299: {}'.format(tuple_shapes))
        else:
            print('x299: {}'.format(x299))
        x300=self.sigmoid19(x299)
        if x300 is None:
            print('x300: {}'.format(x300))
        elif isinstance(x300, torch.Tensor):
            print('x300: {}'.format(x300.shape))
        elif isinstance(x300, tuple):
            tuple_shapes = '('
            for item in x300:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x300: {}'.format(tuple_shapes))
        else:
            print('x300: {}'.format(x300))
        x301=operator.mul(x300, x295)
        if x301 is None:
            print('x301: {}'.format(x301))
        elif isinstance(x301, torch.Tensor):
            print('x301: {}'.format(x301.shape))
        elif isinstance(x301, tuple):
            tuple_shapes = '('
            for item in x301:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x301: {}'.format(tuple_shapes))
        else:
            print('x301: {}'.format(x301))
        x302=self.conv2d98(x301)
        if x302 is None:
            print('x302: {}'.format(x302))
        elif isinstance(x302, torch.Tensor):
            print('x302: {}'.format(x302.shape))
        elif isinstance(x302, tuple):
            tuple_shapes = '('
            for item in x302:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x302: {}'.format(tuple_shapes))
        else:
            print('x302: {}'.format(x302))
        x303=self.batchnorm2d58(x302)
        if x303 is None:
            print('x303: {}'.format(x303))
        elif isinstance(x303, torch.Tensor):
            print('x303: {}'.format(x303.shape))
        elif isinstance(x303, tuple):
            tuple_shapes = '('
            for item in x303:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x303: {}'.format(tuple_shapes))
        else:
            print('x303: {}'.format(x303))
        x304=stochastic_depth(x303, 0.14615384615384616, 'row', False)
        if x304 is None:
            print('x304: {}'.format(x304))
        elif isinstance(x304, torch.Tensor):
            print('x304: {}'.format(x304.shape))
        elif isinstance(x304, tuple):
            tuple_shapes = '('
            for item in x304:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x304: {}'.format(tuple_shapes))
        else:
            print('x304: {}'.format(x304))
        x305=operator.add(x304, x289)
        if x305 is None:
            print('x305: {}'.format(x305))
        elif isinstance(x305, torch.Tensor):
            print('x305: {}'.format(x305.shape))
        elif isinstance(x305, tuple):
            tuple_shapes = '('
            for item in x305:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x305: {}'.format(tuple_shapes))
        else:
            print('x305: {}'.format(x305))
        x306=self.conv2d99(x305)
        if x306 is None:
            print('x306: {}'.format(x306))
        elif isinstance(x306, torch.Tensor):
            print('x306: {}'.format(x306.shape))
        elif isinstance(x306, tuple):
            tuple_shapes = '('
            for item in x306:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x306: {}'.format(tuple_shapes))
        else:
            print('x306: {}'.format(x306))
        x307=self.batchnorm2d59(x306)
        if x307 is None:
            print('x307: {}'.format(x307))
        elif isinstance(x307, torch.Tensor):
            print('x307: {}'.format(x307.shape))
        elif isinstance(x307, tuple):
            tuple_shapes = '('
            for item in x307:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x307: {}'.format(tuple_shapes))
        else:
            print('x307: {}'.format(x307))
        x308=self.silu59(x307)
        if x308 is None:
            print('x308: {}'.format(x308))
        elif isinstance(x308, torch.Tensor):
            print('x308: {}'.format(x308.shape))
        elif isinstance(x308, tuple):
            tuple_shapes = '('
            for item in x308:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x308: {}'.format(tuple_shapes))
        else:
            print('x308: {}'.format(x308))
        x309=self.conv2d100(x308)
        if x309 is None:
            print('x309: {}'.format(x309))
        elif isinstance(x309, torch.Tensor):
            print('x309: {}'.format(x309.shape))
        elif isinstance(x309, tuple):
            tuple_shapes = '('
            for item in x309:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x309: {}'.format(tuple_shapes))
        else:
            print('x309: {}'.format(x309))
        x310=self.batchnorm2d60(x309)
        if x310 is None:
            print('x310: {}'.format(x310))
        elif isinstance(x310, torch.Tensor):
            print('x310: {}'.format(x310.shape))
        elif isinstance(x310, tuple):
            tuple_shapes = '('
            for item in x310:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x310: {}'.format(tuple_shapes))
        else:
            print('x310: {}'.format(x310))
        x311=self.silu60(x310)
        if x311 is None:
            print('x311: {}'.format(x311))
        elif isinstance(x311, torch.Tensor):
            print('x311: {}'.format(x311.shape))
        elif isinstance(x311, tuple):
            tuple_shapes = '('
            for item in x311:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x311: {}'.format(tuple_shapes))
        else:
            print('x311: {}'.format(x311))
        x312=self.adaptiveavgpool2d20(x311)
        if x312 is None:
            print('x312: {}'.format(x312))
        elif isinstance(x312, torch.Tensor):
            print('x312: {}'.format(x312.shape))
        elif isinstance(x312, tuple):
            tuple_shapes = '('
            for item in x312:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x312: {}'.format(tuple_shapes))
        else:
            print('x312: {}'.format(x312))
        x313=self.conv2d101(x312)
        if x313 is None:
            print('x313: {}'.format(x313))
        elif isinstance(x313, torch.Tensor):
            print('x313: {}'.format(x313.shape))
        elif isinstance(x313, tuple):
            tuple_shapes = '('
            for item in x313:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x313: {}'.format(tuple_shapes))
        else:
            print('x313: {}'.format(x313))
        x314=self.silu61(x313)
        if x314 is None:
            print('x314: {}'.format(x314))
        elif isinstance(x314, torch.Tensor):
            print('x314: {}'.format(x314.shape))
        elif isinstance(x314, tuple):
            tuple_shapes = '('
            for item in x314:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x314: {}'.format(tuple_shapes))
        else:
            print('x314: {}'.format(x314))
        x315=self.conv2d102(x314)
        if x315 is None:
            print('x315: {}'.format(x315))
        elif isinstance(x315, torch.Tensor):
            print('x315: {}'.format(x315.shape))
        elif isinstance(x315, tuple):
            tuple_shapes = '('
            for item in x315:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x315: {}'.format(tuple_shapes))
        else:
            print('x315: {}'.format(x315))
        x316=self.sigmoid20(x315)
        if x316 is None:
            print('x316: {}'.format(x316))
        elif isinstance(x316, torch.Tensor):
            print('x316: {}'.format(x316.shape))
        elif isinstance(x316, tuple):
            tuple_shapes = '('
            for item in x316:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x316: {}'.format(tuple_shapes))
        else:
            print('x316: {}'.format(x316))
        x317=operator.mul(x316, x311)
        if x317 is None:
            print('x317: {}'.format(x317))
        elif isinstance(x317, torch.Tensor):
            print('x317: {}'.format(x317.shape))
        elif isinstance(x317, tuple):
            tuple_shapes = '('
            for item in x317:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x317: {}'.format(tuple_shapes))
        else:
            print('x317: {}'.format(x317))
        x318=self.conv2d103(x317)
        if x318 is None:
            print('x318: {}'.format(x318))
        elif isinstance(x318, torch.Tensor):
            print('x318: {}'.format(x318.shape))
        elif isinstance(x318, tuple):
            tuple_shapes = '('
            for item in x318:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x318: {}'.format(tuple_shapes))
        else:
            print('x318: {}'.format(x318))
        x319=self.batchnorm2d61(x318)
        if x319 is None:
            print('x319: {}'.format(x319))
        elif isinstance(x319, torch.Tensor):
            print('x319: {}'.format(x319.shape))
        elif isinstance(x319, tuple):
            tuple_shapes = '('
            for item in x319:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x319: {}'.format(tuple_shapes))
        else:
            print('x319: {}'.format(x319))
        x320=stochastic_depth(x319, 0.15384615384615385, 'row', False)
        if x320 is None:
            print('x320: {}'.format(x320))
        elif isinstance(x320, torch.Tensor):
            print('x320: {}'.format(x320.shape))
        elif isinstance(x320, tuple):
            tuple_shapes = '('
            for item in x320:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x320: {}'.format(tuple_shapes))
        else:
            print('x320: {}'.format(x320))
        x321=operator.add(x320, x305)
        if x321 is None:
            print('x321: {}'.format(x321))
        elif isinstance(x321, torch.Tensor):
            print('x321: {}'.format(x321.shape))
        elif isinstance(x321, tuple):
            tuple_shapes = '('
            for item in x321:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x321: {}'.format(tuple_shapes))
        else:
            print('x321: {}'.format(x321))
        x322=self.conv2d104(x321)
        if x322 is None:
            print('x322: {}'.format(x322))
        elif isinstance(x322, torch.Tensor):
            print('x322: {}'.format(x322.shape))
        elif isinstance(x322, tuple):
            tuple_shapes = '('
            for item in x322:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x322: {}'.format(tuple_shapes))
        else:
            print('x322: {}'.format(x322))
        x323=self.batchnorm2d62(x322)
        if x323 is None:
            print('x323: {}'.format(x323))
        elif isinstance(x323, torch.Tensor):
            print('x323: {}'.format(x323.shape))
        elif isinstance(x323, tuple):
            tuple_shapes = '('
            for item in x323:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x323: {}'.format(tuple_shapes))
        else:
            print('x323: {}'.format(x323))
        x324=self.silu62(x323)
        if x324 is None:
            print('x324: {}'.format(x324))
        elif isinstance(x324, torch.Tensor):
            print('x324: {}'.format(x324.shape))
        elif isinstance(x324, tuple):
            tuple_shapes = '('
            for item in x324:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x324: {}'.format(tuple_shapes))
        else:
            print('x324: {}'.format(x324))
        x325=self.conv2d105(x324)
        if x325 is None:
            print('x325: {}'.format(x325))
        elif isinstance(x325, torch.Tensor):
            print('x325: {}'.format(x325.shape))
        elif isinstance(x325, tuple):
            tuple_shapes = '('
            for item in x325:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x325: {}'.format(tuple_shapes))
        else:
            print('x325: {}'.format(x325))
        x326=self.batchnorm2d63(x325)
        if x326 is None:
            print('x326: {}'.format(x326))
        elif isinstance(x326, torch.Tensor):
            print('x326: {}'.format(x326.shape))
        elif isinstance(x326, tuple):
            tuple_shapes = '('
            for item in x326:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x326: {}'.format(tuple_shapes))
        else:
            print('x326: {}'.format(x326))
        x327=self.silu63(x326)
        if x327 is None:
            print('x327: {}'.format(x327))
        elif isinstance(x327, torch.Tensor):
            print('x327: {}'.format(x327.shape))
        elif isinstance(x327, tuple):
            tuple_shapes = '('
            for item in x327:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x327: {}'.format(tuple_shapes))
        else:
            print('x327: {}'.format(x327))
        x328=self.adaptiveavgpool2d21(x327)
        if x328 is None:
            print('x328: {}'.format(x328))
        elif isinstance(x328, torch.Tensor):
            print('x328: {}'.format(x328.shape))
        elif isinstance(x328, tuple):
            tuple_shapes = '('
            for item in x328:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x328: {}'.format(tuple_shapes))
        else:
            print('x328: {}'.format(x328))
        x329=self.conv2d106(x328)
        if x329 is None:
            print('x329: {}'.format(x329))
        elif isinstance(x329, torch.Tensor):
            print('x329: {}'.format(x329.shape))
        elif isinstance(x329, tuple):
            tuple_shapes = '('
            for item in x329:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x329: {}'.format(tuple_shapes))
        else:
            print('x329: {}'.format(x329))
        x330=self.silu64(x329)
        if x330 is None:
            print('x330: {}'.format(x330))
        elif isinstance(x330, torch.Tensor):
            print('x330: {}'.format(x330.shape))
        elif isinstance(x330, tuple):
            tuple_shapes = '('
            for item in x330:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x330: {}'.format(tuple_shapes))
        else:
            print('x330: {}'.format(x330))
        x331=self.conv2d107(x330)
        if x331 is None:
            print('x331: {}'.format(x331))
        elif isinstance(x331, torch.Tensor):
            print('x331: {}'.format(x331.shape))
        elif isinstance(x331, tuple):
            tuple_shapes = '('
            for item in x331:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x331: {}'.format(tuple_shapes))
        else:
            print('x331: {}'.format(x331))
        x332=self.sigmoid21(x331)
        if x332 is None:
            print('x332: {}'.format(x332))
        elif isinstance(x332, torch.Tensor):
            print('x332: {}'.format(x332.shape))
        elif isinstance(x332, tuple):
            tuple_shapes = '('
            for item in x332:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x332: {}'.format(tuple_shapes))
        else:
            print('x332: {}'.format(x332))
        x333=operator.mul(x332, x327)
        if x333 is None:
            print('x333: {}'.format(x333))
        elif isinstance(x333, torch.Tensor):
            print('x333: {}'.format(x333.shape))
        elif isinstance(x333, tuple):
            tuple_shapes = '('
            for item in x333:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x333: {}'.format(tuple_shapes))
        else:
            print('x333: {}'.format(x333))
        x334=self.conv2d108(x333)
        if x334 is None:
            print('x334: {}'.format(x334))
        elif isinstance(x334, torch.Tensor):
            print('x334: {}'.format(x334.shape))
        elif isinstance(x334, tuple):
            tuple_shapes = '('
            for item in x334:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x334: {}'.format(tuple_shapes))
        else:
            print('x334: {}'.format(x334))
        x335=self.batchnorm2d64(x334)
        if x335 is None:
            print('x335: {}'.format(x335))
        elif isinstance(x335, torch.Tensor):
            print('x335: {}'.format(x335.shape))
        elif isinstance(x335, tuple):
            tuple_shapes = '('
            for item in x335:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x335: {}'.format(tuple_shapes))
        else:
            print('x335: {}'.format(x335))
        x336=stochastic_depth(x335, 0.16153846153846155, 'row', False)
        if x336 is None:
            print('x336: {}'.format(x336))
        elif isinstance(x336, torch.Tensor):
            print('x336: {}'.format(x336.shape))
        elif isinstance(x336, tuple):
            tuple_shapes = '('
            for item in x336:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x336: {}'.format(tuple_shapes))
        else:
            print('x336: {}'.format(x336))
        x337=operator.add(x336, x321)
        if x337 is None:
            print('x337: {}'.format(x337))
        elif isinstance(x337, torch.Tensor):
            print('x337: {}'.format(x337.shape))
        elif isinstance(x337, tuple):
            tuple_shapes = '('
            for item in x337:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x337: {}'.format(tuple_shapes))
        else:
            print('x337: {}'.format(x337))
        x338=self.conv2d109(x337)
        if x338 is None:
            print('x338: {}'.format(x338))
        elif isinstance(x338, torch.Tensor):
            print('x338: {}'.format(x338.shape))
        elif isinstance(x338, tuple):
            tuple_shapes = '('
            for item in x338:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x338: {}'.format(tuple_shapes))
        else:
            print('x338: {}'.format(x338))
        x339=self.batchnorm2d65(x338)
        if x339 is None:
            print('x339: {}'.format(x339))
        elif isinstance(x339, torch.Tensor):
            print('x339: {}'.format(x339.shape))
        elif isinstance(x339, tuple):
            tuple_shapes = '('
            for item in x339:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x339: {}'.format(tuple_shapes))
        else:
            print('x339: {}'.format(x339))
        x340=self.silu65(x339)
        if x340 is None:
            print('x340: {}'.format(x340))
        elif isinstance(x340, torch.Tensor):
            print('x340: {}'.format(x340.shape))
        elif isinstance(x340, tuple):
            tuple_shapes = '('
            for item in x340:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x340: {}'.format(tuple_shapes))
        else:
            print('x340: {}'.format(x340))
        x341=self.conv2d110(x340)
        if x341 is None:
            print('x341: {}'.format(x341))
        elif isinstance(x341, torch.Tensor):
            print('x341: {}'.format(x341.shape))
        elif isinstance(x341, tuple):
            tuple_shapes = '('
            for item in x341:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x341: {}'.format(tuple_shapes))
        else:
            print('x341: {}'.format(x341))
        x342=self.batchnorm2d66(x341)
        if x342 is None:
            print('x342: {}'.format(x342))
        elif isinstance(x342, torch.Tensor):
            print('x342: {}'.format(x342.shape))
        elif isinstance(x342, tuple):
            tuple_shapes = '('
            for item in x342:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x342: {}'.format(tuple_shapes))
        else:
            print('x342: {}'.format(x342))
        x343=self.silu66(x342)
        if x343 is None:
            print('x343: {}'.format(x343))
        elif isinstance(x343, torch.Tensor):
            print('x343: {}'.format(x343.shape))
        elif isinstance(x343, tuple):
            tuple_shapes = '('
            for item in x343:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x343: {}'.format(tuple_shapes))
        else:
            print('x343: {}'.format(x343))
        x344=self.adaptiveavgpool2d22(x343)
        if x344 is None:
            print('x344: {}'.format(x344))
        elif isinstance(x344, torch.Tensor):
            print('x344: {}'.format(x344.shape))
        elif isinstance(x344, tuple):
            tuple_shapes = '('
            for item in x344:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x344: {}'.format(tuple_shapes))
        else:
            print('x344: {}'.format(x344))
        x345=self.conv2d111(x344)
        if x345 is None:
            print('x345: {}'.format(x345))
        elif isinstance(x345, torch.Tensor):
            print('x345: {}'.format(x345.shape))
        elif isinstance(x345, tuple):
            tuple_shapes = '('
            for item in x345:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x345: {}'.format(tuple_shapes))
        else:
            print('x345: {}'.format(x345))
        x346=self.silu67(x345)
        if x346 is None:
            print('x346: {}'.format(x346))
        elif isinstance(x346, torch.Tensor):
            print('x346: {}'.format(x346.shape))
        elif isinstance(x346, tuple):
            tuple_shapes = '('
            for item in x346:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x346: {}'.format(tuple_shapes))
        else:
            print('x346: {}'.format(x346))
        x347=self.conv2d112(x346)
        if x347 is None:
            print('x347: {}'.format(x347))
        elif isinstance(x347, torch.Tensor):
            print('x347: {}'.format(x347.shape))
        elif isinstance(x347, tuple):
            tuple_shapes = '('
            for item in x347:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x347: {}'.format(tuple_shapes))
        else:
            print('x347: {}'.format(x347))
        x348=self.sigmoid22(x347)
        if x348 is None:
            print('x348: {}'.format(x348))
        elif isinstance(x348, torch.Tensor):
            print('x348: {}'.format(x348.shape))
        elif isinstance(x348, tuple):
            tuple_shapes = '('
            for item in x348:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x348: {}'.format(tuple_shapes))
        else:
            print('x348: {}'.format(x348))
        x349=operator.mul(x348, x343)
        if x349 is None:
            print('x349: {}'.format(x349))
        elif isinstance(x349, torch.Tensor):
            print('x349: {}'.format(x349.shape))
        elif isinstance(x349, tuple):
            tuple_shapes = '('
            for item in x349:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x349: {}'.format(tuple_shapes))
        else:
            print('x349: {}'.format(x349))
        x350=self.conv2d113(x349)
        if x350 is None:
            print('x350: {}'.format(x350))
        elif isinstance(x350, torch.Tensor):
            print('x350: {}'.format(x350.shape))
        elif isinstance(x350, tuple):
            tuple_shapes = '('
            for item in x350:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x350: {}'.format(tuple_shapes))
        else:
            print('x350: {}'.format(x350))
        x351=self.batchnorm2d67(x350)
        if x351 is None:
            print('x351: {}'.format(x351))
        elif isinstance(x351, torch.Tensor):
            print('x351: {}'.format(x351.shape))
        elif isinstance(x351, tuple):
            tuple_shapes = '('
            for item in x351:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x351: {}'.format(tuple_shapes))
        else:
            print('x351: {}'.format(x351))
        x352=stochastic_depth(x351, 0.16923076923076924, 'row', False)
        if x352 is None:
            print('x352: {}'.format(x352))
        elif isinstance(x352, torch.Tensor):
            print('x352: {}'.format(x352.shape))
        elif isinstance(x352, tuple):
            tuple_shapes = '('
            for item in x352:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x352: {}'.format(tuple_shapes))
        else:
            print('x352: {}'.format(x352))
        x353=operator.add(x352, x337)
        if x353 is None:
            print('x353: {}'.format(x353))
        elif isinstance(x353, torch.Tensor):
            print('x353: {}'.format(x353.shape))
        elif isinstance(x353, tuple):
            tuple_shapes = '('
            for item in x353:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x353: {}'.format(tuple_shapes))
        else:
            print('x353: {}'.format(x353))
        x354=self.conv2d114(x353)
        if x354 is None:
            print('x354: {}'.format(x354))
        elif isinstance(x354, torch.Tensor):
            print('x354: {}'.format(x354.shape))
        elif isinstance(x354, tuple):
            tuple_shapes = '('
            for item in x354:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x354: {}'.format(tuple_shapes))
        else:
            print('x354: {}'.format(x354))
        x355=self.batchnorm2d68(x354)
        if x355 is None:
            print('x355: {}'.format(x355))
        elif isinstance(x355, torch.Tensor):
            print('x355: {}'.format(x355.shape))
        elif isinstance(x355, tuple):
            tuple_shapes = '('
            for item in x355:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x355: {}'.format(tuple_shapes))
        else:
            print('x355: {}'.format(x355))
        x356=self.silu68(x355)
        if x356 is None:
            print('x356: {}'.format(x356))
        elif isinstance(x356, torch.Tensor):
            print('x356: {}'.format(x356.shape))
        elif isinstance(x356, tuple):
            tuple_shapes = '('
            for item in x356:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x356: {}'.format(tuple_shapes))
        else:
            print('x356: {}'.format(x356))
        x357=self.conv2d115(x356)
        if x357 is None:
            print('x357: {}'.format(x357))
        elif isinstance(x357, torch.Tensor):
            print('x357: {}'.format(x357.shape))
        elif isinstance(x357, tuple):
            tuple_shapes = '('
            for item in x357:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x357: {}'.format(tuple_shapes))
        else:
            print('x357: {}'.format(x357))
        x358=self.batchnorm2d69(x357)
        if x358 is None:
            print('x358: {}'.format(x358))
        elif isinstance(x358, torch.Tensor):
            print('x358: {}'.format(x358.shape))
        elif isinstance(x358, tuple):
            tuple_shapes = '('
            for item in x358:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x358: {}'.format(tuple_shapes))
        else:
            print('x358: {}'.format(x358))
        x359=self.silu69(x358)
        if x359 is None:
            print('x359: {}'.format(x359))
        elif isinstance(x359, torch.Tensor):
            print('x359: {}'.format(x359.shape))
        elif isinstance(x359, tuple):
            tuple_shapes = '('
            for item in x359:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x359: {}'.format(tuple_shapes))
        else:
            print('x359: {}'.format(x359))
        x360=self.adaptiveavgpool2d23(x359)
        if x360 is None:
            print('x360: {}'.format(x360))
        elif isinstance(x360, torch.Tensor):
            print('x360: {}'.format(x360.shape))
        elif isinstance(x360, tuple):
            tuple_shapes = '('
            for item in x360:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x360: {}'.format(tuple_shapes))
        else:
            print('x360: {}'.format(x360))
        x361=self.conv2d116(x360)
        if x361 is None:
            print('x361: {}'.format(x361))
        elif isinstance(x361, torch.Tensor):
            print('x361: {}'.format(x361.shape))
        elif isinstance(x361, tuple):
            tuple_shapes = '('
            for item in x361:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x361: {}'.format(tuple_shapes))
        else:
            print('x361: {}'.format(x361))
        x362=self.silu70(x361)
        if x362 is None:
            print('x362: {}'.format(x362))
        elif isinstance(x362, torch.Tensor):
            print('x362: {}'.format(x362.shape))
        elif isinstance(x362, tuple):
            tuple_shapes = '('
            for item in x362:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x362: {}'.format(tuple_shapes))
        else:
            print('x362: {}'.format(x362))
        x363=self.conv2d117(x362)
        if x363 is None:
            print('x363: {}'.format(x363))
        elif isinstance(x363, torch.Tensor):
            print('x363: {}'.format(x363.shape))
        elif isinstance(x363, tuple):
            tuple_shapes = '('
            for item in x363:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x363: {}'.format(tuple_shapes))
        else:
            print('x363: {}'.format(x363))
        x364=self.sigmoid23(x363)
        if x364 is None:
            print('x364: {}'.format(x364))
        elif isinstance(x364, torch.Tensor):
            print('x364: {}'.format(x364.shape))
        elif isinstance(x364, tuple):
            tuple_shapes = '('
            for item in x364:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x364: {}'.format(tuple_shapes))
        else:
            print('x364: {}'.format(x364))
        x365=operator.mul(x364, x359)
        if x365 is None:
            print('x365: {}'.format(x365))
        elif isinstance(x365, torch.Tensor):
            print('x365: {}'.format(x365.shape))
        elif isinstance(x365, tuple):
            tuple_shapes = '('
            for item in x365:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x365: {}'.format(tuple_shapes))
        else:
            print('x365: {}'.format(x365))
        x366=self.conv2d118(x365)
        if x366 is None:
            print('x366: {}'.format(x366))
        elif isinstance(x366, torch.Tensor):
            print('x366: {}'.format(x366.shape))
        elif isinstance(x366, tuple):
            tuple_shapes = '('
            for item in x366:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x366: {}'.format(tuple_shapes))
        else:
            print('x366: {}'.format(x366))
        x367=self.batchnorm2d70(x366)
        if x367 is None:
            print('x367: {}'.format(x367))
        elif isinstance(x367, torch.Tensor):
            print('x367: {}'.format(x367.shape))
        elif isinstance(x367, tuple):
            tuple_shapes = '('
            for item in x367:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x367: {}'.format(tuple_shapes))
        else:
            print('x367: {}'.format(x367))
        x368=stochastic_depth(x367, 0.17692307692307693, 'row', False)
        if x368 is None:
            print('x368: {}'.format(x368))
        elif isinstance(x368, torch.Tensor):
            print('x368: {}'.format(x368.shape))
        elif isinstance(x368, tuple):
            tuple_shapes = '('
            for item in x368:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x368: {}'.format(tuple_shapes))
        else:
            print('x368: {}'.format(x368))
        x369=operator.add(x368, x353)
        if x369 is None:
            print('x369: {}'.format(x369))
        elif isinstance(x369, torch.Tensor):
            print('x369: {}'.format(x369.shape))
        elif isinstance(x369, tuple):
            tuple_shapes = '('
            for item in x369:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x369: {}'.format(tuple_shapes))
        else:
            print('x369: {}'.format(x369))
        x370=self.conv2d119(x369)
        if x370 is None:
            print('x370: {}'.format(x370))
        elif isinstance(x370, torch.Tensor):
            print('x370: {}'.format(x370.shape))
        elif isinstance(x370, tuple):
            tuple_shapes = '('
            for item in x370:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x370: {}'.format(tuple_shapes))
        else:
            print('x370: {}'.format(x370))
        x371=self.batchnorm2d71(x370)
        if x371 is None:
            print('x371: {}'.format(x371))
        elif isinstance(x371, torch.Tensor):
            print('x371: {}'.format(x371.shape))
        elif isinstance(x371, tuple):
            tuple_shapes = '('
            for item in x371:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x371: {}'.format(tuple_shapes))
        else:
            print('x371: {}'.format(x371))
        x372=self.silu71(x371)
        if x372 is None:
            print('x372: {}'.format(x372))
        elif isinstance(x372, torch.Tensor):
            print('x372: {}'.format(x372.shape))
        elif isinstance(x372, tuple):
            tuple_shapes = '('
            for item in x372:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x372: {}'.format(tuple_shapes))
        else:
            print('x372: {}'.format(x372))
        x373=self.conv2d120(x372)
        if x373 is None:
            print('x373: {}'.format(x373))
        elif isinstance(x373, torch.Tensor):
            print('x373: {}'.format(x373.shape))
        elif isinstance(x373, tuple):
            tuple_shapes = '('
            for item in x373:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x373: {}'.format(tuple_shapes))
        else:
            print('x373: {}'.format(x373))
        x374=self.batchnorm2d72(x373)
        if x374 is None:
            print('x374: {}'.format(x374))
        elif isinstance(x374, torch.Tensor):
            print('x374: {}'.format(x374.shape))
        elif isinstance(x374, tuple):
            tuple_shapes = '('
            for item in x374:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x374: {}'.format(tuple_shapes))
        else:
            print('x374: {}'.format(x374))
        x375=self.silu72(x374)
        if x375 is None:
            print('x375: {}'.format(x375))
        elif isinstance(x375, torch.Tensor):
            print('x375: {}'.format(x375.shape))
        elif isinstance(x375, tuple):
            tuple_shapes = '('
            for item in x375:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x375: {}'.format(tuple_shapes))
        else:
            print('x375: {}'.format(x375))
        x376=self.adaptiveavgpool2d24(x375)
        if x376 is None:
            print('x376: {}'.format(x376))
        elif isinstance(x376, torch.Tensor):
            print('x376: {}'.format(x376.shape))
        elif isinstance(x376, tuple):
            tuple_shapes = '('
            for item in x376:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x376: {}'.format(tuple_shapes))
        else:
            print('x376: {}'.format(x376))
        x377=self.conv2d121(x376)
        if x377 is None:
            print('x377: {}'.format(x377))
        elif isinstance(x377, torch.Tensor):
            print('x377: {}'.format(x377.shape))
        elif isinstance(x377, tuple):
            tuple_shapes = '('
            for item in x377:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x377: {}'.format(tuple_shapes))
        else:
            print('x377: {}'.format(x377))
        x378=self.silu73(x377)
        if x378 is None:
            print('x378: {}'.format(x378))
        elif isinstance(x378, torch.Tensor):
            print('x378: {}'.format(x378.shape))
        elif isinstance(x378, tuple):
            tuple_shapes = '('
            for item in x378:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x378: {}'.format(tuple_shapes))
        else:
            print('x378: {}'.format(x378))
        x379=self.conv2d122(x378)
        if x379 is None:
            print('x379: {}'.format(x379))
        elif isinstance(x379, torch.Tensor):
            print('x379: {}'.format(x379.shape))
        elif isinstance(x379, tuple):
            tuple_shapes = '('
            for item in x379:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x379: {}'.format(tuple_shapes))
        else:
            print('x379: {}'.format(x379))
        x380=self.sigmoid24(x379)
        if x380 is None:
            print('x380: {}'.format(x380))
        elif isinstance(x380, torch.Tensor):
            print('x380: {}'.format(x380.shape))
        elif isinstance(x380, tuple):
            tuple_shapes = '('
            for item in x380:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x380: {}'.format(tuple_shapes))
        else:
            print('x380: {}'.format(x380))
        x381=operator.mul(x380, x375)
        if x381 is None:
            print('x381: {}'.format(x381))
        elif isinstance(x381, torch.Tensor):
            print('x381: {}'.format(x381.shape))
        elif isinstance(x381, tuple):
            tuple_shapes = '('
            for item in x381:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x381: {}'.format(tuple_shapes))
        else:
            print('x381: {}'.format(x381))
        x382=self.conv2d123(x381)
        if x382 is None:
            print('x382: {}'.format(x382))
        elif isinstance(x382, torch.Tensor):
            print('x382: {}'.format(x382.shape))
        elif isinstance(x382, tuple):
            tuple_shapes = '('
            for item in x382:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x382: {}'.format(tuple_shapes))
        else:
            print('x382: {}'.format(x382))
        x383=self.batchnorm2d73(x382)
        if x383 is None:
            print('x383: {}'.format(x383))
        elif isinstance(x383, torch.Tensor):
            print('x383: {}'.format(x383.shape))
        elif isinstance(x383, tuple):
            tuple_shapes = '('
            for item in x383:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x383: {}'.format(tuple_shapes))
        else:
            print('x383: {}'.format(x383))
        x384=self.conv2d124(x383)
        if x384 is None:
            print('x384: {}'.format(x384))
        elif isinstance(x384, torch.Tensor):
            print('x384: {}'.format(x384.shape))
        elif isinstance(x384, tuple):
            tuple_shapes = '('
            for item in x384:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x384: {}'.format(tuple_shapes))
        else:
            print('x384: {}'.format(x384))
        x385=self.batchnorm2d74(x384)
        if x385 is None:
            print('x385: {}'.format(x385))
        elif isinstance(x385, torch.Tensor):
            print('x385: {}'.format(x385.shape))
        elif isinstance(x385, tuple):
            tuple_shapes = '('
            for item in x385:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x385: {}'.format(tuple_shapes))
        else:
            print('x385: {}'.format(x385))
        x386=self.silu74(x385)
        if x386 is None:
            print('x386: {}'.format(x386))
        elif isinstance(x386, torch.Tensor):
            print('x386: {}'.format(x386.shape))
        elif isinstance(x386, tuple):
            tuple_shapes = '('
            for item in x386:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x386: {}'.format(tuple_shapes))
        else:
            print('x386: {}'.format(x386))
        x387=self.conv2d125(x386)
        if x387 is None:
            print('x387: {}'.format(x387))
        elif isinstance(x387, torch.Tensor):
            print('x387: {}'.format(x387.shape))
        elif isinstance(x387, tuple):
            tuple_shapes = '('
            for item in x387:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x387: {}'.format(tuple_shapes))
        else:
            print('x387: {}'.format(x387))
        x388=self.batchnorm2d75(x387)
        if x388 is None:
            print('x388: {}'.format(x388))
        elif isinstance(x388, torch.Tensor):
            print('x388: {}'.format(x388.shape))
        elif isinstance(x388, tuple):
            tuple_shapes = '('
            for item in x388:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x388: {}'.format(tuple_shapes))
        else:
            print('x388: {}'.format(x388))
        x389=self.silu75(x388)
        if x389 is None:
            print('x389: {}'.format(x389))
        elif isinstance(x389, torch.Tensor):
            print('x389: {}'.format(x389.shape))
        elif isinstance(x389, tuple):
            tuple_shapes = '('
            for item in x389:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x389: {}'.format(tuple_shapes))
        else:
            print('x389: {}'.format(x389))
        x390=self.adaptiveavgpool2d25(x389)
        if x390 is None:
            print('x390: {}'.format(x390))
        elif isinstance(x390, torch.Tensor):
            print('x390: {}'.format(x390.shape))
        elif isinstance(x390, tuple):
            tuple_shapes = '('
            for item in x390:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x390: {}'.format(tuple_shapes))
        else:
            print('x390: {}'.format(x390))
        x391=self.conv2d126(x390)
        if x391 is None:
            print('x391: {}'.format(x391))
        elif isinstance(x391, torch.Tensor):
            print('x391: {}'.format(x391.shape))
        elif isinstance(x391, tuple):
            tuple_shapes = '('
            for item in x391:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x391: {}'.format(tuple_shapes))
        else:
            print('x391: {}'.format(x391))
        x392=self.silu76(x391)
        if x392 is None:
            print('x392: {}'.format(x392))
        elif isinstance(x392, torch.Tensor):
            print('x392: {}'.format(x392.shape))
        elif isinstance(x392, tuple):
            tuple_shapes = '('
            for item in x392:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x392: {}'.format(tuple_shapes))
        else:
            print('x392: {}'.format(x392))
        x393=self.conv2d127(x392)
        if x393 is None:
            print('x393: {}'.format(x393))
        elif isinstance(x393, torch.Tensor):
            print('x393: {}'.format(x393.shape))
        elif isinstance(x393, tuple):
            tuple_shapes = '('
            for item in x393:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x393: {}'.format(tuple_shapes))
        else:
            print('x393: {}'.format(x393))
        x394=self.sigmoid25(x393)
        if x394 is None:
            print('x394: {}'.format(x394))
        elif isinstance(x394, torch.Tensor):
            print('x394: {}'.format(x394.shape))
        elif isinstance(x394, tuple):
            tuple_shapes = '('
            for item in x394:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x394: {}'.format(tuple_shapes))
        else:
            print('x394: {}'.format(x394))
        x395=operator.mul(x394, x389)
        if x395 is None:
            print('x395: {}'.format(x395))
        elif isinstance(x395, torch.Tensor):
            print('x395: {}'.format(x395.shape))
        elif isinstance(x395, tuple):
            tuple_shapes = '('
            for item in x395:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x395: {}'.format(tuple_shapes))
        else:
            print('x395: {}'.format(x395))
        x396=self.conv2d128(x395)
        if x396 is None:
            print('x396: {}'.format(x396))
        elif isinstance(x396, torch.Tensor):
            print('x396: {}'.format(x396.shape))
        elif isinstance(x396, tuple):
            tuple_shapes = '('
            for item in x396:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x396: {}'.format(tuple_shapes))
        else:
            print('x396: {}'.format(x396))
        x397=self.batchnorm2d76(x396)
        if x397 is None:
            print('x397: {}'.format(x397))
        elif isinstance(x397, torch.Tensor):
            print('x397: {}'.format(x397.shape))
        elif isinstance(x397, tuple):
            tuple_shapes = '('
            for item in x397:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x397: {}'.format(tuple_shapes))
        else:
            print('x397: {}'.format(x397))
        x398=stochastic_depth(x397, 0.19230769230769232, 'row', False)
        if x398 is None:
            print('x398: {}'.format(x398))
        elif isinstance(x398, torch.Tensor):
            print('x398: {}'.format(x398.shape))
        elif isinstance(x398, tuple):
            tuple_shapes = '('
            for item in x398:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x398: {}'.format(tuple_shapes))
        else:
            print('x398: {}'.format(x398))
        x399=operator.add(x398, x383)
        if x399 is None:
            print('x399: {}'.format(x399))
        elif isinstance(x399, torch.Tensor):
            print('x399: {}'.format(x399.shape))
        elif isinstance(x399, tuple):
            tuple_shapes = '('
            for item in x399:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x399: {}'.format(tuple_shapes))
        else:
            print('x399: {}'.format(x399))
        x400=self.conv2d129(x399)
        if x400 is None:
            print('x400: {}'.format(x400))
        elif isinstance(x400, torch.Tensor):
            print('x400: {}'.format(x400.shape))
        elif isinstance(x400, tuple):
            tuple_shapes = '('
            for item in x400:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x400: {}'.format(tuple_shapes))
        else:
            print('x400: {}'.format(x400))
        x401=self.batchnorm2d77(x400)
        if x401 is None:
            print('x401: {}'.format(x401))
        elif isinstance(x401, torch.Tensor):
            print('x401: {}'.format(x401.shape))
        elif isinstance(x401, tuple):
            tuple_shapes = '('
            for item in x401:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x401: {}'.format(tuple_shapes))
        else:
            print('x401: {}'.format(x401))
        x402=self.silu77(x401)
        if x402 is None:
            print('x402: {}'.format(x402))
        elif isinstance(x402, torch.Tensor):
            print('x402: {}'.format(x402.shape))
        elif isinstance(x402, tuple):
            tuple_shapes = '('
            for item in x402:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x402: {}'.format(tuple_shapes))
        else:
            print('x402: {}'.format(x402))
        x403=self.adaptiveavgpool2d26(x402)
        if x403 is None:
            print('x403: {}'.format(x403))
        elif isinstance(x403, torch.Tensor):
            print('x403: {}'.format(x403.shape))
        elif isinstance(x403, tuple):
            tuple_shapes = '('
            for item in x403:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x403: {}'.format(tuple_shapes))
        else:
            print('x403: {}'.format(x403))
        x404=torch.flatten(x403, 1)
        if x404 is None:
            print('x404: {}'.format(x404))
        elif isinstance(x404, torch.Tensor):
            print('x404: {}'.format(x404.shape))
        elif isinstance(x404, tuple):
            tuple_shapes = '('
            for item in x404:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x404: {}'.format(tuple_shapes))
        else:
            print('x404: {}'.format(x404))
        x405=self.dropout0(x404)
        if x405 is None:
            print('x405: {}'.format(x405))
        elif isinstance(x405, torch.Tensor):
            print('x405: {}'.format(x405.shape))
        elif isinstance(x405, tuple):
            tuple_shapes = '('
            for item in x405:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x405: {}'.format(tuple_shapes))
        else:
            print('x405: {}'.format(x405))
        x406=self.linear0(x405)
        if x406 is None:
            print('x406: {}'.format(x406))
        elif isinstance(x406, torch.Tensor):
            print('x406: {}'.format(x406.shape))
        elif isinstance(x406, tuple):
            tuple_shapes = '('
            for item in x406:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x406: {}'.format(tuple_shapes))
        else:
            print('x406: {}'.format(x406))

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
