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
        self.conv2d0 = Conv2d(3, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu0 = SiLU(inplace=True)
        self.conv2d1 = Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
        self.batchnorm2d1 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu1 = SiLU(inplace=True)
        self.conv2d2 = Conv2d(40, 10, kernel_size=(1, 1), stride=(1, 1))
        self.silu2 = SiLU(inplace=True)
        self.conv2d3 = Conv2d(10, 40, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d4 = Conv2d(40, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.identity0 = Identity()
        self.conv2d5 = Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d3 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu3 = SiLU(inplace=True)
        self.conv2d6 = Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu4 = SiLU(inplace=True)
        self.conv2d7 = Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d8 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.identity1 = Identity()
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu5 = SiLU(inplace=True)
        self.conv2d10 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu6 = SiLU(inplace=True)
        self.conv2d11 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
        self.silu7 = SiLU(inplace=True)
        self.conv2d12 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d13 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu8 = SiLU(inplace=True)
        self.conv2d15 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d9 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu9 = SiLU(inplace=True)
        self.conv2d16 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu10 = SiLU(inplace=True)
        self.conv2d17 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d18 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu11 = SiLU(inplace=True)
        self.conv2d20 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d12 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu12 = SiLU(inplace=True)
        self.conv2d21 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu13 = SiLU(inplace=True)
        self.conv2d22 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d23 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu14 = SiLU(inplace=True)
        self.conv2d25 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu15 = SiLU(inplace=True)
        self.conv2d26 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu16 = SiLU(inplace=True)
        self.conv2d27 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d28 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu17 = SiLU(inplace=True)
        self.conv2d30 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d18 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu18 = SiLU(inplace=True)
        self.conv2d31 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu19 = SiLU(inplace=True)
        self.conv2d32 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d33 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu20 = SiLU(inplace=True)
        self.conv2d35 = Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d21 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu21 = SiLU(inplace=True)
        self.conv2d36 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu22 = SiLU(inplace=True)
        self.conv2d37 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d38 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu23 = SiLU(inplace=True)
        self.conv2d40 = Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu24 = SiLU(inplace=True)
        self.conv2d41 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu25 = SiLU(inplace=True)
        self.conv2d42 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d43 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu26 = SiLU(inplace=True)
        self.conv2d45 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d27 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu27 = SiLU(inplace=True)
        self.conv2d46 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu28 = SiLU(inplace=True)
        self.conv2d47 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d48 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu29 = SiLU(inplace=True)
        self.conv2d50 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d30 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu30 = SiLU(inplace=True)
        self.conv2d51 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu31 = SiLU(inplace=True)
        self.conv2d52 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d53 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu32 = SiLU(inplace=True)
        self.conv2d55 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d33 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu33 = SiLU(inplace=True)
        self.conv2d56 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu34 = SiLU(inplace=True)
        self.conv2d57 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d58 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d59 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu35 = SiLU(inplace=True)
        self.conv2d60 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d36 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu36 = SiLU(inplace=True)
        self.conv2d61 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu37 = SiLU(inplace=True)
        self.conv2d62 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d63 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu38 = SiLU(inplace=True)
        self.conv2d65 = Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
        self.batchnorm2d39 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu39 = SiLU(inplace=True)
        self.conv2d66 = Conv2d(576, 24, kernel_size=(1, 1), stride=(1, 1))
        self.silu40 = SiLU(inplace=True)
        self.conv2d67 = Conv2d(24, 576, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d68 = Conv2d(576, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d69 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu41 = SiLU(inplace=True)
        self.conv2d70 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d42 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu42 = SiLU(inplace=True)
        self.conv2d71 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu43 = SiLU(inplace=True)
        self.conv2d72 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d73 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d74 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu44 = SiLU(inplace=True)
        self.conv2d75 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d45 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu45 = SiLU(inplace=True)
        self.conv2d76 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu46 = SiLU(inplace=True)
        self.conv2d77 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d78 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d79 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu47 = SiLU(inplace=True)
        self.conv2d80 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d48 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu48 = SiLU(inplace=True)
        self.conv2d81 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu49 = SiLU(inplace=True)
        self.conv2d82 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d83 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu50 = SiLU(inplace=True)
        self.conv2d85 = Conv2d(816, 816, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d51 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu51 = SiLU(inplace=True)
        self.conv2d86 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu52 = SiLU(inplace=True)
        self.conv2d87 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d88 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d89 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu53 = SiLU(inplace=True)
        self.conv2d90 = Conv2d(816, 816, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=816, bias=False)
        self.batchnorm2d54 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu54 = SiLU(inplace=True)
        self.conv2d91 = Conv2d(816, 34, kernel_size=(1, 1), stride=(1, 1))
        self.silu55 = SiLU(inplace=True)
        self.conv2d92 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d93 = Conv2d(816, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d94 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu56 = SiLU(inplace=True)
        self.conv2d95 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d57 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu57 = SiLU(inplace=True)
        self.conv2d96 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu58 = SiLU(inplace=True)
        self.conv2d97 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d98 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d99 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu59 = SiLU(inplace=True)
        self.conv2d100 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d60 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu60 = SiLU(inplace=True)
        self.conv2d101 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu61 = SiLU(inplace=True)
        self.conv2d102 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d103 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d104 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu62 = SiLU(inplace=True)
        self.conv2d105 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d63 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu63 = SiLU(inplace=True)
        self.conv2d106 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu64 = SiLU(inplace=True)
        self.conv2d107 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d108 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d109 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu65 = SiLU(inplace=True)
        self.conv2d110 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d66 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu66 = SiLU(inplace=True)
        self.conv2d111 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu67 = SiLU(inplace=True)
        self.conv2d112 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d113 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu68 = SiLU(inplace=True)
        self.conv2d115 = Conv2d(1392, 1392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1392, bias=False)
        self.batchnorm2d69 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu69 = SiLU(inplace=True)
        self.conv2d116 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu70 = SiLU(inplace=True)
        self.conv2d117 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d118 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d119 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu71 = SiLU(inplace=True)
        self.conv2d120 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1392, bias=False)
        self.batchnorm2d72 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu72 = SiLU(inplace=True)
        self.conv2d121 = Conv2d(1392, 58, kernel_size=(1, 1), stride=(1, 1))
        self.silu73 = SiLU(inplace=True)
        self.conv2d122 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d124 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu74 = SiLU(inplace=True)
        self.conv2d125 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
        self.batchnorm2d75 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu75 = SiLU(inplace=True)
        self.conv2d126 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu76 = SiLU(inplace=True)
        self.conv2d127 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d128 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d129 = Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.silu77 = SiLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear0 = Linear(in_features=1536, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.silu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.silu1(x5)
        x7=x6.mean((2, 3),keepdim=True)
        x8=self.conv2d2(x7)
        x9=self.silu2(x8)
        x10=self.conv2d3(x9)
        x11=x10.sigmoid()
        x12=operator.mul(x6, x11)
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d2(x13)
        x15=self.identity0(x14)
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d3(x16)
        x18=self.silu3(x17)
        x19=x18.mean((2, 3),keepdim=True)
        x20=self.conv2d6(x19)
        x21=self.silu4(x20)
        x22=self.conv2d7(x21)
        x23=x22.sigmoid()
        x24=operator.mul(x18, x23)
        x25=self.conv2d8(x24)
        x26=self.batchnorm2d4(x25)
        x27=self.identity1(x26)
        x28=operator.add(x27, x15)
        x29=self.conv2d9(x28)
        x30=self.batchnorm2d5(x29)
        x31=self.silu5(x30)
        x32=self.conv2d10(x31)
        x33=self.batchnorm2d6(x32)
        x34=self.silu6(x33)
        x35=x34.mean((2, 3),keepdim=True)
        x36=self.conv2d11(x35)
        x37=self.silu7(x36)
        x38=self.conv2d12(x37)
        x39=x38.sigmoid()
        x40=operator.mul(x34, x39)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d7(x41)
        x43=self.conv2d14(x42)
        x44=self.batchnorm2d8(x43)
        x45=self.silu8(x44)
        x46=self.conv2d15(x45)
        x47=self.batchnorm2d9(x46)
        x48=self.silu9(x47)
        x49=x48.mean((2, 3),keepdim=True)
        x50=self.conv2d16(x49)
        x51=self.silu10(x50)
        x52=self.conv2d17(x51)
        x53=x52.sigmoid()
        x54=operator.mul(x48, x53)
        x55=self.conv2d18(x54)
        x56=self.batchnorm2d10(x55)
        x57=operator.add(x56, x42)
        x58=self.conv2d19(x57)
        x59=self.batchnorm2d11(x58)
        x60=self.silu11(x59)
        x61=self.conv2d20(x60)
        x62=self.batchnorm2d12(x61)
        x63=self.silu12(x62)
        x64=x63.mean((2, 3),keepdim=True)
        x65=self.conv2d21(x64)
        x66=self.silu13(x65)
        x67=self.conv2d22(x66)
        x68=x67.sigmoid()
        x69=operator.mul(x63, x68)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d13(x70)
        x72=operator.add(x71, x57)
        x73=self.conv2d24(x72)
        x74=self.batchnorm2d14(x73)
        x75=self.silu14(x74)
        x76=self.conv2d25(x75)
        x77=self.batchnorm2d15(x76)
        x78=self.silu15(x77)
        x79=x78.mean((2, 3),keepdim=True)
        x80=self.conv2d26(x79)
        x81=self.silu16(x80)
        x82=self.conv2d27(x81)
        x83=x82.sigmoid()
        x84=operator.mul(x78, x83)
        x85=self.conv2d28(x84)
        x86=self.batchnorm2d16(x85)
        x87=self.conv2d29(x86)
        x88=self.batchnorm2d17(x87)
        x89=self.silu17(x88)
        x90=self.conv2d30(x89)
        x91=self.batchnorm2d18(x90)
        x92=self.silu18(x91)
        x93=x92.mean((2, 3),keepdim=True)
        x94=self.conv2d31(x93)
        x95=self.silu19(x94)
        x96=self.conv2d32(x95)
        x97=x96.sigmoid()
        x98=operator.mul(x92, x97)
        x99=self.conv2d33(x98)
        x100=self.batchnorm2d19(x99)
        x101=operator.add(x100, x86)
        x102=self.conv2d34(x101)
        x103=self.batchnorm2d20(x102)
        x104=self.silu20(x103)
        x105=self.conv2d35(x104)
        x106=self.batchnorm2d21(x105)
        x107=self.silu21(x106)
        x108=x107.mean((2, 3),keepdim=True)
        x109=self.conv2d36(x108)
        x110=self.silu22(x109)
        x111=self.conv2d37(x110)
        x112=x111.sigmoid()
        x113=operator.mul(x107, x112)
        x114=self.conv2d38(x113)
        x115=self.batchnorm2d22(x114)
        x116=operator.add(x115, x101)
        x117=self.conv2d39(x116)
        x118=self.batchnorm2d23(x117)
        x119=self.silu23(x118)
        x120=self.conv2d40(x119)
        x121=self.batchnorm2d24(x120)
        x122=self.silu24(x121)
        x123=x122.mean((2, 3),keepdim=True)
        x124=self.conv2d41(x123)
        x125=self.silu25(x124)
        x126=self.conv2d42(x125)
        x127=x126.sigmoid()
        x128=operator.mul(x122, x127)
        x129=self.conv2d43(x128)
        x130=self.batchnorm2d25(x129)
        x131=self.conv2d44(x130)
        x132=self.batchnorm2d26(x131)
        x133=self.silu26(x132)
        x134=self.conv2d45(x133)
        x135=self.batchnorm2d27(x134)
        x136=self.silu27(x135)
        x137=x136.mean((2, 3),keepdim=True)
        x138=self.conv2d46(x137)
        x139=self.silu28(x138)
        x140=self.conv2d47(x139)
        x141=x140.sigmoid()
        x142=operator.mul(x136, x141)
        x143=self.conv2d48(x142)
        x144=self.batchnorm2d28(x143)
        x145=operator.add(x144, x130)
        x146=self.conv2d49(x145)
        x147=self.batchnorm2d29(x146)
        x148=self.silu29(x147)
        x149=self.conv2d50(x148)
        x150=self.batchnorm2d30(x149)
        x151=self.silu30(x150)
        x152=x151.mean((2, 3),keepdim=True)
        x153=self.conv2d51(x152)
        x154=self.silu31(x153)
        x155=self.conv2d52(x154)
        x156=x155.sigmoid()
        x157=operator.mul(x151, x156)
        x158=self.conv2d53(x157)
        x159=self.batchnorm2d31(x158)
        x160=operator.add(x159, x145)
        x161=self.conv2d54(x160)
        x162=self.batchnorm2d32(x161)
        x163=self.silu32(x162)
        x164=self.conv2d55(x163)
        x165=self.batchnorm2d33(x164)
        x166=self.silu33(x165)
        x167=x166.mean((2, 3),keepdim=True)
        x168=self.conv2d56(x167)
        x169=self.silu34(x168)
        x170=self.conv2d57(x169)
        x171=x170.sigmoid()
        x172=operator.mul(x166, x171)
        x173=self.conv2d58(x172)
        x174=self.batchnorm2d34(x173)
        x175=operator.add(x174, x160)
        x176=self.conv2d59(x175)
        x177=self.batchnorm2d35(x176)
        x178=self.silu35(x177)
        x179=self.conv2d60(x178)
        x180=self.batchnorm2d36(x179)
        x181=self.silu36(x180)
        x182=x181.mean((2, 3),keepdim=True)
        x183=self.conv2d61(x182)
        x184=self.silu37(x183)
        x185=self.conv2d62(x184)
        x186=x185.sigmoid()
        x187=operator.mul(x181, x186)
        x188=self.conv2d63(x187)
        x189=self.batchnorm2d37(x188)
        x190=operator.add(x189, x175)
        x191=self.conv2d64(x190)
        x192=self.batchnorm2d38(x191)
        x193=self.silu38(x192)
        x194=self.conv2d65(x193)
        x195=self.batchnorm2d39(x194)
        x196=self.silu39(x195)
        x197=x196.mean((2, 3),keepdim=True)
        x198=self.conv2d66(x197)
        x199=self.silu40(x198)
        x200=self.conv2d67(x199)
        x201=x200.sigmoid()
        x202=operator.mul(x196, x201)
        x203=self.conv2d68(x202)
        x204=self.batchnorm2d40(x203)
        x205=self.conv2d69(x204)
        x206=self.batchnorm2d41(x205)
        x207=self.silu41(x206)
        x208=self.conv2d70(x207)
        x209=self.batchnorm2d42(x208)
        x210=self.silu42(x209)
        x211=x210.mean((2, 3),keepdim=True)
        x212=self.conv2d71(x211)
        x213=self.silu43(x212)
        x214=self.conv2d72(x213)
        x215=x214.sigmoid()
        x216=operator.mul(x210, x215)
        x217=self.conv2d73(x216)
        x218=self.batchnorm2d43(x217)
        x219=operator.add(x218, x204)
        x220=self.conv2d74(x219)
        x221=self.batchnorm2d44(x220)
        x222=self.silu44(x221)
        x223=self.conv2d75(x222)
        x224=self.batchnorm2d45(x223)
        x225=self.silu45(x224)
        x226=x225.mean((2, 3),keepdim=True)
        x227=self.conv2d76(x226)
        x228=self.silu46(x227)
        x229=self.conv2d77(x228)
        x230=x229.sigmoid()
        x231=operator.mul(x225, x230)
        x232=self.conv2d78(x231)
        x233=self.batchnorm2d46(x232)
        x234=operator.add(x233, x219)
        x235=self.conv2d79(x234)
        x236=self.batchnorm2d47(x235)
        x237=self.silu47(x236)
        x238=self.conv2d80(x237)
        x239=self.batchnorm2d48(x238)
        x240=self.silu48(x239)
        x241=x240.mean((2, 3),keepdim=True)
        x242=self.conv2d81(x241)
        x243=self.silu49(x242)
        x244=self.conv2d82(x243)
        x245=x244.sigmoid()
        x246=operator.mul(x240, x245)
        x247=self.conv2d83(x246)
        x248=self.batchnorm2d49(x247)
        x249=operator.add(x248, x234)
        x250=self.conv2d84(x249)
        x251=self.batchnorm2d50(x250)
        x252=self.silu50(x251)
        x253=self.conv2d85(x252)
        x254=self.batchnorm2d51(x253)
        x255=self.silu51(x254)
        x256=x255.mean((2, 3),keepdim=True)
        x257=self.conv2d86(x256)
        x258=self.silu52(x257)
        x259=self.conv2d87(x258)
        x260=x259.sigmoid()
        x261=operator.mul(x255, x260)
        x262=self.conv2d88(x261)
        x263=self.batchnorm2d52(x262)
        x264=operator.add(x263, x249)
        x265=self.conv2d89(x264)
        x266=self.batchnorm2d53(x265)
        x267=self.silu53(x266)
        x268=self.conv2d90(x267)
        x269=self.batchnorm2d54(x268)
        x270=self.silu54(x269)
        x271=x270.mean((2, 3),keepdim=True)
        x272=self.conv2d91(x271)
        x273=self.silu55(x272)
        x274=self.conv2d92(x273)
        x275=x274.sigmoid()
        x276=operator.mul(x270, x275)
        x277=self.conv2d93(x276)
        x278=self.batchnorm2d55(x277)
        x279=self.conv2d94(x278)
        x280=self.batchnorm2d56(x279)
        x281=self.silu56(x280)
        x282=self.conv2d95(x281)
        x283=self.batchnorm2d57(x282)
        x284=self.silu57(x283)
        x285=x284.mean((2, 3),keepdim=True)
        x286=self.conv2d96(x285)
        x287=self.silu58(x286)
        x288=self.conv2d97(x287)
        x289=x288.sigmoid()
        x290=operator.mul(x284, x289)
        x291=self.conv2d98(x290)
        x292=self.batchnorm2d58(x291)
        x293=operator.add(x292, x278)
        x294=self.conv2d99(x293)
        x295=self.batchnorm2d59(x294)
        x296=self.silu59(x295)
        x297=self.conv2d100(x296)
        x298=self.batchnorm2d60(x297)
        x299=self.silu60(x298)
        x300=x299.mean((2, 3),keepdim=True)
        x301=self.conv2d101(x300)
        x302=self.silu61(x301)
        x303=self.conv2d102(x302)
        x304=x303.sigmoid()
        x305=operator.mul(x299, x304)
        x306=self.conv2d103(x305)
        x307=self.batchnorm2d61(x306)
        x308=operator.add(x307, x293)
        x309=self.conv2d104(x308)
        x310=self.batchnorm2d62(x309)
        x311=self.silu62(x310)
        x312=self.conv2d105(x311)
        x313=self.batchnorm2d63(x312)
        x314=self.silu63(x313)
        x315=x314.mean((2, 3),keepdim=True)
        x316=self.conv2d106(x315)
        x317=self.silu64(x316)
        x318=self.conv2d107(x317)
        x319=x318.sigmoid()
        x320=operator.mul(x314, x319)
        x321=self.conv2d108(x320)
        x322=self.batchnorm2d64(x321)
        x323=operator.add(x322, x308)
        x324=self.conv2d109(x323)
        x325=self.batchnorm2d65(x324)
        x326=self.silu65(x325)
        x327=self.conv2d110(x326)
        x328=self.batchnorm2d66(x327)
        x329=self.silu66(x328)
        x330=x329.mean((2, 3),keepdim=True)
        x331=self.conv2d111(x330)
        x332=self.silu67(x331)
        x333=self.conv2d112(x332)
        x334=x333.sigmoid()
        x335=operator.mul(x329, x334)
        x336=self.conv2d113(x335)
        x337=self.batchnorm2d67(x336)
        x338=operator.add(x337, x323)
        x339=self.conv2d114(x338)
        x340=self.batchnorm2d68(x339)
        x341=self.silu68(x340)
        x342=self.conv2d115(x341)
        x343=self.batchnorm2d69(x342)
        x344=self.silu69(x343)
        x345=x344.mean((2, 3),keepdim=True)
        x346=self.conv2d116(x345)
        x347=self.silu70(x346)
        x348=self.conv2d117(x347)
        x349=x348.sigmoid()
        x350=operator.mul(x344, x349)
        x351=self.conv2d118(x350)
        x352=self.batchnorm2d70(x351)
        x353=operator.add(x352, x338)
        x354=self.conv2d119(x353)
        x355=self.batchnorm2d71(x354)
        x356=self.silu71(x355)
        x357=self.conv2d120(x356)
        x358=self.batchnorm2d72(x357)
        x359=self.silu72(x358)
        x360=x359.mean((2, 3),keepdim=True)
        x361=self.conv2d121(x360)
        x362=self.silu73(x361)
        x363=self.conv2d122(x362)
        x364=x363.sigmoid()
        x365=operator.mul(x359, x364)
        x366=self.conv2d123(x365)
        x367=self.batchnorm2d73(x366)
        x368=self.conv2d124(x367)
        x369=self.batchnorm2d74(x368)
        x370=self.silu74(x369)
        x371=self.conv2d125(x370)
        x372=self.batchnorm2d75(x371)
        x373=self.silu75(x372)
        x374=x373.mean((2, 3),keepdim=True)
        x375=self.conv2d126(x374)
        x376=self.silu76(x375)
        x377=self.conv2d127(x376)
        x378=x377.sigmoid()
        x379=operator.mul(x373, x378)
        x380=self.conv2d128(x379)
        x381=self.batchnorm2d76(x380)
        x382=operator.add(x381, x367)
        x383=self.conv2d129(x382)
        x384=self.batchnorm2d77(x383)
        x385=self.silu77(x384)
        x386=self.adaptiveavgpool2d0(x385)
        x387=x386.flatten(1)
        x388=self.linear0(x387)

m = M().eval()
x = torch.randn(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)