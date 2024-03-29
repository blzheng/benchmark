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
        self.conv2d0 = Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu0 = SiLU(inplace=True)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu1 = SiLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d2 = Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.silu2 = SiLU(inplace=True)
        self.conv2d3 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d4 = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d3 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu3 = SiLU(inplace=True)
        self.adaptiveavgpool2d1 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d6 = Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu4 = SiLU(inplace=True)
        self.conv2d7 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d8 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d5 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu5 = SiLU(inplace=True)
        self.adaptiveavgpool2d2 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d10 = Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu6 = SiLU(inplace=True)
        self.conv2d11 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d12 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d7 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu7 = SiLU(inplace=True)
        self.adaptiveavgpool2d3 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d14 = Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu8 = SiLU(inplace=True)
        self.conv2d15 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()
        self.conv2d16 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(192, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu9 = SiLU(inplace=True)
        self.conv2d18 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d10 = BatchNorm2d(192, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu10 = SiLU(inplace=True)
        self.adaptiveavgpool2d4 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d19 = Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
        self.silu11 = SiLU(inplace=True)
        self.conv2d20 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d21 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu12 = SiLU(inplace=True)
        self.conv2d23 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d13 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu13 = SiLU(inplace=True)
        self.adaptiveavgpool2d5 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d24 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu14 = SiLU(inplace=True)
        self.conv2d25 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d26 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu15 = SiLU(inplace=True)
        self.conv2d28 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d16 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu16 = SiLU(inplace=True)
        self.adaptiveavgpool2d6 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d29 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu17 = SiLU(inplace=True)
        self.conv2d30 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d31 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d32 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu18 = SiLU(inplace=True)
        self.conv2d33 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d19 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu19 = SiLU(inplace=True)
        self.adaptiveavgpool2d7 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d34 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu20 = SiLU(inplace=True)
        self.conv2d35 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d36 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu21 = SiLU(inplace=True)
        self.conv2d38 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d22 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu22 = SiLU(inplace=True)
        self.adaptiveavgpool2d8 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d39 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu23 = SiLU(inplace=True)
        self.conv2d40 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d41 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu24 = SiLU(inplace=True)
        self.conv2d43 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d25 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu25 = SiLU(inplace=True)
        self.adaptiveavgpool2d9 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d44 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu26 = SiLU(inplace=True)
        self.conv2d45 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d46 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d47 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu27 = SiLU(inplace=True)
        self.conv2d48 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d28 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu28 = SiLU(inplace=True)
        self.adaptiveavgpool2d10 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d49 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu29 = SiLU(inplace=True)
        self.conv2d50 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d51 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu30 = SiLU(inplace=True)
        self.conv2d53 = Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
        self.batchnorm2d31 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu31 = SiLU(inplace=True)
        self.adaptiveavgpool2d11 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d54 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
        self.silu32 = SiLU(inplace=True)
        self.conv2d55 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()
        self.conv2d56 = Conv2d(288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d57 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu33 = SiLU(inplace=True)
        self.conv2d58 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d34 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu34 = SiLU(inplace=True)
        self.adaptiveavgpool2d12 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d59 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu35 = SiLU(inplace=True)
        self.conv2d60 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()
        self.conv2d61 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d62 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu36 = SiLU(inplace=True)
        self.conv2d63 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d37 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu37 = SiLU(inplace=True)
        self.adaptiveavgpool2d13 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d64 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu38 = SiLU(inplace=True)
        self.conv2d65 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d66 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d67 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu39 = SiLU(inplace=True)
        self.conv2d68 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d40 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu40 = SiLU(inplace=True)
        self.adaptiveavgpool2d14 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d69 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu41 = SiLU(inplace=True)
        self.conv2d70 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d71 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d72 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu42 = SiLU(inplace=True)
        self.conv2d73 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d43 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu43 = SiLU(inplace=True)
        self.adaptiveavgpool2d15 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d74 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu44 = SiLU(inplace=True)
        self.conv2d75 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d76 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d77 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu45 = SiLU(inplace=True)
        self.conv2d78 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d46 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu46 = SiLU(inplace=True)
        self.adaptiveavgpool2d16 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d79 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu47 = SiLU(inplace=True)
        self.conv2d80 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d81 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d82 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu48 = SiLU(inplace=True)
        self.conv2d83 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d49 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu49 = SiLU(inplace=True)
        self.adaptiveavgpool2d17 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d84 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu50 = SiLU(inplace=True)
        self.conv2d85 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d87 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu51 = SiLU(inplace=True)
        self.conv2d88 = Conv2d(480, 480, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d52 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu52 = SiLU(inplace=True)
        self.adaptiveavgpool2d18 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d89 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
        self.silu53 = SiLU(inplace=True)
        self.conv2d90 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()
        self.conv2d91 = Conv2d(480, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d92 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu54 = SiLU(inplace=True)
        self.conv2d93 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d55 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu55 = SiLU(inplace=True)
        self.adaptiveavgpool2d19 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d94 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu56 = SiLU(inplace=True)
        self.conv2d95 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d96 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d97 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu57 = SiLU(inplace=True)
        self.conv2d98 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d58 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu58 = SiLU(inplace=True)
        self.adaptiveavgpool2d20 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d99 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu59 = SiLU(inplace=True)
        self.conv2d100 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()
        self.conv2d101 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d102 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu60 = SiLU(inplace=True)
        self.conv2d103 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d61 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu61 = SiLU(inplace=True)
        self.adaptiveavgpool2d21 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d104 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu62 = SiLU(inplace=True)
        self.conv2d105 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d106 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d107 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu63 = SiLU(inplace=True)
        self.conv2d108 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d64 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu64 = SiLU(inplace=True)
        self.adaptiveavgpool2d22 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d109 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu65 = SiLU(inplace=True)
        self.conv2d110 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d111 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d112 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu66 = SiLU(inplace=True)
        self.conv2d113 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d67 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu67 = SiLU(inplace=True)
        self.adaptiveavgpool2d23 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d114 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu68 = SiLU(inplace=True)
        self.conv2d115 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()
        self.conv2d116 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d117 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu69 = SiLU(inplace=True)
        self.conv2d118 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d70 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu70 = SiLU(inplace=True)
        self.adaptiveavgpool2d24 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d119 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu71 = SiLU(inplace=True)
        self.conv2d120 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()
        self.conv2d121 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d122 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu72 = SiLU(inplace=True)
        self.conv2d123 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d73 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu73 = SiLU(inplace=True)
        self.adaptiveavgpool2d25 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d124 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu74 = SiLU(inplace=True)
        self.conv2d125 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d126 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d127 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu75 = SiLU(inplace=True)
        self.conv2d128 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d76 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu76 = SiLU(inplace=True)
        self.adaptiveavgpool2d26 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d129 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu77 = SiLU(inplace=True)
        self.conv2d130 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid26 = Sigmoid()
        self.conv2d131 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d132 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu78 = SiLU(inplace=True)
        self.conv2d133 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d79 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu79 = SiLU(inplace=True)
        self.adaptiveavgpool2d27 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d134 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu80 = SiLU(inplace=True)
        self.conv2d135 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()
        self.conv2d136 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d137 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu81 = SiLU(inplace=True)
        self.conv2d138 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        self.batchnorm2d82 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu82 = SiLU(inplace=True)
        self.adaptiveavgpool2d28 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d139 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
        self.silu83 = SiLU(inplace=True)
        self.conv2d140 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()
        self.conv2d141 = Conv2d(960, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d83 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d142 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu84 = SiLU(inplace=True)
        self.conv2d143 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d85 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu85 = SiLU(inplace=True)
        self.adaptiveavgpool2d29 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d144 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu86 = SiLU(inplace=True)
        self.conv2d145 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid29 = Sigmoid()
        self.conv2d146 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d147 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu87 = SiLU(inplace=True)
        self.conv2d148 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d88 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu88 = SiLU(inplace=True)
        self.adaptiveavgpool2d30 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d149 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu89 = SiLU(inplace=True)
        self.conv2d150 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()
        self.conv2d151 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d152 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu90 = SiLU(inplace=True)
        self.conv2d153 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d91 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu91 = SiLU(inplace=True)
        self.adaptiveavgpool2d31 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d154 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu92 = SiLU(inplace=True)
        self.conv2d155 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()
        self.conv2d156 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d92 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d157 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu93 = SiLU(inplace=True)
        self.conv2d158 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d94 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu94 = SiLU(inplace=True)
        self.adaptiveavgpool2d32 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d159 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu95 = SiLU(inplace=True)
        self.conv2d160 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid32 = Sigmoid()
        self.conv2d161 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d162 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu96 = SiLU(inplace=True)
        self.conv2d163 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d97 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu97 = SiLU(inplace=True)
        self.adaptiveavgpool2d33 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d164 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu98 = SiLU(inplace=True)
        self.conv2d165 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid33 = Sigmoid()
        self.conv2d166 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d167 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d99 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu99 = SiLU(inplace=True)
        self.conv2d168 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d100 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu100 = SiLU(inplace=True)
        self.adaptiveavgpool2d34 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d169 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu101 = SiLU(inplace=True)
        self.conv2d170 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid34 = Sigmoid()
        self.conv2d171 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d172 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu102 = SiLU(inplace=True)
        self.conv2d173 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d103 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu103 = SiLU(inplace=True)
        self.adaptiveavgpool2d35 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d174 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu104 = SiLU(inplace=True)
        self.conv2d175 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid35 = Sigmoid()
        self.conv2d176 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d177 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu105 = SiLU(inplace=True)
        self.conv2d178 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d106 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu106 = SiLU(inplace=True)
        self.adaptiveavgpool2d36 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d179 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu107 = SiLU(inplace=True)
        self.conv2d180 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()
        self.conv2d181 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d182 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu108 = SiLU(inplace=True)
        self.conv2d183 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d109 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu109 = SiLU(inplace=True)
        self.adaptiveavgpool2d37 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d184 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu110 = SiLU(inplace=True)
        self.conv2d185 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()
        self.conv2d186 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d187 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu111 = SiLU(inplace=True)
        self.conv2d188 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d112 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu112 = SiLU(inplace=True)
        self.adaptiveavgpool2d38 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d189 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))
        self.silu113 = SiLU(inplace=True)
        self.conv2d190 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d192 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu114 = SiLU(inplace=True)
        self.conv2d193 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d115 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu115 = SiLU(inplace=True)
        self.adaptiveavgpool2d39 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d194 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu116 = SiLU(inplace=True)
        self.conv2d195 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid39 = Sigmoid()
        self.conv2d196 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d197 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d117 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu117 = SiLU(inplace=True)
        self.conv2d198 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d118 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu118 = SiLU(inplace=True)
        self.adaptiveavgpool2d40 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d199 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu119 = SiLU(inplace=True)
        self.conv2d200 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid40 = Sigmoid()
        self.conv2d201 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d202 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d120 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu120 = SiLU(inplace=True)
        self.conv2d203 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d121 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu121 = SiLU(inplace=True)
        self.adaptiveavgpool2d41 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d204 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu122 = SiLU(inplace=True)
        self.conv2d205 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid41 = Sigmoid()
        self.conv2d206 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d207 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d123 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu123 = SiLU(inplace=True)
        self.conv2d208 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d124 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu124 = SiLU(inplace=True)
        self.adaptiveavgpool2d42 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d209 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu125 = SiLU(inplace=True)
        self.conv2d210 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid42 = Sigmoid()
        self.conv2d211 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d125 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d212 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu126 = SiLU(inplace=True)
        self.conv2d213 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d127 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu127 = SiLU(inplace=True)
        self.adaptiveavgpool2d43 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d214 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu128 = SiLU(inplace=True)
        self.conv2d215 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid43 = Sigmoid()
        self.conv2d216 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d128 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d217 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu129 = SiLU(inplace=True)
        self.conv2d218 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d130 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu130 = SiLU(inplace=True)
        self.adaptiveavgpool2d44 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d219 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu131 = SiLU(inplace=True)
        self.conv2d220 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid44 = Sigmoid()
        self.conv2d221 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d131 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d222 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d132 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu132 = SiLU(inplace=True)
        self.conv2d223 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d133 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu133 = SiLU(inplace=True)
        self.adaptiveavgpool2d45 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d224 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu134 = SiLU(inplace=True)
        self.conv2d225 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid45 = Sigmoid()
        self.conv2d226 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d134 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d227 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d135 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu135 = SiLU(inplace=True)
        self.conv2d228 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d136 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu136 = SiLU(inplace=True)
        self.adaptiveavgpool2d46 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d229 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu137 = SiLU(inplace=True)
        self.conv2d230 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid46 = Sigmoid()
        self.conv2d231 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d232 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d138 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu138 = SiLU(inplace=True)
        self.conv2d233 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d139 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu139 = SiLU(inplace=True)
        self.adaptiveavgpool2d47 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d234 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu140 = SiLU(inplace=True)
        self.conv2d235 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid47 = Sigmoid()
        self.conv2d236 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d237 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d141 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu141 = SiLU(inplace=True)
        self.conv2d238 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d142 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu142 = SiLU(inplace=True)
        self.adaptiveavgpool2d48 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d239 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu143 = SiLU(inplace=True)
        self.conv2d240 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid48 = Sigmoid()
        self.conv2d241 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d143 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d242 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d144 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu144 = SiLU(inplace=True)
        self.conv2d243 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d145 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu145 = SiLU(inplace=True)
        self.adaptiveavgpool2d49 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d244 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu146 = SiLU(inplace=True)
        self.conv2d245 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid49 = Sigmoid()
        self.conv2d246 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d146 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d247 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d147 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu147 = SiLU(inplace=True)
        self.conv2d248 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d148 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu148 = SiLU(inplace=True)
        self.adaptiveavgpool2d50 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d249 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu149 = SiLU(inplace=True)
        self.conv2d250 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid50 = Sigmoid()
        self.conv2d251 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d149 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d252 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d150 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu150 = SiLU(inplace=True)
        self.conv2d253 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
        self.batchnorm2d151 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu151 = SiLU(inplace=True)
        self.adaptiveavgpool2d51 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d254 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))
        self.silu152 = SiLU(inplace=True)
        self.conv2d255 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid51 = Sigmoid()
        self.conv2d256 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d257 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d153 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu153 = SiLU(inplace=True)
        self.conv2d258 = Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
        self.batchnorm2d154 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu154 = SiLU(inplace=True)
        self.adaptiveavgpool2d52 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d259 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
        self.silu155 = SiLU(inplace=True)
        self.conv2d260 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid52 = Sigmoid()
        self.conv2d261 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d155 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d262 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d156 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu156 = SiLU(inplace=True)
        self.conv2d263 = Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
        self.batchnorm2d157 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu157 = SiLU(inplace=True)
        self.adaptiveavgpool2d53 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d264 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
        self.silu158 = SiLU(inplace=True)
        self.conv2d265 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid53 = Sigmoid()
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d158 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d267 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d159 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu159 = SiLU(inplace=True)
        self.conv2d268 = Conv2d(3840, 3840, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3840, bias=False)
        self.batchnorm2d160 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu160 = SiLU(inplace=True)
        self.adaptiveavgpool2d54 = AdaptiveAvgPool2d(output_size=1)
        self.conv2d269 = Conv2d(3840, 160, kernel_size=(1, 1), stride=(1, 1))
        self.silu161 = SiLU(inplace=True)
        self.conv2d270 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid54 = Sigmoid()
        self.conv2d271 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d161 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d272 = Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d162 = BatchNorm2d(2560, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.silu162 = SiLU(inplace=True)
        self.adaptiveavgpool2d55 = AdaptiveAvgPool2d(output_size=1)
        self.dropout0 = Dropout(p=0.5, inplace=True)
        self.linear0 = Linear(in_features=2560, out_features=1000, bias=True)

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
        x26=stochastic_depth(x25, 0.0036363636363636364, 'row', False)
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
        x31=self.adaptiveavgpool2d2(x30)
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
        x32=self.conv2d10(x31)
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
        x34=self.conv2d11(x33)
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
        x35=self.sigmoid2(x34)
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
        x36=operator.mul(x35, x30)
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
        x38=self.batchnorm2d6(x37)
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
        x39=stochastic_depth(x38, 0.007272727272727273, 'row', False)
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
        x40=operator.add(x39, x27)
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
        x41=self.conv2d13(x40)
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
        x42=self.batchnorm2d7(x41)
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
        x43=self.silu7(x42)
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
        x44=self.adaptiveavgpool2d3(x43)
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
        x45=self.conv2d14(x44)
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
        x46=self.silu8(x45)
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
        x47=self.conv2d15(x46)
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
        x48=self.sigmoid3(x47)
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
        x49=operator.mul(x48, x43)
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
        x50=self.conv2d16(x49)
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
        x51=self.batchnorm2d8(x50)
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
        x52=stochastic_depth(x51, 0.01090909090909091, 'row', False)
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
        x53=operator.add(x52, x40)
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
        x54=self.conv2d17(x53)
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
        x55=self.batchnorm2d9(x54)
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
        x56=self.silu9(x55)
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
        x57=self.conv2d18(x56)
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
        x58=self.batchnorm2d10(x57)
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
        x59=self.silu10(x58)
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
        x60=self.adaptiveavgpool2d4(x59)
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
        x61=self.conv2d19(x60)
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
        x62=self.silu11(x61)
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
        x63=self.conv2d20(x62)
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
        x64=self.sigmoid4(x63)
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
        x65=operator.mul(x64, x59)
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
        x66=self.conv2d21(x65)
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
        x67=self.batchnorm2d11(x66)
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
        x68=self.conv2d22(x67)
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
        x69=self.batchnorm2d12(x68)
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
        x70=self.silu12(x69)
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
        x71=self.conv2d23(x70)
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
        x72=self.batchnorm2d13(x71)
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
        x73=self.silu13(x72)
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
        x74=self.adaptiveavgpool2d5(x73)
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
        x75=self.conv2d24(x74)
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
        x78=self.sigmoid5(x77)
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
        x79=operator.mul(x78, x73)
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
        x80=self.conv2d26(x79)
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
        x81=self.batchnorm2d14(x80)
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
        x82=stochastic_depth(x81, 0.01818181818181818, 'row', False)
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
        x83=operator.add(x82, x67)
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
        x84=self.conv2d27(x83)
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
        x85=self.batchnorm2d15(x84)
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
        x86=self.silu15(x85)
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
        x87=self.conv2d28(x86)
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
        x88=self.batchnorm2d16(x87)
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
        x89=self.silu16(x88)
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
        x90=self.adaptiveavgpool2d6(x89)
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
        x91=self.conv2d29(x90)
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
        x92=self.silu17(x91)
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
        x93=self.conv2d30(x92)
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
        x94=self.sigmoid6(x93)
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
        x95=operator.mul(x94, x89)
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
        x96=self.conv2d31(x95)
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
        x97=self.batchnorm2d17(x96)
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
        x98=stochastic_depth(x97, 0.02181818181818182, 'row', False)
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
        x99=operator.add(x98, x83)
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
        x100=self.conv2d32(x99)
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
        x101=self.batchnorm2d18(x100)
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
        x102=self.silu18(x101)
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
        x103=self.conv2d33(x102)
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
        x104=self.batchnorm2d19(x103)
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
        x105=self.silu19(x104)
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
        x106=self.adaptiveavgpool2d7(x105)
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
        x107=self.conv2d34(x106)
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
        x108=self.silu20(x107)
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
        x109=self.conv2d35(x108)
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
        x110=self.sigmoid7(x109)
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
        x111=operator.mul(x110, x105)
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
        x112=self.conv2d36(x111)
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
        x113=self.batchnorm2d20(x112)
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
        x114=stochastic_depth(x113, 0.025454545454545455, 'row', False)
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
        x115=operator.add(x114, x99)
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
        x116=self.conv2d37(x115)
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
        x117=self.batchnorm2d21(x116)
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
        x118=self.silu21(x117)
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
        x119=self.conv2d38(x118)
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
        x120=self.batchnorm2d22(x119)
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
        x121=self.silu22(x120)
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
        x122=self.adaptiveavgpool2d8(x121)
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
        x123=self.conv2d39(x122)
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
        x124=self.silu23(x123)
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
        x125=self.conv2d40(x124)
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
        x126=self.sigmoid8(x125)
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
        x127=operator.mul(x126, x121)
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
        x128=self.conv2d41(x127)
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
        x129=self.batchnorm2d23(x128)
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
        x130=stochastic_depth(x129, 0.02909090909090909, 'row', False)
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
        x131=operator.add(x130, x115)
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
        x132=self.conv2d42(x131)
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
        x133=self.batchnorm2d24(x132)
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
        x134=self.silu24(x133)
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
        x135=self.conv2d43(x134)
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
        x136=self.batchnorm2d25(x135)
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
        x137=self.silu25(x136)
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
        x138=self.adaptiveavgpool2d9(x137)
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
        x139=self.conv2d44(x138)
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
        x140=self.silu26(x139)
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
        x141=self.conv2d45(x140)
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
        x142=self.sigmoid9(x141)
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
        x143=operator.mul(x142, x137)
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
        x144=self.conv2d46(x143)
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
        x145=self.batchnorm2d26(x144)
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
        x146=stochastic_depth(x145, 0.03272727272727273, 'row', False)
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
        x147=operator.add(x146, x131)
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
        x148=self.conv2d47(x147)
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
        x149=self.batchnorm2d27(x148)
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
        x150=self.silu27(x149)
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
        x151=self.conv2d48(x150)
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
        x152=self.batchnorm2d28(x151)
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
        x153=self.silu28(x152)
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
        x154=self.adaptiveavgpool2d10(x153)
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
        x155=self.conv2d49(x154)
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
        x156=self.silu29(x155)
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
        x157=self.conv2d50(x156)
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
        x158=self.sigmoid10(x157)
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
        x159=operator.mul(x158, x153)
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
        x160=self.conv2d51(x159)
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
        x161=self.batchnorm2d29(x160)
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
        x162=stochastic_depth(x161, 0.03636363636363636, 'row', False)
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
        x163=operator.add(x162, x147)
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
        x164=self.conv2d52(x163)
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
        x165=self.batchnorm2d30(x164)
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
        x166=self.silu30(x165)
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
        x167=self.conv2d53(x166)
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
        x168=self.batchnorm2d31(x167)
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
        x169=self.silu31(x168)
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
        x170=self.adaptiveavgpool2d11(x169)
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
        x171=self.conv2d54(x170)
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
        x172=self.silu32(x171)
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
        x173=self.conv2d55(x172)
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
        x174=self.sigmoid11(x173)
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
        x175=operator.mul(x174, x169)
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
        x176=self.conv2d56(x175)
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
        x177=self.batchnorm2d32(x176)
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
        x178=self.conv2d57(x177)
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
        x179=self.batchnorm2d33(x178)
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
        x180=self.silu33(x179)
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
        x181=self.conv2d58(x180)
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
        x182=self.batchnorm2d34(x181)
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
        x183=self.silu34(x182)
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
        x184=self.adaptiveavgpool2d12(x183)
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
        x185=self.conv2d59(x184)
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
        x186=self.silu35(x185)
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
        x187=self.conv2d60(x186)
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
        x188=self.sigmoid12(x187)
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
        x189=operator.mul(x188, x183)
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
        x190=self.conv2d61(x189)
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
        x191=self.batchnorm2d35(x190)
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
        x192=stochastic_depth(x191, 0.04363636363636364, 'row', False)
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
        x193=operator.add(x192, x177)
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
        x194=self.conv2d62(x193)
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
        x195=self.batchnorm2d36(x194)
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
        x196=self.silu36(x195)
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
        x197=self.conv2d63(x196)
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
        x198=self.batchnorm2d37(x197)
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
        x199=self.silu37(x198)
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
        x200=self.adaptiveavgpool2d13(x199)
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
        x201=self.conv2d64(x200)
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
        x202=self.silu38(x201)
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
        x203=self.conv2d65(x202)
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
        x204=self.sigmoid13(x203)
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
        x205=operator.mul(x204, x199)
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
        x206=self.conv2d66(x205)
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
        x207=self.batchnorm2d38(x206)
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
        x208=stochastic_depth(x207, 0.04727272727272727, 'row', False)
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
        x209=operator.add(x208, x193)
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
        x210=self.conv2d67(x209)
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
        x211=self.batchnorm2d39(x210)
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
        x212=self.silu39(x211)
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
        x213=self.conv2d68(x212)
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
        x214=self.batchnorm2d40(x213)
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
        x215=self.silu40(x214)
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
        x216=self.adaptiveavgpool2d14(x215)
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
        x217=self.conv2d69(x216)
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
        x218=self.silu41(x217)
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
        x219=self.conv2d70(x218)
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
        x220=self.sigmoid14(x219)
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
        x221=operator.mul(x220, x215)
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
        x222=self.conv2d71(x221)
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
        x223=self.batchnorm2d41(x222)
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
        x224=stochastic_depth(x223, 0.05090909090909091, 'row', False)
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
        x225=operator.add(x224, x209)
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
        x226=self.conv2d72(x225)
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
        x227=self.batchnorm2d42(x226)
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
        x228=self.silu42(x227)
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
        x229=self.conv2d73(x228)
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
        x230=self.batchnorm2d43(x229)
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
        x231=self.silu43(x230)
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
        x232=self.adaptiveavgpool2d15(x231)
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
        x233=self.conv2d74(x232)
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
        x234=self.silu44(x233)
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
        x235=self.conv2d75(x234)
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
        x236=self.sigmoid15(x235)
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
        x237=operator.mul(x236, x231)
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
        x238=self.conv2d76(x237)
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
        x239=self.batchnorm2d44(x238)
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
        x240=stochastic_depth(x239, 0.05454545454545454, 'row', False)
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
        x241=operator.add(x240, x225)
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
        x242=self.conv2d77(x241)
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
        x243=self.batchnorm2d45(x242)
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
        x244=self.silu45(x243)
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
        x245=self.conv2d78(x244)
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
        x246=self.batchnorm2d46(x245)
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
        x247=self.silu46(x246)
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
        x248=self.adaptiveavgpool2d16(x247)
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
        x249=self.conv2d79(x248)
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
        x250=self.silu47(x249)
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
        x251=self.conv2d80(x250)
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
        x252=self.sigmoid16(x251)
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
        x253=operator.mul(x252, x247)
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
        x254=self.conv2d81(x253)
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
        x255=self.batchnorm2d47(x254)
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
        x256=stochastic_depth(x255, 0.05818181818181818, 'row', False)
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
        x257=operator.add(x256, x241)
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
        x258=self.conv2d82(x257)
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
        x259=self.batchnorm2d48(x258)
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
        x260=self.silu48(x259)
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
        x261=self.conv2d83(x260)
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
        x262=self.batchnorm2d49(x261)
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
        x263=self.silu49(x262)
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
        x264=self.adaptiveavgpool2d17(x263)
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
        x265=self.conv2d84(x264)
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
        x266=self.silu50(x265)
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
        x267=self.conv2d85(x266)
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
        x268=self.sigmoid17(x267)
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
        x269=operator.mul(x268, x263)
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
        x270=self.conv2d86(x269)
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
        x271=self.batchnorm2d50(x270)
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
        x272=stochastic_depth(x271, 0.06181818181818183, 'row', False)
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
        x273=operator.add(x272, x257)
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
        x274=self.conv2d87(x273)
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
        x275=self.batchnorm2d51(x274)
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
        x276=self.silu51(x275)
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
        x277=self.conv2d88(x276)
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
        x278=self.batchnorm2d52(x277)
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
        x279=self.silu52(x278)
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
        x280=self.adaptiveavgpool2d18(x279)
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
        x281=self.conv2d89(x280)
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
        x282=self.silu53(x281)
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
        x283=self.conv2d90(x282)
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
        x284=self.sigmoid18(x283)
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
        x285=operator.mul(x284, x279)
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
        x286=self.conv2d91(x285)
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
        x287=self.batchnorm2d53(x286)
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
        x288=self.conv2d92(x287)
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
        x289=self.batchnorm2d54(x288)
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
        x290=self.silu54(x289)
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
        x291=self.conv2d93(x290)
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
        x292=self.batchnorm2d55(x291)
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
        x293=self.silu55(x292)
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
        x294=self.adaptiveavgpool2d19(x293)
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
        x295=self.conv2d94(x294)
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
        x296=self.silu56(x295)
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
        x297=self.conv2d95(x296)
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
        x298=self.sigmoid19(x297)
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
        x299=operator.mul(x298, x293)
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
        x300=self.conv2d96(x299)
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
        x301=self.batchnorm2d56(x300)
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
        x302=stochastic_depth(x301, 0.06909090909090909, 'row', False)
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
        x303=operator.add(x302, x287)
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
        x304=self.conv2d97(x303)
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
        x305=self.batchnorm2d57(x304)
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
        x306=self.silu57(x305)
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
        x307=self.conv2d98(x306)
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
        x308=self.batchnorm2d58(x307)
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
        x309=self.silu58(x308)
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
        x310=self.adaptiveavgpool2d20(x309)
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
        x311=self.conv2d99(x310)
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
        x312=self.silu59(x311)
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
        x313=self.conv2d100(x312)
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
        x314=self.sigmoid20(x313)
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
        x315=operator.mul(x314, x309)
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
        x316=self.conv2d101(x315)
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
        x317=self.batchnorm2d59(x316)
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
        x318=stochastic_depth(x317, 0.07272727272727272, 'row', False)
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
        x319=operator.add(x318, x303)
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
        x320=self.conv2d102(x319)
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
        x321=self.batchnorm2d60(x320)
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
        x322=self.silu60(x321)
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
        x323=self.conv2d103(x322)
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
        x324=self.batchnorm2d61(x323)
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
        x325=self.silu61(x324)
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
        x326=self.adaptiveavgpool2d21(x325)
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
        x327=self.conv2d104(x326)
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
        x328=self.silu62(x327)
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
        x329=self.conv2d105(x328)
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
        x330=self.sigmoid21(x329)
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
        x331=operator.mul(x330, x325)
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
        x332=self.conv2d106(x331)
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
        x333=self.batchnorm2d62(x332)
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
        x334=stochastic_depth(x333, 0.07636363636363637, 'row', False)
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
        x335=operator.add(x334, x319)
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
        x336=self.conv2d107(x335)
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
        x337=self.batchnorm2d63(x336)
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
        x338=self.silu63(x337)
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
        x339=self.conv2d108(x338)
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
        x340=self.batchnorm2d64(x339)
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
        x341=self.silu64(x340)
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
        x342=self.adaptiveavgpool2d22(x341)
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
        x343=self.conv2d109(x342)
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
        x344=self.silu65(x343)
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
        x345=self.conv2d110(x344)
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
        x346=self.sigmoid22(x345)
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
        x347=operator.mul(x346, x341)
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
        x348=self.conv2d111(x347)
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
        x349=self.batchnorm2d65(x348)
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
        x350=stochastic_depth(x349, 0.08, 'row', False)
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
        x351=operator.add(x350, x335)
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
        x352=self.conv2d112(x351)
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
        x353=self.batchnorm2d66(x352)
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
        x354=self.silu66(x353)
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
        x355=self.conv2d113(x354)
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
        x356=self.batchnorm2d67(x355)
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
        x357=self.silu67(x356)
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
        x358=self.adaptiveavgpool2d23(x357)
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
        x359=self.conv2d114(x358)
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
        x360=self.silu68(x359)
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
        x361=self.conv2d115(x360)
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
        x362=self.sigmoid23(x361)
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
        x363=operator.mul(x362, x357)
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
        x364=self.conv2d116(x363)
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
        x365=self.batchnorm2d68(x364)
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
        x366=stochastic_depth(x365, 0.08363636363636365, 'row', False)
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
        x367=operator.add(x366, x351)
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
        x368=self.conv2d117(x367)
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
        x369=self.batchnorm2d69(x368)
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
        x370=self.silu69(x369)
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
        x371=self.conv2d118(x370)
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
        x372=self.batchnorm2d70(x371)
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
        x373=self.silu70(x372)
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
        x374=self.adaptiveavgpool2d24(x373)
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
        x375=self.conv2d119(x374)
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
        x376=self.silu71(x375)
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
        x377=self.conv2d120(x376)
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
        x378=self.sigmoid24(x377)
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
        x379=operator.mul(x378, x373)
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
        x380=self.conv2d121(x379)
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
        x381=self.batchnorm2d71(x380)
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
        x382=stochastic_depth(x381, 0.08727272727272728, 'row', False)
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
        x383=operator.add(x382, x367)
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
        x384=self.conv2d122(x383)
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
        x385=self.batchnorm2d72(x384)
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
        x386=self.silu72(x385)
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
        x387=self.conv2d123(x386)
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
        x388=self.batchnorm2d73(x387)
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
        x389=self.silu73(x388)
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
        x391=self.conv2d124(x390)
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
        x392=self.silu74(x391)
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
        x393=self.conv2d125(x392)
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
        x396=self.conv2d126(x395)
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
        x397=self.batchnorm2d74(x396)
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
        x398=stochastic_depth(x397, 0.09090909090909091, 'row', False)
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
        x400=self.conv2d127(x399)
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
        x401=self.batchnorm2d75(x400)
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
        x402=self.silu75(x401)
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
        x403=self.conv2d128(x402)
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
        x404=self.batchnorm2d76(x403)
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
        x405=self.silu76(x404)
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
        x406=self.adaptiveavgpool2d26(x405)
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
        x407=self.conv2d129(x406)
        if x407 is None:
            print('x407: {}'.format(x407))
        elif isinstance(x407, torch.Tensor):
            print('x407: {}'.format(x407.shape))
        elif isinstance(x407, tuple):
            tuple_shapes = '('
            for item in x407:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x407: {}'.format(tuple_shapes))
        else:
            print('x407: {}'.format(x407))
        x408=self.silu77(x407)
        if x408 is None:
            print('x408: {}'.format(x408))
        elif isinstance(x408, torch.Tensor):
            print('x408: {}'.format(x408.shape))
        elif isinstance(x408, tuple):
            tuple_shapes = '('
            for item in x408:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x408: {}'.format(tuple_shapes))
        else:
            print('x408: {}'.format(x408))
        x409=self.conv2d130(x408)
        if x409 is None:
            print('x409: {}'.format(x409))
        elif isinstance(x409, torch.Tensor):
            print('x409: {}'.format(x409.shape))
        elif isinstance(x409, tuple):
            tuple_shapes = '('
            for item in x409:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x409: {}'.format(tuple_shapes))
        else:
            print('x409: {}'.format(x409))
        x410=self.sigmoid26(x409)
        if x410 is None:
            print('x410: {}'.format(x410))
        elif isinstance(x410, torch.Tensor):
            print('x410: {}'.format(x410.shape))
        elif isinstance(x410, tuple):
            tuple_shapes = '('
            for item in x410:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x410: {}'.format(tuple_shapes))
        else:
            print('x410: {}'.format(x410))
        x411=operator.mul(x410, x405)
        if x411 is None:
            print('x411: {}'.format(x411))
        elif isinstance(x411, torch.Tensor):
            print('x411: {}'.format(x411.shape))
        elif isinstance(x411, tuple):
            tuple_shapes = '('
            for item in x411:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x411: {}'.format(tuple_shapes))
        else:
            print('x411: {}'.format(x411))
        x412=self.conv2d131(x411)
        if x412 is None:
            print('x412: {}'.format(x412))
        elif isinstance(x412, torch.Tensor):
            print('x412: {}'.format(x412.shape))
        elif isinstance(x412, tuple):
            tuple_shapes = '('
            for item in x412:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x412: {}'.format(tuple_shapes))
        else:
            print('x412: {}'.format(x412))
        x413=self.batchnorm2d77(x412)
        if x413 is None:
            print('x413: {}'.format(x413))
        elif isinstance(x413, torch.Tensor):
            print('x413: {}'.format(x413.shape))
        elif isinstance(x413, tuple):
            tuple_shapes = '('
            for item in x413:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x413: {}'.format(tuple_shapes))
        else:
            print('x413: {}'.format(x413))
        x414=stochastic_depth(x413, 0.09454545454545454, 'row', False)
        if x414 is None:
            print('x414: {}'.format(x414))
        elif isinstance(x414, torch.Tensor):
            print('x414: {}'.format(x414.shape))
        elif isinstance(x414, tuple):
            tuple_shapes = '('
            for item in x414:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x414: {}'.format(tuple_shapes))
        else:
            print('x414: {}'.format(x414))
        x415=operator.add(x414, x399)
        if x415 is None:
            print('x415: {}'.format(x415))
        elif isinstance(x415, torch.Tensor):
            print('x415: {}'.format(x415.shape))
        elif isinstance(x415, tuple):
            tuple_shapes = '('
            for item in x415:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x415: {}'.format(tuple_shapes))
        else:
            print('x415: {}'.format(x415))
        x416=self.conv2d132(x415)
        if x416 is None:
            print('x416: {}'.format(x416))
        elif isinstance(x416, torch.Tensor):
            print('x416: {}'.format(x416.shape))
        elif isinstance(x416, tuple):
            tuple_shapes = '('
            for item in x416:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x416: {}'.format(tuple_shapes))
        else:
            print('x416: {}'.format(x416))
        x417=self.batchnorm2d78(x416)
        if x417 is None:
            print('x417: {}'.format(x417))
        elif isinstance(x417, torch.Tensor):
            print('x417: {}'.format(x417.shape))
        elif isinstance(x417, tuple):
            tuple_shapes = '('
            for item in x417:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x417: {}'.format(tuple_shapes))
        else:
            print('x417: {}'.format(x417))
        x418=self.silu78(x417)
        if x418 is None:
            print('x418: {}'.format(x418))
        elif isinstance(x418, torch.Tensor):
            print('x418: {}'.format(x418.shape))
        elif isinstance(x418, tuple):
            tuple_shapes = '('
            for item in x418:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x418: {}'.format(tuple_shapes))
        else:
            print('x418: {}'.format(x418))
        x419=self.conv2d133(x418)
        if x419 is None:
            print('x419: {}'.format(x419))
        elif isinstance(x419, torch.Tensor):
            print('x419: {}'.format(x419.shape))
        elif isinstance(x419, tuple):
            tuple_shapes = '('
            for item in x419:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x419: {}'.format(tuple_shapes))
        else:
            print('x419: {}'.format(x419))
        x420=self.batchnorm2d79(x419)
        if x420 is None:
            print('x420: {}'.format(x420))
        elif isinstance(x420, torch.Tensor):
            print('x420: {}'.format(x420.shape))
        elif isinstance(x420, tuple):
            tuple_shapes = '('
            for item in x420:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x420: {}'.format(tuple_shapes))
        else:
            print('x420: {}'.format(x420))
        x421=self.silu79(x420)
        if x421 is None:
            print('x421: {}'.format(x421))
        elif isinstance(x421, torch.Tensor):
            print('x421: {}'.format(x421.shape))
        elif isinstance(x421, tuple):
            tuple_shapes = '('
            for item in x421:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x421: {}'.format(tuple_shapes))
        else:
            print('x421: {}'.format(x421))
        x422=self.adaptiveavgpool2d27(x421)
        if x422 is None:
            print('x422: {}'.format(x422))
        elif isinstance(x422, torch.Tensor):
            print('x422: {}'.format(x422.shape))
        elif isinstance(x422, tuple):
            tuple_shapes = '('
            for item in x422:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x422: {}'.format(tuple_shapes))
        else:
            print('x422: {}'.format(x422))
        x423=self.conv2d134(x422)
        if x423 is None:
            print('x423: {}'.format(x423))
        elif isinstance(x423, torch.Tensor):
            print('x423: {}'.format(x423.shape))
        elif isinstance(x423, tuple):
            tuple_shapes = '('
            for item in x423:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x423: {}'.format(tuple_shapes))
        else:
            print('x423: {}'.format(x423))
        x424=self.silu80(x423)
        if x424 is None:
            print('x424: {}'.format(x424))
        elif isinstance(x424, torch.Tensor):
            print('x424: {}'.format(x424.shape))
        elif isinstance(x424, tuple):
            tuple_shapes = '('
            for item in x424:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x424: {}'.format(tuple_shapes))
        else:
            print('x424: {}'.format(x424))
        x425=self.conv2d135(x424)
        if x425 is None:
            print('x425: {}'.format(x425))
        elif isinstance(x425, torch.Tensor):
            print('x425: {}'.format(x425.shape))
        elif isinstance(x425, tuple):
            tuple_shapes = '('
            for item in x425:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x425: {}'.format(tuple_shapes))
        else:
            print('x425: {}'.format(x425))
        x426=self.sigmoid27(x425)
        if x426 is None:
            print('x426: {}'.format(x426))
        elif isinstance(x426, torch.Tensor):
            print('x426: {}'.format(x426.shape))
        elif isinstance(x426, tuple):
            tuple_shapes = '('
            for item in x426:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x426: {}'.format(tuple_shapes))
        else:
            print('x426: {}'.format(x426))
        x427=operator.mul(x426, x421)
        if x427 is None:
            print('x427: {}'.format(x427))
        elif isinstance(x427, torch.Tensor):
            print('x427: {}'.format(x427.shape))
        elif isinstance(x427, tuple):
            tuple_shapes = '('
            for item in x427:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x427: {}'.format(tuple_shapes))
        else:
            print('x427: {}'.format(x427))
        x428=self.conv2d136(x427)
        if x428 is None:
            print('x428: {}'.format(x428))
        elif isinstance(x428, torch.Tensor):
            print('x428: {}'.format(x428.shape))
        elif isinstance(x428, tuple):
            tuple_shapes = '('
            for item in x428:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x428: {}'.format(tuple_shapes))
        else:
            print('x428: {}'.format(x428))
        x429=self.batchnorm2d80(x428)
        if x429 is None:
            print('x429: {}'.format(x429))
        elif isinstance(x429, torch.Tensor):
            print('x429: {}'.format(x429.shape))
        elif isinstance(x429, tuple):
            tuple_shapes = '('
            for item in x429:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x429: {}'.format(tuple_shapes))
        else:
            print('x429: {}'.format(x429))
        x430=stochastic_depth(x429, 0.09818181818181819, 'row', False)
        if x430 is None:
            print('x430: {}'.format(x430))
        elif isinstance(x430, torch.Tensor):
            print('x430: {}'.format(x430.shape))
        elif isinstance(x430, tuple):
            tuple_shapes = '('
            for item in x430:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x430: {}'.format(tuple_shapes))
        else:
            print('x430: {}'.format(x430))
        x431=operator.add(x430, x415)
        if x431 is None:
            print('x431: {}'.format(x431))
        elif isinstance(x431, torch.Tensor):
            print('x431: {}'.format(x431.shape))
        elif isinstance(x431, tuple):
            tuple_shapes = '('
            for item in x431:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x431: {}'.format(tuple_shapes))
        else:
            print('x431: {}'.format(x431))
        x432=self.conv2d137(x431)
        if x432 is None:
            print('x432: {}'.format(x432))
        elif isinstance(x432, torch.Tensor):
            print('x432: {}'.format(x432.shape))
        elif isinstance(x432, tuple):
            tuple_shapes = '('
            for item in x432:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x432: {}'.format(tuple_shapes))
        else:
            print('x432: {}'.format(x432))
        x433=self.batchnorm2d81(x432)
        if x433 is None:
            print('x433: {}'.format(x433))
        elif isinstance(x433, torch.Tensor):
            print('x433: {}'.format(x433.shape))
        elif isinstance(x433, tuple):
            tuple_shapes = '('
            for item in x433:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x433: {}'.format(tuple_shapes))
        else:
            print('x433: {}'.format(x433))
        x434=self.silu81(x433)
        if x434 is None:
            print('x434: {}'.format(x434))
        elif isinstance(x434, torch.Tensor):
            print('x434: {}'.format(x434.shape))
        elif isinstance(x434, tuple):
            tuple_shapes = '('
            for item in x434:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x434: {}'.format(tuple_shapes))
        else:
            print('x434: {}'.format(x434))
        x435=self.conv2d138(x434)
        if x435 is None:
            print('x435: {}'.format(x435))
        elif isinstance(x435, torch.Tensor):
            print('x435: {}'.format(x435.shape))
        elif isinstance(x435, tuple):
            tuple_shapes = '('
            for item in x435:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x435: {}'.format(tuple_shapes))
        else:
            print('x435: {}'.format(x435))
        x436=self.batchnorm2d82(x435)
        if x436 is None:
            print('x436: {}'.format(x436))
        elif isinstance(x436, torch.Tensor):
            print('x436: {}'.format(x436.shape))
        elif isinstance(x436, tuple):
            tuple_shapes = '('
            for item in x436:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x436: {}'.format(tuple_shapes))
        else:
            print('x436: {}'.format(x436))
        x437=self.silu82(x436)
        if x437 is None:
            print('x437: {}'.format(x437))
        elif isinstance(x437, torch.Tensor):
            print('x437: {}'.format(x437.shape))
        elif isinstance(x437, tuple):
            tuple_shapes = '('
            for item in x437:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x437: {}'.format(tuple_shapes))
        else:
            print('x437: {}'.format(x437))
        x438=self.adaptiveavgpool2d28(x437)
        if x438 is None:
            print('x438: {}'.format(x438))
        elif isinstance(x438, torch.Tensor):
            print('x438: {}'.format(x438.shape))
        elif isinstance(x438, tuple):
            tuple_shapes = '('
            for item in x438:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x438: {}'.format(tuple_shapes))
        else:
            print('x438: {}'.format(x438))
        x439=self.conv2d139(x438)
        if x439 is None:
            print('x439: {}'.format(x439))
        elif isinstance(x439, torch.Tensor):
            print('x439: {}'.format(x439.shape))
        elif isinstance(x439, tuple):
            tuple_shapes = '('
            for item in x439:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x439: {}'.format(tuple_shapes))
        else:
            print('x439: {}'.format(x439))
        x440=self.silu83(x439)
        if x440 is None:
            print('x440: {}'.format(x440))
        elif isinstance(x440, torch.Tensor):
            print('x440: {}'.format(x440.shape))
        elif isinstance(x440, tuple):
            tuple_shapes = '('
            for item in x440:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x440: {}'.format(tuple_shapes))
        else:
            print('x440: {}'.format(x440))
        x441=self.conv2d140(x440)
        if x441 is None:
            print('x441: {}'.format(x441))
        elif isinstance(x441, torch.Tensor):
            print('x441: {}'.format(x441.shape))
        elif isinstance(x441, tuple):
            tuple_shapes = '('
            for item in x441:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x441: {}'.format(tuple_shapes))
        else:
            print('x441: {}'.format(x441))
        x442=self.sigmoid28(x441)
        if x442 is None:
            print('x442: {}'.format(x442))
        elif isinstance(x442, torch.Tensor):
            print('x442: {}'.format(x442.shape))
        elif isinstance(x442, tuple):
            tuple_shapes = '('
            for item in x442:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x442: {}'.format(tuple_shapes))
        else:
            print('x442: {}'.format(x442))
        x443=operator.mul(x442, x437)
        if x443 is None:
            print('x443: {}'.format(x443))
        elif isinstance(x443, torch.Tensor):
            print('x443: {}'.format(x443.shape))
        elif isinstance(x443, tuple):
            tuple_shapes = '('
            for item in x443:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x443: {}'.format(tuple_shapes))
        else:
            print('x443: {}'.format(x443))
        x444=self.conv2d141(x443)
        if x444 is None:
            print('x444: {}'.format(x444))
        elif isinstance(x444, torch.Tensor):
            print('x444: {}'.format(x444.shape))
        elif isinstance(x444, tuple):
            tuple_shapes = '('
            for item in x444:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x444: {}'.format(tuple_shapes))
        else:
            print('x444: {}'.format(x444))
        x445=self.batchnorm2d83(x444)
        if x445 is None:
            print('x445: {}'.format(x445))
        elif isinstance(x445, torch.Tensor):
            print('x445: {}'.format(x445.shape))
        elif isinstance(x445, tuple):
            tuple_shapes = '('
            for item in x445:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x445: {}'.format(tuple_shapes))
        else:
            print('x445: {}'.format(x445))
        x446=self.conv2d142(x445)
        if x446 is None:
            print('x446: {}'.format(x446))
        elif isinstance(x446, torch.Tensor):
            print('x446: {}'.format(x446.shape))
        elif isinstance(x446, tuple):
            tuple_shapes = '('
            for item in x446:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x446: {}'.format(tuple_shapes))
        else:
            print('x446: {}'.format(x446))
        x447=self.batchnorm2d84(x446)
        if x447 is None:
            print('x447: {}'.format(x447))
        elif isinstance(x447, torch.Tensor):
            print('x447: {}'.format(x447.shape))
        elif isinstance(x447, tuple):
            tuple_shapes = '('
            for item in x447:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x447: {}'.format(tuple_shapes))
        else:
            print('x447: {}'.format(x447))
        x448=self.silu84(x447)
        if x448 is None:
            print('x448: {}'.format(x448))
        elif isinstance(x448, torch.Tensor):
            print('x448: {}'.format(x448.shape))
        elif isinstance(x448, tuple):
            tuple_shapes = '('
            for item in x448:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x448: {}'.format(tuple_shapes))
        else:
            print('x448: {}'.format(x448))
        x449=self.conv2d143(x448)
        if x449 is None:
            print('x449: {}'.format(x449))
        elif isinstance(x449, torch.Tensor):
            print('x449: {}'.format(x449.shape))
        elif isinstance(x449, tuple):
            tuple_shapes = '('
            for item in x449:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x449: {}'.format(tuple_shapes))
        else:
            print('x449: {}'.format(x449))
        x450=self.batchnorm2d85(x449)
        if x450 is None:
            print('x450: {}'.format(x450))
        elif isinstance(x450, torch.Tensor):
            print('x450: {}'.format(x450.shape))
        elif isinstance(x450, tuple):
            tuple_shapes = '('
            for item in x450:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x450: {}'.format(tuple_shapes))
        else:
            print('x450: {}'.format(x450))
        x451=self.silu85(x450)
        if x451 is None:
            print('x451: {}'.format(x451))
        elif isinstance(x451, torch.Tensor):
            print('x451: {}'.format(x451.shape))
        elif isinstance(x451, tuple):
            tuple_shapes = '('
            for item in x451:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x451: {}'.format(tuple_shapes))
        else:
            print('x451: {}'.format(x451))
        x452=self.adaptiveavgpool2d29(x451)
        if x452 is None:
            print('x452: {}'.format(x452))
        elif isinstance(x452, torch.Tensor):
            print('x452: {}'.format(x452.shape))
        elif isinstance(x452, tuple):
            tuple_shapes = '('
            for item in x452:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x452: {}'.format(tuple_shapes))
        else:
            print('x452: {}'.format(x452))
        x453=self.conv2d144(x452)
        if x453 is None:
            print('x453: {}'.format(x453))
        elif isinstance(x453, torch.Tensor):
            print('x453: {}'.format(x453.shape))
        elif isinstance(x453, tuple):
            tuple_shapes = '('
            for item in x453:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x453: {}'.format(tuple_shapes))
        else:
            print('x453: {}'.format(x453))
        x454=self.silu86(x453)
        if x454 is None:
            print('x454: {}'.format(x454))
        elif isinstance(x454, torch.Tensor):
            print('x454: {}'.format(x454.shape))
        elif isinstance(x454, tuple):
            tuple_shapes = '('
            for item in x454:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x454: {}'.format(tuple_shapes))
        else:
            print('x454: {}'.format(x454))
        x455=self.conv2d145(x454)
        if x455 is None:
            print('x455: {}'.format(x455))
        elif isinstance(x455, torch.Tensor):
            print('x455: {}'.format(x455.shape))
        elif isinstance(x455, tuple):
            tuple_shapes = '('
            for item in x455:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x455: {}'.format(tuple_shapes))
        else:
            print('x455: {}'.format(x455))
        x456=self.sigmoid29(x455)
        if x456 is None:
            print('x456: {}'.format(x456))
        elif isinstance(x456, torch.Tensor):
            print('x456: {}'.format(x456.shape))
        elif isinstance(x456, tuple):
            tuple_shapes = '('
            for item in x456:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x456: {}'.format(tuple_shapes))
        else:
            print('x456: {}'.format(x456))
        x457=operator.mul(x456, x451)
        if x457 is None:
            print('x457: {}'.format(x457))
        elif isinstance(x457, torch.Tensor):
            print('x457: {}'.format(x457.shape))
        elif isinstance(x457, tuple):
            tuple_shapes = '('
            for item in x457:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x457: {}'.format(tuple_shapes))
        else:
            print('x457: {}'.format(x457))
        x458=self.conv2d146(x457)
        if x458 is None:
            print('x458: {}'.format(x458))
        elif isinstance(x458, torch.Tensor):
            print('x458: {}'.format(x458.shape))
        elif isinstance(x458, tuple):
            tuple_shapes = '('
            for item in x458:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x458: {}'.format(tuple_shapes))
        else:
            print('x458: {}'.format(x458))
        x459=self.batchnorm2d86(x458)
        if x459 is None:
            print('x459: {}'.format(x459))
        elif isinstance(x459, torch.Tensor):
            print('x459: {}'.format(x459.shape))
        elif isinstance(x459, tuple):
            tuple_shapes = '('
            for item in x459:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x459: {}'.format(tuple_shapes))
        else:
            print('x459: {}'.format(x459))
        x460=stochastic_depth(x459, 0.10545454545454547, 'row', False)
        if x460 is None:
            print('x460: {}'.format(x460))
        elif isinstance(x460, torch.Tensor):
            print('x460: {}'.format(x460.shape))
        elif isinstance(x460, tuple):
            tuple_shapes = '('
            for item in x460:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x460: {}'.format(tuple_shapes))
        else:
            print('x460: {}'.format(x460))
        x461=operator.add(x460, x445)
        if x461 is None:
            print('x461: {}'.format(x461))
        elif isinstance(x461, torch.Tensor):
            print('x461: {}'.format(x461.shape))
        elif isinstance(x461, tuple):
            tuple_shapes = '('
            for item in x461:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x461: {}'.format(tuple_shapes))
        else:
            print('x461: {}'.format(x461))
        x462=self.conv2d147(x461)
        if x462 is None:
            print('x462: {}'.format(x462))
        elif isinstance(x462, torch.Tensor):
            print('x462: {}'.format(x462.shape))
        elif isinstance(x462, tuple):
            tuple_shapes = '('
            for item in x462:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x462: {}'.format(tuple_shapes))
        else:
            print('x462: {}'.format(x462))
        x463=self.batchnorm2d87(x462)
        if x463 is None:
            print('x463: {}'.format(x463))
        elif isinstance(x463, torch.Tensor):
            print('x463: {}'.format(x463.shape))
        elif isinstance(x463, tuple):
            tuple_shapes = '('
            for item in x463:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x463: {}'.format(tuple_shapes))
        else:
            print('x463: {}'.format(x463))
        x464=self.silu87(x463)
        if x464 is None:
            print('x464: {}'.format(x464))
        elif isinstance(x464, torch.Tensor):
            print('x464: {}'.format(x464.shape))
        elif isinstance(x464, tuple):
            tuple_shapes = '('
            for item in x464:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x464: {}'.format(tuple_shapes))
        else:
            print('x464: {}'.format(x464))
        x465=self.conv2d148(x464)
        if x465 is None:
            print('x465: {}'.format(x465))
        elif isinstance(x465, torch.Tensor):
            print('x465: {}'.format(x465.shape))
        elif isinstance(x465, tuple):
            tuple_shapes = '('
            for item in x465:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x465: {}'.format(tuple_shapes))
        else:
            print('x465: {}'.format(x465))
        x466=self.batchnorm2d88(x465)
        if x466 is None:
            print('x466: {}'.format(x466))
        elif isinstance(x466, torch.Tensor):
            print('x466: {}'.format(x466.shape))
        elif isinstance(x466, tuple):
            tuple_shapes = '('
            for item in x466:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x466: {}'.format(tuple_shapes))
        else:
            print('x466: {}'.format(x466))
        x467=self.silu88(x466)
        if x467 is None:
            print('x467: {}'.format(x467))
        elif isinstance(x467, torch.Tensor):
            print('x467: {}'.format(x467.shape))
        elif isinstance(x467, tuple):
            tuple_shapes = '('
            for item in x467:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x467: {}'.format(tuple_shapes))
        else:
            print('x467: {}'.format(x467))
        x468=self.adaptiveavgpool2d30(x467)
        if x468 is None:
            print('x468: {}'.format(x468))
        elif isinstance(x468, torch.Tensor):
            print('x468: {}'.format(x468.shape))
        elif isinstance(x468, tuple):
            tuple_shapes = '('
            for item in x468:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x468: {}'.format(tuple_shapes))
        else:
            print('x468: {}'.format(x468))
        x469=self.conv2d149(x468)
        if x469 is None:
            print('x469: {}'.format(x469))
        elif isinstance(x469, torch.Tensor):
            print('x469: {}'.format(x469.shape))
        elif isinstance(x469, tuple):
            tuple_shapes = '('
            for item in x469:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x469: {}'.format(tuple_shapes))
        else:
            print('x469: {}'.format(x469))
        x470=self.silu89(x469)
        if x470 is None:
            print('x470: {}'.format(x470))
        elif isinstance(x470, torch.Tensor):
            print('x470: {}'.format(x470.shape))
        elif isinstance(x470, tuple):
            tuple_shapes = '('
            for item in x470:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x470: {}'.format(tuple_shapes))
        else:
            print('x470: {}'.format(x470))
        x471=self.conv2d150(x470)
        if x471 is None:
            print('x471: {}'.format(x471))
        elif isinstance(x471, torch.Tensor):
            print('x471: {}'.format(x471.shape))
        elif isinstance(x471, tuple):
            tuple_shapes = '('
            for item in x471:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x471: {}'.format(tuple_shapes))
        else:
            print('x471: {}'.format(x471))
        x472=self.sigmoid30(x471)
        if x472 is None:
            print('x472: {}'.format(x472))
        elif isinstance(x472, torch.Tensor):
            print('x472: {}'.format(x472.shape))
        elif isinstance(x472, tuple):
            tuple_shapes = '('
            for item in x472:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x472: {}'.format(tuple_shapes))
        else:
            print('x472: {}'.format(x472))
        x473=operator.mul(x472, x467)
        if x473 is None:
            print('x473: {}'.format(x473))
        elif isinstance(x473, torch.Tensor):
            print('x473: {}'.format(x473.shape))
        elif isinstance(x473, tuple):
            tuple_shapes = '('
            for item in x473:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x473: {}'.format(tuple_shapes))
        else:
            print('x473: {}'.format(x473))
        x474=self.conv2d151(x473)
        if x474 is None:
            print('x474: {}'.format(x474))
        elif isinstance(x474, torch.Tensor):
            print('x474: {}'.format(x474.shape))
        elif isinstance(x474, tuple):
            tuple_shapes = '('
            for item in x474:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x474: {}'.format(tuple_shapes))
        else:
            print('x474: {}'.format(x474))
        x475=self.batchnorm2d89(x474)
        if x475 is None:
            print('x475: {}'.format(x475))
        elif isinstance(x475, torch.Tensor):
            print('x475: {}'.format(x475.shape))
        elif isinstance(x475, tuple):
            tuple_shapes = '('
            for item in x475:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x475: {}'.format(tuple_shapes))
        else:
            print('x475: {}'.format(x475))
        x476=stochastic_depth(x475, 0.10909090909090909, 'row', False)
        if x476 is None:
            print('x476: {}'.format(x476))
        elif isinstance(x476, torch.Tensor):
            print('x476: {}'.format(x476.shape))
        elif isinstance(x476, tuple):
            tuple_shapes = '('
            for item in x476:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x476: {}'.format(tuple_shapes))
        else:
            print('x476: {}'.format(x476))
        x477=operator.add(x476, x461)
        if x477 is None:
            print('x477: {}'.format(x477))
        elif isinstance(x477, torch.Tensor):
            print('x477: {}'.format(x477.shape))
        elif isinstance(x477, tuple):
            tuple_shapes = '('
            for item in x477:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x477: {}'.format(tuple_shapes))
        else:
            print('x477: {}'.format(x477))
        x478=self.conv2d152(x477)
        if x478 is None:
            print('x478: {}'.format(x478))
        elif isinstance(x478, torch.Tensor):
            print('x478: {}'.format(x478.shape))
        elif isinstance(x478, tuple):
            tuple_shapes = '('
            for item in x478:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x478: {}'.format(tuple_shapes))
        else:
            print('x478: {}'.format(x478))
        x479=self.batchnorm2d90(x478)
        if x479 is None:
            print('x479: {}'.format(x479))
        elif isinstance(x479, torch.Tensor):
            print('x479: {}'.format(x479.shape))
        elif isinstance(x479, tuple):
            tuple_shapes = '('
            for item in x479:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x479: {}'.format(tuple_shapes))
        else:
            print('x479: {}'.format(x479))
        x480=self.silu90(x479)
        if x480 is None:
            print('x480: {}'.format(x480))
        elif isinstance(x480, torch.Tensor):
            print('x480: {}'.format(x480.shape))
        elif isinstance(x480, tuple):
            tuple_shapes = '('
            for item in x480:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x480: {}'.format(tuple_shapes))
        else:
            print('x480: {}'.format(x480))
        x481=self.conv2d153(x480)
        if x481 is None:
            print('x481: {}'.format(x481))
        elif isinstance(x481, torch.Tensor):
            print('x481: {}'.format(x481.shape))
        elif isinstance(x481, tuple):
            tuple_shapes = '('
            for item in x481:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x481: {}'.format(tuple_shapes))
        else:
            print('x481: {}'.format(x481))
        x482=self.batchnorm2d91(x481)
        if x482 is None:
            print('x482: {}'.format(x482))
        elif isinstance(x482, torch.Tensor):
            print('x482: {}'.format(x482.shape))
        elif isinstance(x482, tuple):
            tuple_shapes = '('
            for item in x482:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x482: {}'.format(tuple_shapes))
        else:
            print('x482: {}'.format(x482))
        x483=self.silu91(x482)
        if x483 is None:
            print('x483: {}'.format(x483))
        elif isinstance(x483, torch.Tensor):
            print('x483: {}'.format(x483.shape))
        elif isinstance(x483, tuple):
            tuple_shapes = '('
            for item in x483:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x483: {}'.format(tuple_shapes))
        else:
            print('x483: {}'.format(x483))
        x484=self.adaptiveavgpool2d31(x483)
        if x484 is None:
            print('x484: {}'.format(x484))
        elif isinstance(x484, torch.Tensor):
            print('x484: {}'.format(x484.shape))
        elif isinstance(x484, tuple):
            tuple_shapes = '('
            for item in x484:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x484: {}'.format(tuple_shapes))
        else:
            print('x484: {}'.format(x484))
        x485=self.conv2d154(x484)
        if x485 is None:
            print('x485: {}'.format(x485))
        elif isinstance(x485, torch.Tensor):
            print('x485: {}'.format(x485.shape))
        elif isinstance(x485, tuple):
            tuple_shapes = '('
            for item in x485:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x485: {}'.format(tuple_shapes))
        else:
            print('x485: {}'.format(x485))
        x486=self.silu92(x485)
        if x486 is None:
            print('x486: {}'.format(x486))
        elif isinstance(x486, torch.Tensor):
            print('x486: {}'.format(x486.shape))
        elif isinstance(x486, tuple):
            tuple_shapes = '('
            for item in x486:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x486: {}'.format(tuple_shapes))
        else:
            print('x486: {}'.format(x486))
        x487=self.conv2d155(x486)
        if x487 is None:
            print('x487: {}'.format(x487))
        elif isinstance(x487, torch.Tensor):
            print('x487: {}'.format(x487.shape))
        elif isinstance(x487, tuple):
            tuple_shapes = '('
            for item in x487:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x487: {}'.format(tuple_shapes))
        else:
            print('x487: {}'.format(x487))
        x488=self.sigmoid31(x487)
        if x488 is None:
            print('x488: {}'.format(x488))
        elif isinstance(x488, torch.Tensor):
            print('x488: {}'.format(x488.shape))
        elif isinstance(x488, tuple):
            tuple_shapes = '('
            for item in x488:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x488: {}'.format(tuple_shapes))
        else:
            print('x488: {}'.format(x488))
        x489=operator.mul(x488, x483)
        if x489 is None:
            print('x489: {}'.format(x489))
        elif isinstance(x489, torch.Tensor):
            print('x489: {}'.format(x489.shape))
        elif isinstance(x489, tuple):
            tuple_shapes = '('
            for item in x489:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x489: {}'.format(tuple_shapes))
        else:
            print('x489: {}'.format(x489))
        x490=self.conv2d156(x489)
        if x490 is None:
            print('x490: {}'.format(x490))
        elif isinstance(x490, torch.Tensor):
            print('x490: {}'.format(x490.shape))
        elif isinstance(x490, tuple):
            tuple_shapes = '('
            for item in x490:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x490: {}'.format(tuple_shapes))
        else:
            print('x490: {}'.format(x490))
        x491=self.batchnorm2d92(x490)
        if x491 is None:
            print('x491: {}'.format(x491))
        elif isinstance(x491, torch.Tensor):
            print('x491: {}'.format(x491.shape))
        elif isinstance(x491, tuple):
            tuple_shapes = '('
            for item in x491:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x491: {}'.format(tuple_shapes))
        else:
            print('x491: {}'.format(x491))
        x492=stochastic_depth(x491, 0.11272727272727273, 'row', False)
        if x492 is None:
            print('x492: {}'.format(x492))
        elif isinstance(x492, torch.Tensor):
            print('x492: {}'.format(x492.shape))
        elif isinstance(x492, tuple):
            tuple_shapes = '('
            for item in x492:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x492: {}'.format(tuple_shapes))
        else:
            print('x492: {}'.format(x492))
        x493=operator.add(x492, x477)
        if x493 is None:
            print('x493: {}'.format(x493))
        elif isinstance(x493, torch.Tensor):
            print('x493: {}'.format(x493.shape))
        elif isinstance(x493, tuple):
            tuple_shapes = '('
            for item in x493:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x493: {}'.format(tuple_shapes))
        else:
            print('x493: {}'.format(x493))
        x494=self.conv2d157(x493)
        if x494 is None:
            print('x494: {}'.format(x494))
        elif isinstance(x494, torch.Tensor):
            print('x494: {}'.format(x494.shape))
        elif isinstance(x494, tuple):
            tuple_shapes = '('
            for item in x494:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x494: {}'.format(tuple_shapes))
        else:
            print('x494: {}'.format(x494))
        x495=self.batchnorm2d93(x494)
        if x495 is None:
            print('x495: {}'.format(x495))
        elif isinstance(x495, torch.Tensor):
            print('x495: {}'.format(x495.shape))
        elif isinstance(x495, tuple):
            tuple_shapes = '('
            for item in x495:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x495: {}'.format(tuple_shapes))
        else:
            print('x495: {}'.format(x495))
        x496=self.silu93(x495)
        if x496 is None:
            print('x496: {}'.format(x496))
        elif isinstance(x496, torch.Tensor):
            print('x496: {}'.format(x496.shape))
        elif isinstance(x496, tuple):
            tuple_shapes = '('
            for item in x496:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x496: {}'.format(tuple_shapes))
        else:
            print('x496: {}'.format(x496))
        x497=self.conv2d158(x496)
        if x497 is None:
            print('x497: {}'.format(x497))
        elif isinstance(x497, torch.Tensor):
            print('x497: {}'.format(x497.shape))
        elif isinstance(x497, tuple):
            tuple_shapes = '('
            for item in x497:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x497: {}'.format(tuple_shapes))
        else:
            print('x497: {}'.format(x497))
        x498=self.batchnorm2d94(x497)
        if x498 is None:
            print('x498: {}'.format(x498))
        elif isinstance(x498, torch.Tensor):
            print('x498: {}'.format(x498.shape))
        elif isinstance(x498, tuple):
            tuple_shapes = '('
            for item in x498:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x498: {}'.format(tuple_shapes))
        else:
            print('x498: {}'.format(x498))
        x499=self.silu94(x498)
        if x499 is None:
            print('x499: {}'.format(x499))
        elif isinstance(x499, torch.Tensor):
            print('x499: {}'.format(x499.shape))
        elif isinstance(x499, tuple):
            tuple_shapes = '('
            for item in x499:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x499: {}'.format(tuple_shapes))
        else:
            print('x499: {}'.format(x499))
        x500=self.adaptiveavgpool2d32(x499)
        if x500 is None:
            print('x500: {}'.format(x500))
        elif isinstance(x500, torch.Tensor):
            print('x500: {}'.format(x500.shape))
        elif isinstance(x500, tuple):
            tuple_shapes = '('
            for item in x500:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x500: {}'.format(tuple_shapes))
        else:
            print('x500: {}'.format(x500))
        x501=self.conv2d159(x500)
        if x501 is None:
            print('x501: {}'.format(x501))
        elif isinstance(x501, torch.Tensor):
            print('x501: {}'.format(x501.shape))
        elif isinstance(x501, tuple):
            tuple_shapes = '('
            for item in x501:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x501: {}'.format(tuple_shapes))
        else:
            print('x501: {}'.format(x501))
        x502=self.silu95(x501)
        if x502 is None:
            print('x502: {}'.format(x502))
        elif isinstance(x502, torch.Tensor):
            print('x502: {}'.format(x502.shape))
        elif isinstance(x502, tuple):
            tuple_shapes = '('
            for item in x502:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x502: {}'.format(tuple_shapes))
        else:
            print('x502: {}'.format(x502))
        x503=self.conv2d160(x502)
        if x503 is None:
            print('x503: {}'.format(x503))
        elif isinstance(x503, torch.Tensor):
            print('x503: {}'.format(x503.shape))
        elif isinstance(x503, tuple):
            tuple_shapes = '('
            for item in x503:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x503: {}'.format(tuple_shapes))
        else:
            print('x503: {}'.format(x503))
        x504=self.sigmoid32(x503)
        if x504 is None:
            print('x504: {}'.format(x504))
        elif isinstance(x504, torch.Tensor):
            print('x504: {}'.format(x504.shape))
        elif isinstance(x504, tuple):
            tuple_shapes = '('
            for item in x504:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x504: {}'.format(tuple_shapes))
        else:
            print('x504: {}'.format(x504))
        x505=operator.mul(x504, x499)
        if x505 is None:
            print('x505: {}'.format(x505))
        elif isinstance(x505, torch.Tensor):
            print('x505: {}'.format(x505.shape))
        elif isinstance(x505, tuple):
            tuple_shapes = '('
            for item in x505:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x505: {}'.format(tuple_shapes))
        else:
            print('x505: {}'.format(x505))
        x506=self.conv2d161(x505)
        if x506 is None:
            print('x506: {}'.format(x506))
        elif isinstance(x506, torch.Tensor):
            print('x506: {}'.format(x506.shape))
        elif isinstance(x506, tuple):
            tuple_shapes = '('
            for item in x506:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x506: {}'.format(tuple_shapes))
        else:
            print('x506: {}'.format(x506))
        x507=self.batchnorm2d95(x506)
        if x507 is None:
            print('x507: {}'.format(x507))
        elif isinstance(x507, torch.Tensor):
            print('x507: {}'.format(x507.shape))
        elif isinstance(x507, tuple):
            tuple_shapes = '('
            for item in x507:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x507: {}'.format(tuple_shapes))
        else:
            print('x507: {}'.format(x507))
        x508=stochastic_depth(x507, 0.11636363636363636, 'row', False)
        if x508 is None:
            print('x508: {}'.format(x508))
        elif isinstance(x508, torch.Tensor):
            print('x508: {}'.format(x508.shape))
        elif isinstance(x508, tuple):
            tuple_shapes = '('
            for item in x508:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x508: {}'.format(tuple_shapes))
        else:
            print('x508: {}'.format(x508))
        x509=operator.add(x508, x493)
        if x509 is None:
            print('x509: {}'.format(x509))
        elif isinstance(x509, torch.Tensor):
            print('x509: {}'.format(x509.shape))
        elif isinstance(x509, tuple):
            tuple_shapes = '('
            for item in x509:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x509: {}'.format(tuple_shapes))
        else:
            print('x509: {}'.format(x509))
        x510=self.conv2d162(x509)
        if x510 is None:
            print('x510: {}'.format(x510))
        elif isinstance(x510, torch.Tensor):
            print('x510: {}'.format(x510.shape))
        elif isinstance(x510, tuple):
            tuple_shapes = '('
            for item in x510:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x510: {}'.format(tuple_shapes))
        else:
            print('x510: {}'.format(x510))
        x511=self.batchnorm2d96(x510)
        if x511 is None:
            print('x511: {}'.format(x511))
        elif isinstance(x511, torch.Tensor):
            print('x511: {}'.format(x511.shape))
        elif isinstance(x511, tuple):
            tuple_shapes = '('
            for item in x511:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x511: {}'.format(tuple_shapes))
        else:
            print('x511: {}'.format(x511))
        x512=self.silu96(x511)
        if x512 is None:
            print('x512: {}'.format(x512))
        elif isinstance(x512, torch.Tensor):
            print('x512: {}'.format(x512.shape))
        elif isinstance(x512, tuple):
            tuple_shapes = '('
            for item in x512:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x512: {}'.format(tuple_shapes))
        else:
            print('x512: {}'.format(x512))
        x513=self.conv2d163(x512)
        if x513 is None:
            print('x513: {}'.format(x513))
        elif isinstance(x513, torch.Tensor):
            print('x513: {}'.format(x513.shape))
        elif isinstance(x513, tuple):
            tuple_shapes = '('
            for item in x513:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x513: {}'.format(tuple_shapes))
        else:
            print('x513: {}'.format(x513))
        x514=self.batchnorm2d97(x513)
        if x514 is None:
            print('x514: {}'.format(x514))
        elif isinstance(x514, torch.Tensor):
            print('x514: {}'.format(x514.shape))
        elif isinstance(x514, tuple):
            tuple_shapes = '('
            for item in x514:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x514: {}'.format(tuple_shapes))
        else:
            print('x514: {}'.format(x514))
        x515=self.silu97(x514)
        if x515 is None:
            print('x515: {}'.format(x515))
        elif isinstance(x515, torch.Tensor):
            print('x515: {}'.format(x515.shape))
        elif isinstance(x515, tuple):
            tuple_shapes = '('
            for item in x515:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x515: {}'.format(tuple_shapes))
        else:
            print('x515: {}'.format(x515))
        x516=self.adaptiveavgpool2d33(x515)
        if x516 is None:
            print('x516: {}'.format(x516))
        elif isinstance(x516, torch.Tensor):
            print('x516: {}'.format(x516.shape))
        elif isinstance(x516, tuple):
            tuple_shapes = '('
            for item in x516:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x516: {}'.format(tuple_shapes))
        else:
            print('x516: {}'.format(x516))
        x517=self.conv2d164(x516)
        if x517 is None:
            print('x517: {}'.format(x517))
        elif isinstance(x517, torch.Tensor):
            print('x517: {}'.format(x517.shape))
        elif isinstance(x517, tuple):
            tuple_shapes = '('
            for item in x517:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x517: {}'.format(tuple_shapes))
        else:
            print('x517: {}'.format(x517))
        x518=self.silu98(x517)
        if x518 is None:
            print('x518: {}'.format(x518))
        elif isinstance(x518, torch.Tensor):
            print('x518: {}'.format(x518.shape))
        elif isinstance(x518, tuple):
            tuple_shapes = '('
            for item in x518:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x518: {}'.format(tuple_shapes))
        else:
            print('x518: {}'.format(x518))
        x519=self.conv2d165(x518)
        if x519 is None:
            print('x519: {}'.format(x519))
        elif isinstance(x519, torch.Tensor):
            print('x519: {}'.format(x519.shape))
        elif isinstance(x519, tuple):
            tuple_shapes = '('
            for item in x519:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x519: {}'.format(tuple_shapes))
        else:
            print('x519: {}'.format(x519))
        x520=self.sigmoid33(x519)
        if x520 is None:
            print('x520: {}'.format(x520))
        elif isinstance(x520, torch.Tensor):
            print('x520: {}'.format(x520.shape))
        elif isinstance(x520, tuple):
            tuple_shapes = '('
            for item in x520:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x520: {}'.format(tuple_shapes))
        else:
            print('x520: {}'.format(x520))
        x521=operator.mul(x520, x515)
        if x521 is None:
            print('x521: {}'.format(x521))
        elif isinstance(x521, torch.Tensor):
            print('x521: {}'.format(x521.shape))
        elif isinstance(x521, tuple):
            tuple_shapes = '('
            for item in x521:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x521: {}'.format(tuple_shapes))
        else:
            print('x521: {}'.format(x521))
        x522=self.conv2d166(x521)
        if x522 is None:
            print('x522: {}'.format(x522))
        elif isinstance(x522, torch.Tensor):
            print('x522: {}'.format(x522.shape))
        elif isinstance(x522, tuple):
            tuple_shapes = '('
            for item in x522:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x522: {}'.format(tuple_shapes))
        else:
            print('x522: {}'.format(x522))
        x523=self.batchnorm2d98(x522)
        if x523 is None:
            print('x523: {}'.format(x523))
        elif isinstance(x523, torch.Tensor):
            print('x523: {}'.format(x523.shape))
        elif isinstance(x523, tuple):
            tuple_shapes = '('
            for item in x523:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x523: {}'.format(tuple_shapes))
        else:
            print('x523: {}'.format(x523))
        x524=stochastic_depth(x523, 0.12000000000000001, 'row', False)
        if x524 is None:
            print('x524: {}'.format(x524))
        elif isinstance(x524, torch.Tensor):
            print('x524: {}'.format(x524.shape))
        elif isinstance(x524, tuple):
            tuple_shapes = '('
            for item in x524:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x524: {}'.format(tuple_shapes))
        else:
            print('x524: {}'.format(x524))
        x525=operator.add(x524, x509)
        if x525 is None:
            print('x525: {}'.format(x525))
        elif isinstance(x525, torch.Tensor):
            print('x525: {}'.format(x525.shape))
        elif isinstance(x525, tuple):
            tuple_shapes = '('
            for item in x525:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x525: {}'.format(tuple_shapes))
        else:
            print('x525: {}'.format(x525))
        x526=self.conv2d167(x525)
        if x526 is None:
            print('x526: {}'.format(x526))
        elif isinstance(x526, torch.Tensor):
            print('x526: {}'.format(x526.shape))
        elif isinstance(x526, tuple):
            tuple_shapes = '('
            for item in x526:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x526: {}'.format(tuple_shapes))
        else:
            print('x526: {}'.format(x526))
        x527=self.batchnorm2d99(x526)
        if x527 is None:
            print('x527: {}'.format(x527))
        elif isinstance(x527, torch.Tensor):
            print('x527: {}'.format(x527.shape))
        elif isinstance(x527, tuple):
            tuple_shapes = '('
            for item in x527:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x527: {}'.format(tuple_shapes))
        else:
            print('x527: {}'.format(x527))
        x528=self.silu99(x527)
        if x528 is None:
            print('x528: {}'.format(x528))
        elif isinstance(x528, torch.Tensor):
            print('x528: {}'.format(x528.shape))
        elif isinstance(x528, tuple):
            tuple_shapes = '('
            for item in x528:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x528: {}'.format(tuple_shapes))
        else:
            print('x528: {}'.format(x528))
        x529=self.conv2d168(x528)
        if x529 is None:
            print('x529: {}'.format(x529))
        elif isinstance(x529, torch.Tensor):
            print('x529: {}'.format(x529.shape))
        elif isinstance(x529, tuple):
            tuple_shapes = '('
            for item in x529:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x529: {}'.format(tuple_shapes))
        else:
            print('x529: {}'.format(x529))
        x530=self.batchnorm2d100(x529)
        if x530 is None:
            print('x530: {}'.format(x530))
        elif isinstance(x530, torch.Tensor):
            print('x530: {}'.format(x530.shape))
        elif isinstance(x530, tuple):
            tuple_shapes = '('
            for item in x530:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x530: {}'.format(tuple_shapes))
        else:
            print('x530: {}'.format(x530))
        x531=self.silu100(x530)
        if x531 is None:
            print('x531: {}'.format(x531))
        elif isinstance(x531, torch.Tensor):
            print('x531: {}'.format(x531.shape))
        elif isinstance(x531, tuple):
            tuple_shapes = '('
            for item in x531:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x531: {}'.format(tuple_shapes))
        else:
            print('x531: {}'.format(x531))
        x532=self.adaptiveavgpool2d34(x531)
        if x532 is None:
            print('x532: {}'.format(x532))
        elif isinstance(x532, torch.Tensor):
            print('x532: {}'.format(x532.shape))
        elif isinstance(x532, tuple):
            tuple_shapes = '('
            for item in x532:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x532: {}'.format(tuple_shapes))
        else:
            print('x532: {}'.format(x532))
        x533=self.conv2d169(x532)
        if x533 is None:
            print('x533: {}'.format(x533))
        elif isinstance(x533, torch.Tensor):
            print('x533: {}'.format(x533.shape))
        elif isinstance(x533, tuple):
            tuple_shapes = '('
            for item in x533:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x533: {}'.format(tuple_shapes))
        else:
            print('x533: {}'.format(x533))
        x534=self.silu101(x533)
        if x534 is None:
            print('x534: {}'.format(x534))
        elif isinstance(x534, torch.Tensor):
            print('x534: {}'.format(x534.shape))
        elif isinstance(x534, tuple):
            tuple_shapes = '('
            for item in x534:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x534: {}'.format(tuple_shapes))
        else:
            print('x534: {}'.format(x534))
        x535=self.conv2d170(x534)
        if x535 is None:
            print('x535: {}'.format(x535))
        elif isinstance(x535, torch.Tensor):
            print('x535: {}'.format(x535.shape))
        elif isinstance(x535, tuple):
            tuple_shapes = '('
            for item in x535:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x535: {}'.format(tuple_shapes))
        else:
            print('x535: {}'.format(x535))
        x536=self.sigmoid34(x535)
        if x536 is None:
            print('x536: {}'.format(x536))
        elif isinstance(x536, torch.Tensor):
            print('x536: {}'.format(x536.shape))
        elif isinstance(x536, tuple):
            tuple_shapes = '('
            for item in x536:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x536: {}'.format(tuple_shapes))
        else:
            print('x536: {}'.format(x536))
        x537=operator.mul(x536, x531)
        if x537 is None:
            print('x537: {}'.format(x537))
        elif isinstance(x537, torch.Tensor):
            print('x537: {}'.format(x537.shape))
        elif isinstance(x537, tuple):
            tuple_shapes = '('
            for item in x537:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x537: {}'.format(tuple_shapes))
        else:
            print('x537: {}'.format(x537))
        x538=self.conv2d171(x537)
        if x538 is None:
            print('x538: {}'.format(x538))
        elif isinstance(x538, torch.Tensor):
            print('x538: {}'.format(x538.shape))
        elif isinstance(x538, tuple):
            tuple_shapes = '('
            for item in x538:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x538: {}'.format(tuple_shapes))
        else:
            print('x538: {}'.format(x538))
        x539=self.batchnorm2d101(x538)
        if x539 is None:
            print('x539: {}'.format(x539))
        elif isinstance(x539, torch.Tensor):
            print('x539: {}'.format(x539.shape))
        elif isinstance(x539, tuple):
            tuple_shapes = '('
            for item in x539:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x539: {}'.format(tuple_shapes))
        else:
            print('x539: {}'.format(x539))
        x540=stochastic_depth(x539, 0.12363636363636366, 'row', False)
        if x540 is None:
            print('x540: {}'.format(x540))
        elif isinstance(x540, torch.Tensor):
            print('x540: {}'.format(x540.shape))
        elif isinstance(x540, tuple):
            tuple_shapes = '('
            for item in x540:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x540: {}'.format(tuple_shapes))
        else:
            print('x540: {}'.format(x540))
        x541=operator.add(x540, x525)
        if x541 is None:
            print('x541: {}'.format(x541))
        elif isinstance(x541, torch.Tensor):
            print('x541: {}'.format(x541.shape))
        elif isinstance(x541, tuple):
            tuple_shapes = '('
            for item in x541:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x541: {}'.format(tuple_shapes))
        else:
            print('x541: {}'.format(x541))
        x542=self.conv2d172(x541)
        if x542 is None:
            print('x542: {}'.format(x542))
        elif isinstance(x542, torch.Tensor):
            print('x542: {}'.format(x542.shape))
        elif isinstance(x542, tuple):
            tuple_shapes = '('
            for item in x542:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x542: {}'.format(tuple_shapes))
        else:
            print('x542: {}'.format(x542))
        x543=self.batchnorm2d102(x542)
        if x543 is None:
            print('x543: {}'.format(x543))
        elif isinstance(x543, torch.Tensor):
            print('x543: {}'.format(x543.shape))
        elif isinstance(x543, tuple):
            tuple_shapes = '('
            for item in x543:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x543: {}'.format(tuple_shapes))
        else:
            print('x543: {}'.format(x543))
        x544=self.silu102(x543)
        if x544 is None:
            print('x544: {}'.format(x544))
        elif isinstance(x544, torch.Tensor):
            print('x544: {}'.format(x544.shape))
        elif isinstance(x544, tuple):
            tuple_shapes = '('
            for item in x544:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x544: {}'.format(tuple_shapes))
        else:
            print('x544: {}'.format(x544))
        x545=self.conv2d173(x544)
        if x545 is None:
            print('x545: {}'.format(x545))
        elif isinstance(x545, torch.Tensor):
            print('x545: {}'.format(x545.shape))
        elif isinstance(x545, tuple):
            tuple_shapes = '('
            for item in x545:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x545: {}'.format(tuple_shapes))
        else:
            print('x545: {}'.format(x545))
        x546=self.batchnorm2d103(x545)
        if x546 is None:
            print('x546: {}'.format(x546))
        elif isinstance(x546, torch.Tensor):
            print('x546: {}'.format(x546.shape))
        elif isinstance(x546, tuple):
            tuple_shapes = '('
            for item in x546:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x546: {}'.format(tuple_shapes))
        else:
            print('x546: {}'.format(x546))
        x547=self.silu103(x546)
        if x547 is None:
            print('x547: {}'.format(x547))
        elif isinstance(x547, torch.Tensor):
            print('x547: {}'.format(x547.shape))
        elif isinstance(x547, tuple):
            tuple_shapes = '('
            for item in x547:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x547: {}'.format(tuple_shapes))
        else:
            print('x547: {}'.format(x547))
        x548=self.adaptiveavgpool2d35(x547)
        if x548 is None:
            print('x548: {}'.format(x548))
        elif isinstance(x548, torch.Tensor):
            print('x548: {}'.format(x548.shape))
        elif isinstance(x548, tuple):
            tuple_shapes = '('
            for item in x548:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x548: {}'.format(tuple_shapes))
        else:
            print('x548: {}'.format(x548))
        x549=self.conv2d174(x548)
        if x549 is None:
            print('x549: {}'.format(x549))
        elif isinstance(x549, torch.Tensor):
            print('x549: {}'.format(x549.shape))
        elif isinstance(x549, tuple):
            tuple_shapes = '('
            for item in x549:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x549: {}'.format(tuple_shapes))
        else:
            print('x549: {}'.format(x549))
        x550=self.silu104(x549)
        if x550 is None:
            print('x550: {}'.format(x550))
        elif isinstance(x550, torch.Tensor):
            print('x550: {}'.format(x550.shape))
        elif isinstance(x550, tuple):
            tuple_shapes = '('
            for item in x550:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x550: {}'.format(tuple_shapes))
        else:
            print('x550: {}'.format(x550))
        x551=self.conv2d175(x550)
        if x551 is None:
            print('x551: {}'.format(x551))
        elif isinstance(x551, torch.Tensor):
            print('x551: {}'.format(x551.shape))
        elif isinstance(x551, tuple):
            tuple_shapes = '('
            for item in x551:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x551: {}'.format(tuple_shapes))
        else:
            print('x551: {}'.format(x551))
        x552=self.sigmoid35(x551)
        if x552 is None:
            print('x552: {}'.format(x552))
        elif isinstance(x552, torch.Tensor):
            print('x552: {}'.format(x552.shape))
        elif isinstance(x552, tuple):
            tuple_shapes = '('
            for item in x552:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x552: {}'.format(tuple_shapes))
        else:
            print('x552: {}'.format(x552))
        x553=operator.mul(x552, x547)
        if x553 is None:
            print('x553: {}'.format(x553))
        elif isinstance(x553, torch.Tensor):
            print('x553: {}'.format(x553.shape))
        elif isinstance(x553, tuple):
            tuple_shapes = '('
            for item in x553:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x553: {}'.format(tuple_shapes))
        else:
            print('x553: {}'.format(x553))
        x554=self.conv2d176(x553)
        if x554 is None:
            print('x554: {}'.format(x554))
        elif isinstance(x554, torch.Tensor):
            print('x554: {}'.format(x554.shape))
        elif isinstance(x554, tuple):
            tuple_shapes = '('
            for item in x554:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x554: {}'.format(tuple_shapes))
        else:
            print('x554: {}'.format(x554))
        x555=self.batchnorm2d104(x554)
        if x555 is None:
            print('x555: {}'.format(x555))
        elif isinstance(x555, torch.Tensor):
            print('x555: {}'.format(x555.shape))
        elif isinstance(x555, tuple):
            tuple_shapes = '('
            for item in x555:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x555: {}'.format(tuple_shapes))
        else:
            print('x555: {}'.format(x555))
        x556=stochastic_depth(x555, 0.12727272727272726, 'row', False)
        if x556 is None:
            print('x556: {}'.format(x556))
        elif isinstance(x556, torch.Tensor):
            print('x556: {}'.format(x556.shape))
        elif isinstance(x556, tuple):
            tuple_shapes = '('
            for item in x556:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x556: {}'.format(tuple_shapes))
        else:
            print('x556: {}'.format(x556))
        x557=operator.add(x556, x541)
        if x557 is None:
            print('x557: {}'.format(x557))
        elif isinstance(x557, torch.Tensor):
            print('x557: {}'.format(x557.shape))
        elif isinstance(x557, tuple):
            tuple_shapes = '('
            for item in x557:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x557: {}'.format(tuple_shapes))
        else:
            print('x557: {}'.format(x557))
        x558=self.conv2d177(x557)
        if x558 is None:
            print('x558: {}'.format(x558))
        elif isinstance(x558, torch.Tensor):
            print('x558: {}'.format(x558.shape))
        elif isinstance(x558, tuple):
            tuple_shapes = '('
            for item in x558:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x558: {}'.format(tuple_shapes))
        else:
            print('x558: {}'.format(x558))
        x559=self.batchnorm2d105(x558)
        if x559 is None:
            print('x559: {}'.format(x559))
        elif isinstance(x559, torch.Tensor):
            print('x559: {}'.format(x559.shape))
        elif isinstance(x559, tuple):
            tuple_shapes = '('
            for item in x559:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x559: {}'.format(tuple_shapes))
        else:
            print('x559: {}'.format(x559))
        x560=self.silu105(x559)
        if x560 is None:
            print('x560: {}'.format(x560))
        elif isinstance(x560, torch.Tensor):
            print('x560: {}'.format(x560.shape))
        elif isinstance(x560, tuple):
            tuple_shapes = '('
            for item in x560:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x560: {}'.format(tuple_shapes))
        else:
            print('x560: {}'.format(x560))
        x561=self.conv2d178(x560)
        if x561 is None:
            print('x561: {}'.format(x561))
        elif isinstance(x561, torch.Tensor):
            print('x561: {}'.format(x561.shape))
        elif isinstance(x561, tuple):
            tuple_shapes = '('
            for item in x561:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x561: {}'.format(tuple_shapes))
        else:
            print('x561: {}'.format(x561))
        x562=self.batchnorm2d106(x561)
        if x562 is None:
            print('x562: {}'.format(x562))
        elif isinstance(x562, torch.Tensor):
            print('x562: {}'.format(x562.shape))
        elif isinstance(x562, tuple):
            tuple_shapes = '('
            for item in x562:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x562: {}'.format(tuple_shapes))
        else:
            print('x562: {}'.format(x562))
        x563=self.silu106(x562)
        if x563 is None:
            print('x563: {}'.format(x563))
        elif isinstance(x563, torch.Tensor):
            print('x563: {}'.format(x563.shape))
        elif isinstance(x563, tuple):
            tuple_shapes = '('
            for item in x563:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x563: {}'.format(tuple_shapes))
        else:
            print('x563: {}'.format(x563))
        x564=self.adaptiveavgpool2d36(x563)
        if x564 is None:
            print('x564: {}'.format(x564))
        elif isinstance(x564, torch.Tensor):
            print('x564: {}'.format(x564.shape))
        elif isinstance(x564, tuple):
            tuple_shapes = '('
            for item in x564:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x564: {}'.format(tuple_shapes))
        else:
            print('x564: {}'.format(x564))
        x565=self.conv2d179(x564)
        if x565 is None:
            print('x565: {}'.format(x565))
        elif isinstance(x565, torch.Tensor):
            print('x565: {}'.format(x565.shape))
        elif isinstance(x565, tuple):
            tuple_shapes = '('
            for item in x565:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x565: {}'.format(tuple_shapes))
        else:
            print('x565: {}'.format(x565))
        x566=self.silu107(x565)
        if x566 is None:
            print('x566: {}'.format(x566))
        elif isinstance(x566, torch.Tensor):
            print('x566: {}'.format(x566.shape))
        elif isinstance(x566, tuple):
            tuple_shapes = '('
            for item in x566:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x566: {}'.format(tuple_shapes))
        else:
            print('x566: {}'.format(x566))
        x567=self.conv2d180(x566)
        if x567 is None:
            print('x567: {}'.format(x567))
        elif isinstance(x567, torch.Tensor):
            print('x567: {}'.format(x567.shape))
        elif isinstance(x567, tuple):
            tuple_shapes = '('
            for item in x567:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x567: {}'.format(tuple_shapes))
        else:
            print('x567: {}'.format(x567))
        x568=self.sigmoid36(x567)
        if x568 is None:
            print('x568: {}'.format(x568))
        elif isinstance(x568, torch.Tensor):
            print('x568: {}'.format(x568.shape))
        elif isinstance(x568, tuple):
            tuple_shapes = '('
            for item in x568:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x568: {}'.format(tuple_shapes))
        else:
            print('x568: {}'.format(x568))
        x569=operator.mul(x568, x563)
        if x569 is None:
            print('x569: {}'.format(x569))
        elif isinstance(x569, torch.Tensor):
            print('x569: {}'.format(x569.shape))
        elif isinstance(x569, tuple):
            tuple_shapes = '('
            for item in x569:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x569: {}'.format(tuple_shapes))
        else:
            print('x569: {}'.format(x569))
        x570=self.conv2d181(x569)
        if x570 is None:
            print('x570: {}'.format(x570))
        elif isinstance(x570, torch.Tensor):
            print('x570: {}'.format(x570.shape))
        elif isinstance(x570, tuple):
            tuple_shapes = '('
            for item in x570:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x570: {}'.format(tuple_shapes))
        else:
            print('x570: {}'.format(x570))
        x571=self.batchnorm2d107(x570)
        if x571 is None:
            print('x571: {}'.format(x571))
        elif isinstance(x571, torch.Tensor):
            print('x571: {}'.format(x571.shape))
        elif isinstance(x571, tuple):
            tuple_shapes = '('
            for item in x571:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x571: {}'.format(tuple_shapes))
        else:
            print('x571: {}'.format(x571))
        x572=stochastic_depth(x571, 0.13090909090909092, 'row', False)
        if x572 is None:
            print('x572: {}'.format(x572))
        elif isinstance(x572, torch.Tensor):
            print('x572: {}'.format(x572.shape))
        elif isinstance(x572, tuple):
            tuple_shapes = '('
            for item in x572:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x572: {}'.format(tuple_shapes))
        else:
            print('x572: {}'.format(x572))
        x573=operator.add(x572, x557)
        if x573 is None:
            print('x573: {}'.format(x573))
        elif isinstance(x573, torch.Tensor):
            print('x573: {}'.format(x573.shape))
        elif isinstance(x573, tuple):
            tuple_shapes = '('
            for item in x573:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x573: {}'.format(tuple_shapes))
        else:
            print('x573: {}'.format(x573))
        x574=self.conv2d182(x573)
        if x574 is None:
            print('x574: {}'.format(x574))
        elif isinstance(x574, torch.Tensor):
            print('x574: {}'.format(x574.shape))
        elif isinstance(x574, tuple):
            tuple_shapes = '('
            for item in x574:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x574: {}'.format(tuple_shapes))
        else:
            print('x574: {}'.format(x574))
        x575=self.batchnorm2d108(x574)
        if x575 is None:
            print('x575: {}'.format(x575))
        elif isinstance(x575, torch.Tensor):
            print('x575: {}'.format(x575.shape))
        elif isinstance(x575, tuple):
            tuple_shapes = '('
            for item in x575:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x575: {}'.format(tuple_shapes))
        else:
            print('x575: {}'.format(x575))
        x576=self.silu108(x575)
        if x576 is None:
            print('x576: {}'.format(x576))
        elif isinstance(x576, torch.Tensor):
            print('x576: {}'.format(x576.shape))
        elif isinstance(x576, tuple):
            tuple_shapes = '('
            for item in x576:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x576: {}'.format(tuple_shapes))
        else:
            print('x576: {}'.format(x576))
        x577=self.conv2d183(x576)
        if x577 is None:
            print('x577: {}'.format(x577))
        elif isinstance(x577, torch.Tensor):
            print('x577: {}'.format(x577.shape))
        elif isinstance(x577, tuple):
            tuple_shapes = '('
            for item in x577:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x577: {}'.format(tuple_shapes))
        else:
            print('x577: {}'.format(x577))
        x578=self.batchnorm2d109(x577)
        if x578 is None:
            print('x578: {}'.format(x578))
        elif isinstance(x578, torch.Tensor):
            print('x578: {}'.format(x578.shape))
        elif isinstance(x578, tuple):
            tuple_shapes = '('
            for item in x578:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x578: {}'.format(tuple_shapes))
        else:
            print('x578: {}'.format(x578))
        x579=self.silu109(x578)
        if x579 is None:
            print('x579: {}'.format(x579))
        elif isinstance(x579, torch.Tensor):
            print('x579: {}'.format(x579.shape))
        elif isinstance(x579, tuple):
            tuple_shapes = '('
            for item in x579:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x579: {}'.format(tuple_shapes))
        else:
            print('x579: {}'.format(x579))
        x580=self.adaptiveavgpool2d37(x579)
        if x580 is None:
            print('x580: {}'.format(x580))
        elif isinstance(x580, torch.Tensor):
            print('x580: {}'.format(x580.shape))
        elif isinstance(x580, tuple):
            tuple_shapes = '('
            for item in x580:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x580: {}'.format(tuple_shapes))
        else:
            print('x580: {}'.format(x580))
        x581=self.conv2d184(x580)
        if x581 is None:
            print('x581: {}'.format(x581))
        elif isinstance(x581, torch.Tensor):
            print('x581: {}'.format(x581.shape))
        elif isinstance(x581, tuple):
            tuple_shapes = '('
            for item in x581:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x581: {}'.format(tuple_shapes))
        else:
            print('x581: {}'.format(x581))
        x582=self.silu110(x581)
        if x582 is None:
            print('x582: {}'.format(x582))
        elif isinstance(x582, torch.Tensor):
            print('x582: {}'.format(x582.shape))
        elif isinstance(x582, tuple):
            tuple_shapes = '('
            for item in x582:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x582: {}'.format(tuple_shapes))
        else:
            print('x582: {}'.format(x582))
        x583=self.conv2d185(x582)
        if x583 is None:
            print('x583: {}'.format(x583))
        elif isinstance(x583, torch.Tensor):
            print('x583: {}'.format(x583.shape))
        elif isinstance(x583, tuple):
            tuple_shapes = '('
            for item in x583:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x583: {}'.format(tuple_shapes))
        else:
            print('x583: {}'.format(x583))
        x584=self.sigmoid37(x583)
        if x584 is None:
            print('x584: {}'.format(x584))
        elif isinstance(x584, torch.Tensor):
            print('x584: {}'.format(x584.shape))
        elif isinstance(x584, tuple):
            tuple_shapes = '('
            for item in x584:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x584: {}'.format(tuple_shapes))
        else:
            print('x584: {}'.format(x584))
        x585=operator.mul(x584, x579)
        if x585 is None:
            print('x585: {}'.format(x585))
        elif isinstance(x585, torch.Tensor):
            print('x585: {}'.format(x585.shape))
        elif isinstance(x585, tuple):
            tuple_shapes = '('
            for item in x585:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x585: {}'.format(tuple_shapes))
        else:
            print('x585: {}'.format(x585))
        x586=self.conv2d186(x585)
        if x586 is None:
            print('x586: {}'.format(x586))
        elif isinstance(x586, torch.Tensor):
            print('x586: {}'.format(x586.shape))
        elif isinstance(x586, tuple):
            tuple_shapes = '('
            for item in x586:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x586: {}'.format(tuple_shapes))
        else:
            print('x586: {}'.format(x586))
        x587=self.batchnorm2d110(x586)
        if x587 is None:
            print('x587: {}'.format(x587))
        elif isinstance(x587, torch.Tensor):
            print('x587: {}'.format(x587.shape))
        elif isinstance(x587, tuple):
            tuple_shapes = '('
            for item in x587:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x587: {}'.format(tuple_shapes))
        else:
            print('x587: {}'.format(x587))
        x588=stochastic_depth(x587, 0.13454545454545455, 'row', False)
        if x588 is None:
            print('x588: {}'.format(x588))
        elif isinstance(x588, torch.Tensor):
            print('x588: {}'.format(x588.shape))
        elif isinstance(x588, tuple):
            tuple_shapes = '('
            for item in x588:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x588: {}'.format(tuple_shapes))
        else:
            print('x588: {}'.format(x588))
        x589=operator.add(x588, x573)
        if x589 is None:
            print('x589: {}'.format(x589))
        elif isinstance(x589, torch.Tensor):
            print('x589: {}'.format(x589.shape))
        elif isinstance(x589, tuple):
            tuple_shapes = '('
            for item in x589:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x589: {}'.format(tuple_shapes))
        else:
            print('x589: {}'.format(x589))
        x590=self.conv2d187(x589)
        if x590 is None:
            print('x590: {}'.format(x590))
        elif isinstance(x590, torch.Tensor):
            print('x590: {}'.format(x590.shape))
        elif isinstance(x590, tuple):
            tuple_shapes = '('
            for item in x590:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x590: {}'.format(tuple_shapes))
        else:
            print('x590: {}'.format(x590))
        x591=self.batchnorm2d111(x590)
        if x591 is None:
            print('x591: {}'.format(x591))
        elif isinstance(x591, torch.Tensor):
            print('x591: {}'.format(x591.shape))
        elif isinstance(x591, tuple):
            tuple_shapes = '('
            for item in x591:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x591: {}'.format(tuple_shapes))
        else:
            print('x591: {}'.format(x591))
        x592=self.silu111(x591)
        if x592 is None:
            print('x592: {}'.format(x592))
        elif isinstance(x592, torch.Tensor):
            print('x592: {}'.format(x592.shape))
        elif isinstance(x592, tuple):
            tuple_shapes = '('
            for item in x592:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x592: {}'.format(tuple_shapes))
        else:
            print('x592: {}'.format(x592))
        x593=self.conv2d188(x592)
        if x593 is None:
            print('x593: {}'.format(x593))
        elif isinstance(x593, torch.Tensor):
            print('x593: {}'.format(x593.shape))
        elif isinstance(x593, tuple):
            tuple_shapes = '('
            for item in x593:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x593: {}'.format(tuple_shapes))
        else:
            print('x593: {}'.format(x593))
        x594=self.batchnorm2d112(x593)
        if x594 is None:
            print('x594: {}'.format(x594))
        elif isinstance(x594, torch.Tensor):
            print('x594: {}'.format(x594.shape))
        elif isinstance(x594, tuple):
            tuple_shapes = '('
            for item in x594:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x594: {}'.format(tuple_shapes))
        else:
            print('x594: {}'.format(x594))
        x595=self.silu112(x594)
        if x595 is None:
            print('x595: {}'.format(x595))
        elif isinstance(x595, torch.Tensor):
            print('x595: {}'.format(x595.shape))
        elif isinstance(x595, tuple):
            tuple_shapes = '('
            for item in x595:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x595: {}'.format(tuple_shapes))
        else:
            print('x595: {}'.format(x595))
        x596=self.adaptiveavgpool2d38(x595)
        if x596 is None:
            print('x596: {}'.format(x596))
        elif isinstance(x596, torch.Tensor):
            print('x596: {}'.format(x596.shape))
        elif isinstance(x596, tuple):
            tuple_shapes = '('
            for item in x596:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x596: {}'.format(tuple_shapes))
        else:
            print('x596: {}'.format(x596))
        x597=self.conv2d189(x596)
        if x597 is None:
            print('x597: {}'.format(x597))
        elif isinstance(x597, torch.Tensor):
            print('x597: {}'.format(x597.shape))
        elif isinstance(x597, tuple):
            tuple_shapes = '('
            for item in x597:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x597: {}'.format(tuple_shapes))
        else:
            print('x597: {}'.format(x597))
        x598=self.silu113(x597)
        if x598 is None:
            print('x598: {}'.format(x598))
        elif isinstance(x598, torch.Tensor):
            print('x598: {}'.format(x598.shape))
        elif isinstance(x598, tuple):
            tuple_shapes = '('
            for item in x598:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x598: {}'.format(tuple_shapes))
        else:
            print('x598: {}'.format(x598))
        x599=self.conv2d190(x598)
        if x599 is None:
            print('x599: {}'.format(x599))
        elif isinstance(x599, torch.Tensor):
            print('x599: {}'.format(x599.shape))
        elif isinstance(x599, tuple):
            tuple_shapes = '('
            for item in x599:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x599: {}'.format(tuple_shapes))
        else:
            print('x599: {}'.format(x599))
        x600=self.sigmoid38(x599)
        if x600 is None:
            print('x600: {}'.format(x600))
        elif isinstance(x600, torch.Tensor):
            print('x600: {}'.format(x600.shape))
        elif isinstance(x600, tuple):
            tuple_shapes = '('
            for item in x600:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x600: {}'.format(tuple_shapes))
        else:
            print('x600: {}'.format(x600))
        x601=operator.mul(x600, x595)
        if x601 is None:
            print('x601: {}'.format(x601))
        elif isinstance(x601, torch.Tensor):
            print('x601: {}'.format(x601.shape))
        elif isinstance(x601, tuple):
            tuple_shapes = '('
            for item in x601:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x601: {}'.format(tuple_shapes))
        else:
            print('x601: {}'.format(x601))
        x602=self.conv2d191(x601)
        if x602 is None:
            print('x602: {}'.format(x602))
        elif isinstance(x602, torch.Tensor):
            print('x602: {}'.format(x602.shape))
        elif isinstance(x602, tuple):
            tuple_shapes = '('
            for item in x602:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x602: {}'.format(tuple_shapes))
        else:
            print('x602: {}'.format(x602))
        x603=self.batchnorm2d113(x602)
        if x603 is None:
            print('x603: {}'.format(x603))
        elif isinstance(x603, torch.Tensor):
            print('x603: {}'.format(x603.shape))
        elif isinstance(x603, tuple):
            tuple_shapes = '('
            for item in x603:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x603: {}'.format(tuple_shapes))
        else:
            print('x603: {}'.format(x603))
        x604=self.conv2d192(x603)
        if x604 is None:
            print('x604: {}'.format(x604))
        elif isinstance(x604, torch.Tensor):
            print('x604: {}'.format(x604.shape))
        elif isinstance(x604, tuple):
            tuple_shapes = '('
            for item in x604:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x604: {}'.format(tuple_shapes))
        else:
            print('x604: {}'.format(x604))
        x605=self.batchnorm2d114(x604)
        if x605 is None:
            print('x605: {}'.format(x605))
        elif isinstance(x605, torch.Tensor):
            print('x605: {}'.format(x605.shape))
        elif isinstance(x605, tuple):
            tuple_shapes = '('
            for item in x605:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x605: {}'.format(tuple_shapes))
        else:
            print('x605: {}'.format(x605))
        x606=self.silu114(x605)
        if x606 is None:
            print('x606: {}'.format(x606))
        elif isinstance(x606, torch.Tensor):
            print('x606: {}'.format(x606.shape))
        elif isinstance(x606, tuple):
            tuple_shapes = '('
            for item in x606:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x606: {}'.format(tuple_shapes))
        else:
            print('x606: {}'.format(x606))
        x607=self.conv2d193(x606)
        if x607 is None:
            print('x607: {}'.format(x607))
        elif isinstance(x607, torch.Tensor):
            print('x607: {}'.format(x607.shape))
        elif isinstance(x607, tuple):
            tuple_shapes = '('
            for item in x607:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x607: {}'.format(tuple_shapes))
        else:
            print('x607: {}'.format(x607))
        x608=self.batchnorm2d115(x607)
        if x608 is None:
            print('x608: {}'.format(x608))
        elif isinstance(x608, torch.Tensor):
            print('x608: {}'.format(x608.shape))
        elif isinstance(x608, tuple):
            tuple_shapes = '('
            for item in x608:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x608: {}'.format(tuple_shapes))
        else:
            print('x608: {}'.format(x608))
        x609=self.silu115(x608)
        if x609 is None:
            print('x609: {}'.format(x609))
        elif isinstance(x609, torch.Tensor):
            print('x609: {}'.format(x609.shape))
        elif isinstance(x609, tuple):
            tuple_shapes = '('
            for item in x609:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x609: {}'.format(tuple_shapes))
        else:
            print('x609: {}'.format(x609))
        x610=self.adaptiveavgpool2d39(x609)
        if x610 is None:
            print('x610: {}'.format(x610))
        elif isinstance(x610, torch.Tensor):
            print('x610: {}'.format(x610.shape))
        elif isinstance(x610, tuple):
            tuple_shapes = '('
            for item in x610:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x610: {}'.format(tuple_shapes))
        else:
            print('x610: {}'.format(x610))
        x611=self.conv2d194(x610)
        if x611 is None:
            print('x611: {}'.format(x611))
        elif isinstance(x611, torch.Tensor):
            print('x611: {}'.format(x611.shape))
        elif isinstance(x611, tuple):
            tuple_shapes = '('
            for item in x611:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x611: {}'.format(tuple_shapes))
        else:
            print('x611: {}'.format(x611))
        x612=self.silu116(x611)
        if x612 is None:
            print('x612: {}'.format(x612))
        elif isinstance(x612, torch.Tensor):
            print('x612: {}'.format(x612.shape))
        elif isinstance(x612, tuple):
            tuple_shapes = '('
            for item in x612:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x612: {}'.format(tuple_shapes))
        else:
            print('x612: {}'.format(x612))
        x613=self.conv2d195(x612)
        if x613 is None:
            print('x613: {}'.format(x613))
        elif isinstance(x613, torch.Tensor):
            print('x613: {}'.format(x613.shape))
        elif isinstance(x613, tuple):
            tuple_shapes = '('
            for item in x613:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x613: {}'.format(tuple_shapes))
        else:
            print('x613: {}'.format(x613))
        x614=self.sigmoid39(x613)
        if x614 is None:
            print('x614: {}'.format(x614))
        elif isinstance(x614, torch.Tensor):
            print('x614: {}'.format(x614.shape))
        elif isinstance(x614, tuple):
            tuple_shapes = '('
            for item in x614:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x614: {}'.format(tuple_shapes))
        else:
            print('x614: {}'.format(x614))
        x615=operator.mul(x614, x609)
        if x615 is None:
            print('x615: {}'.format(x615))
        elif isinstance(x615, torch.Tensor):
            print('x615: {}'.format(x615.shape))
        elif isinstance(x615, tuple):
            tuple_shapes = '('
            for item in x615:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x615: {}'.format(tuple_shapes))
        else:
            print('x615: {}'.format(x615))
        x616=self.conv2d196(x615)
        if x616 is None:
            print('x616: {}'.format(x616))
        elif isinstance(x616, torch.Tensor):
            print('x616: {}'.format(x616.shape))
        elif isinstance(x616, tuple):
            tuple_shapes = '('
            for item in x616:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x616: {}'.format(tuple_shapes))
        else:
            print('x616: {}'.format(x616))
        x617=self.batchnorm2d116(x616)
        if x617 is None:
            print('x617: {}'.format(x617))
        elif isinstance(x617, torch.Tensor):
            print('x617: {}'.format(x617.shape))
        elif isinstance(x617, tuple):
            tuple_shapes = '('
            for item in x617:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x617: {}'.format(tuple_shapes))
        else:
            print('x617: {}'.format(x617))
        x618=stochastic_depth(x617, 0.14181818181818184, 'row', False)
        if x618 is None:
            print('x618: {}'.format(x618))
        elif isinstance(x618, torch.Tensor):
            print('x618: {}'.format(x618.shape))
        elif isinstance(x618, tuple):
            tuple_shapes = '('
            for item in x618:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x618: {}'.format(tuple_shapes))
        else:
            print('x618: {}'.format(x618))
        x619=operator.add(x618, x603)
        if x619 is None:
            print('x619: {}'.format(x619))
        elif isinstance(x619, torch.Tensor):
            print('x619: {}'.format(x619.shape))
        elif isinstance(x619, tuple):
            tuple_shapes = '('
            for item in x619:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x619: {}'.format(tuple_shapes))
        else:
            print('x619: {}'.format(x619))
        x620=self.conv2d197(x619)
        if x620 is None:
            print('x620: {}'.format(x620))
        elif isinstance(x620, torch.Tensor):
            print('x620: {}'.format(x620.shape))
        elif isinstance(x620, tuple):
            tuple_shapes = '('
            for item in x620:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x620: {}'.format(tuple_shapes))
        else:
            print('x620: {}'.format(x620))
        x621=self.batchnorm2d117(x620)
        if x621 is None:
            print('x621: {}'.format(x621))
        elif isinstance(x621, torch.Tensor):
            print('x621: {}'.format(x621.shape))
        elif isinstance(x621, tuple):
            tuple_shapes = '('
            for item in x621:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x621: {}'.format(tuple_shapes))
        else:
            print('x621: {}'.format(x621))
        x622=self.silu117(x621)
        if x622 is None:
            print('x622: {}'.format(x622))
        elif isinstance(x622, torch.Tensor):
            print('x622: {}'.format(x622.shape))
        elif isinstance(x622, tuple):
            tuple_shapes = '('
            for item in x622:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x622: {}'.format(tuple_shapes))
        else:
            print('x622: {}'.format(x622))
        x623=self.conv2d198(x622)
        if x623 is None:
            print('x623: {}'.format(x623))
        elif isinstance(x623, torch.Tensor):
            print('x623: {}'.format(x623.shape))
        elif isinstance(x623, tuple):
            tuple_shapes = '('
            for item in x623:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x623: {}'.format(tuple_shapes))
        else:
            print('x623: {}'.format(x623))
        x624=self.batchnorm2d118(x623)
        if x624 is None:
            print('x624: {}'.format(x624))
        elif isinstance(x624, torch.Tensor):
            print('x624: {}'.format(x624.shape))
        elif isinstance(x624, tuple):
            tuple_shapes = '('
            for item in x624:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x624: {}'.format(tuple_shapes))
        else:
            print('x624: {}'.format(x624))
        x625=self.silu118(x624)
        if x625 is None:
            print('x625: {}'.format(x625))
        elif isinstance(x625, torch.Tensor):
            print('x625: {}'.format(x625.shape))
        elif isinstance(x625, tuple):
            tuple_shapes = '('
            for item in x625:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x625: {}'.format(tuple_shapes))
        else:
            print('x625: {}'.format(x625))
        x626=self.adaptiveavgpool2d40(x625)
        if x626 is None:
            print('x626: {}'.format(x626))
        elif isinstance(x626, torch.Tensor):
            print('x626: {}'.format(x626.shape))
        elif isinstance(x626, tuple):
            tuple_shapes = '('
            for item in x626:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x626: {}'.format(tuple_shapes))
        else:
            print('x626: {}'.format(x626))
        x627=self.conv2d199(x626)
        if x627 is None:
            print('x627: {}'.format(x627))
        elif isinstance(x627, torch.Tensor):
            print('x627: {}'.format(x627.shape))
        elif isinstance(x627, tuple):
            tuple_shapes = '('
            for item in x627:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x627: {}'.format(tuple_shapes))
        else:
            print('x627: {}'.format(x627))
        x628=self.silu119(x627)
        if x628 is None:
            print('x628: {}'.format(x628))
        elif isinstance(x628, torch.Tensor):
            print('x628: {}'.format(x628.shape))
        elif isinstance(x628, tuple):
            tuple_shapes = '('
            for item in x628:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x628: {}'.format(tuple_shapes))
        else:
            print('x628: {}'.format(x628))
        x629=self.conv2d200(x628)
        if x629 is None:
            print('x629: {}'.format(x629))
        elif isinstance(x629, torch.Tensor):
            print('x629: {}'.format(x629.shape))
        elif isinstance(x629, tuple):
            tuple_shapes = '('
            for item in x629:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x629: {}'.format(tuple_shapes))
        else:
            print('x629: {}'.format(x629))
        x630=self.sigmoid40(x629)
        if x630 is None:
            print('x630: {}'.format(x630))
        elif isinstance(x630, torch.Tensor):
            print('x630: {}'.format(x630.shape))
        elif isinstance(x630, tuple):
            tuple_shapes = '('
            for item in x630:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x630: {}'.format(tuple_shapes))
        else:
            print('x630: {}'.format(x630))
        x631=operator.mul(x630, x625)
        if x631 is None:
            print('x631: {}'.format(x631))
        elif isinstance(x631, torch.Tensor):
            print('x631: {}'.format(x631.shape))
        elif isinstance(x631, tuple):
            tuple_shapes = '('
            for item in x631:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x631: {}'.format(tuple_shapes))
        else:
            print('x631: {}'.format(x631))
        x632=self.conv2d201(x631)
        if x632 is None:
            print('x632: {}'.format(x632))
        elif isinstance(x632, torch.Tensor):
            print('x632: {}'.format(x632.shape))
        elif isinstance(x632, tuple):
            tuple_shapes = '('
            for item in x632:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x632: {}'.format(tuple_shapes))
        else:
            print('x632: {}'.format(x632))
        x633=self.batchnorm2d119(x632)
        if x633 is None:
            print('x633: {}'.format(x633))
        elif isinstance(x633, torch.Tensor):
            print('x633: {}'.format(x633.shape))
        elif isinstance(x633, tuple):
            tuple_shapes = '('
            for item in x633:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x633: {}'.format(tuple_shapes))
        else:
            print('x633: {}'.format(x633))
        x634=stochastic_depth(x633, 0.14545454545454545, 'row', False)
        if x634 is None:
            print('x634: {}'.format(x634))
        elif isinstance(x634, torch.Tensor):
            print('x634: {}'.format(x634.shape))
        elif isinstance(x634, tuple):
            tuple_shapes = '('
            for item in x634:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x634: {}'.format(tuple_shapes))
        else:
            print('x634: {}'.format(x634))
        x635=operator.add(x634, x619)
        if x635 is None:
            print('x635: {}'.format(x635))
        elif isinstance(x635, torch.Tensor):
            print('x635: {}'.format(x635.shape))
        elif isinstance(x635, tuple):
            tuple_shapes = '('
            for item in x635:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x635: {}'.format(tuple_shapes))
        else:
            print('x635: {}'.format(x635))
        x636=self.conv2d202(x635)
        if x636 is None:
            print('x636: {}'.format(x636))
        elif isinstance(x636, torch.Tensor):
            print('x636: {}'.format(x636.shape))
        elif isinstance(x636, tuple):
            tuple_shapes = '('
            for item in x636:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x636: {}'.format(tuple_shapes))
        else:
            print('x636: {}'.format(x636))
        x637=self.batchnorm2d120(x636)
        if x637 is None:
            print('x637: {}'.format(x637))
        elif isinstance(x637, torch.Tensor):
            print('x637: {}'.format(x637.shape))
        elif isinstance(x637, tuple):
            tuple_shapes = '('
            for item in x637:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x637: {}'.format(tuple_shapes))
        else:
            print('x637: {}'.format(x637))
        x638=self.silu120(x637)
        if x638 is None:
            print('x638: {}'.format(x638))
        elif isinstance(x638, torch.Tensor):
            print('x638: {}'.format(x638.shape))
        elif isinstance(x638, tuple):
            tuple_shapes = '('
            for item in x638:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x638: {}'.format(tuple_shapes))
        else:
            print('x638: {}'.format(x638))
        x639=self.conv2d203(x638)
        if x639 is None:
            print('x639: {}'.format(x639))
        elif isinstance(x639, torch.Tensor):
            print('x639: {}'.format(x639.shape))
        elif isinstance(x639, tuple):
            tuple_shapes = '('
            for item in x639:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x639: {}'.format(tuple_shapes))
        else:
            print('x639: {}'.format(x639))
        x640=self.batchnorm2d121(x639)
        if x640 is None:
            print('x640: {}'.format(x640))
        elif isinstance(x640, torch.Tensor):
            print('x640: {}'.format(x640.shape))
        elif isinstance(x640, tuple):
            tuple_shapes = '('
            for item in x640:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x640: {}'.format(tuple_shapes))
        else:
            print('x640: {}'.format(x640))
        x641=self.silu121(x640)
        if x641 is None:
            print('x641: {}'.format(x641))
        elif isinstance(x641, torch.Tensor):
            print('x641: {}'.format(x641.shape))
        elif isinstance(x641, tuple):
            tuple_shapes = '('
            for item in x641:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x641: {}'.format(tuple_shapes))
        else:
            print('x641: {}'.format(x641))
        x642=self.adaptiveavgpool2d41(x641)
        if x642 is None:
            print('x642: {}'.format(x642))
        elif isinstance(x642, torch.Tensor):
            print('x642: {}'.format(x642.shape))
        elif isinstance(x642, tuple):
            tuple_shapes = '('
            for item in x642:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x642: {}'.format(tuple_shapes))
        else:
            print('x642: {}'.format(x642))
        x643=self.conv2d204(x642)
        if x643 is None:
            print('x643: {}'.format(x643))
        elif isinstance(x643, torch.Tensor):
            print('x643: {}'.format(x643.shape))
        elif isinstance(x643, tuple):
            tuple_shapes = '('
            for item in x643:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x643: {}'.format(tuple_shapes))
        else:
            print('x643: {}'.format(x643))
        x644=self.silu122(x643)
        if x644 is None:
            print('x644: {}'.format(x644))
        elif isinstance(x644, torch.Tensor):
            print('x644: {}'.format(x644.shape))
        elif isinstance(x644, tuple):
            tuple_shapes = '('
            for item in x644:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x644: {}'.format(tuple_shapes))
        else:
            print('x644: {}'.format(x644))
        x645=self.conv2d205(x644)
        if x645 is None:
            print('x645: {}'.format(x645))
        elif isinstance(x645, torch.Tensor):
            print('x645: {}'.format(x645.shape))
        elif isinstance(x645, tuple):
            tuple_shapes = '('
            for item in x645:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x645: {}'.format(tuple_shapes))
        else:
            print('x645: {}'.format(x645))
        x646=self.sigmoid41(x645)
        if x646 is None:
            print('x646: {}'.format(x646))
        elif isinstance(x646, torch.Tensor):
            print('x646: {}'.format(x646.shape))
        elif isinstance(x646, tuple):
            tuple_shapes = '('
            for item in x646:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x646: {}'.format(tuple_shapes))
        else:
            print('x646: {}'.format(x646))
        x647=operator.mul(x646, x641)
        if x647 is None:
            print('x647: {}'.format(x647))
        elif isinstance(x647, torch.Tensor):
            print('x647: {}'.format(x647.shape))
        elif isinstance(x647, tuple):
            tuple_shapes = '('
            for item in x647:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x647: {}'.format(tuple_shapes))
        else:
            print('x647: {}'.format(x647))
        x648=self.conv2d206(x647)
        if x648 is None:
            print('x648: {}'.format(x648))
        elif isinstance(x648, torch.Tensor):
            print('x648: {}'.format(x648.shape))
        elif isinstance(x648, tuple):
            tuple_shapes = '('
            for item in x648:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x648: {}'.format(tuple_shapes))
        else:
            print('x648: {}'.format(x648))
        x649=self.batchnorm2d122(x648)
        if x649 is None:
            print('x649: {}'.format(x649))
        elif isinstance(x649, torch.Tensor):
            print('x649: {}'.format(x649.shape))
        elif isinstance(x649, tuple):
            tuple_shapes = '('
            for item in x649:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x649: {}'.format(tuple_shapes))
        else:
            print('x649: {}'.format(x649))
        x650=stochastic_depth(x649, 0.1490909090909091, 'row', False)
        if x650 is None:
            print('x650: {}'.format(x650))
        elif isinstance(x650, torch.Tensor):
            print('x650: {}'.format(x650.shape))
        elif isinstance(x650, tuple):
            tuple_shapes = '('
            for item in x650:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x650: {}'.format(tuple_shapes))
        else:
            print('x650: {}'.format(x650))
        x651=operator.add(x650, x635)
        if x651 is None:
            print('x651: {}'.format(x651))
        elif isinstance(x651, torch.Tensor):
            print('x651: {}'.format(x651.shape))
        elif isinstance(x651, tuple):
            tuple_shapes = '('
            for item in x651:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x651: {}'.format(tuple_shapes))
        else:
            print('x651: {}'.format(x651))
        x652=self.conv2d207(x651)
        if x652 is None:
            print('x652: {}'.format(x652))
        elif isinstance(x652, torch.Tensor):
            print('x652: {}'.format(x652.shape))
        elif isinstance(x652, tuple):
            tuple_shapes = '('
            for item in x652:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x652: {}'.format(tuple_shapes))
        else:
            print('x652: {}'.format(x652))
        x653=self.batchnorm2d123(x652)
        if x653 is None:
            print('x653: {}'.format(x653))
        elif isinstance(x653, torch.Tensor):
            print('x653: {}'.format(x653.shape))
        elif isinstance(x653, tuple):
            tuple_shapes = '('
            for item in x653:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x653: {}'.format(tuple_shapes))
        else:
            print('x653: {}'.format(x653))
        x654=self.silu123(x653)
        if x654 is None:
            print('x654: {}'.format(x654))
        elif isinstance(x654, torch.Tensor):
            print('x654: {}'.format(x654.shape))
        elif isinstance(x654, tuple):
            tuple_shapes = '('
            for item in x654:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x654: {}'.format(tuple_shapes))
        else:
            print('x654: {}'.format(x654))
        x655=self.conv2d208(x654)
        if x655 is None:
            print('x655: {}'.format(x655))
        elif isinstance(x655, torch.Tensor):
            print('x655: {}'.format(x655.shape))
        elif isinstance(x655, tuple):
            tuple_shapes = '('
            for item in x655:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x655: {}'.format(tuple_shapes))
        else:
            print('x655: {}'.format(x655))
        x656=self.batchnorm2d124(x655)
        if x656 is None:
            print('x656: {}'.format(x656))
        elif isinstance(x656, torch.Tensor):
            print('x656: {}'.format(x656.shape))
        elif isinstance(x656, tuple):
            tuple_shapes = '('
            for item in x656:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x656: {}'.format(tuple_shapes))
        else:
            print('x656: {}'.format(x656))
        x657=self.silu124(x656)
        if x657 is None:
            print('x657: {}'.format(x657))
        elif isinstance(x657, torch.Tensor):
            print('x657: {}'.format(x657.shape))
        elif isinstance(x657, tuple):
            tuple_shapes = '('
            for item in x657:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x657: {}'.format(tuple_shapes))
        else:
            print('x657: {}'.format(x657))
        x658=self.adaptiveavgpool2d42(x657)
        if x658 is None:
            print('x658: {}'.format(x658))
        elif isinstance(x658, torch.Tensor):
            print('x658: {}'.format(x658.shape))
        elif isinstance(x658, tuple):
            tuple_shapes = '('
            for item in x658:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x658: {}'.format(tuple_shapes))
        else:
            print('x658: {}'.format(x658))
        x659=self.conv2d209(x658)
        if x659 is None:
            print('x659: {}'.format(x659))
        elif isinstance(x659, torch.Tensor):
            print('x659: {}'.format(x659.shape))
        elif isinstance(x659, tuple):
            tuple_shapes = '('
            for item in x659:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x659: {}'.format(tuple_shapes))
        else:
            print('x659: {}'.format(x659))
        x660=self.silu125(x659)
        if x660 is None:
            print('x660: {}'.format(x660))
        elif isinstance(x660, torch.Tensor):
            print('x660: {}'.format(x660.shape))
        elif isinstance(x660, tuple):
            tuple_shapes = '('
            for item in x660:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x660: {}'.format(tuple_shapes))
        else:
            print('x660: {}'.format(x660))
        x661=self.conv2d210(x660)
        if x661 is None:
            print('x661: {}'.format(x661))
        elif isinstance(x661, torch.Tensor):
            print('x661: {}'.format(x661.shape))
        elif isinstance(x661, tuple):
            tuple_shapes = '('
            for item in x661:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x661: {}'.format(tuple_shapes))
        else:
            print('x661: {}'.format(x661))
        x662=self.sigmoid42(x661)
        if x662 is None:
            print('x662: {}'.format(x662))
        elif isinstance(x662, torch.Tensor):
            print('x662: {}'.format(x662.shape))
        elif isinstance(x662, tuple):
            tuple_shapes = '('
            for item in x662:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x662: {}'.format(tuple_shapes))
        else:
            print('x662: {}'.format(x662))
        x663=operator.mul(x662, x657)
        if x663 is None:
            print('x663: {}'.format(x663))
        elif isinstance(x663, torch.Tensor):
            print('x663: {}'.format(x663.shape))
        elif isinstance(x663, tuple):
            tuple_shapes = '('
            for item in x663:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x663: {}'.format(tuple_shapes))
        else:
            print('x663: {}'.format(x663))
        x664=self.conv2d211(x663)
        if x664 is None:
            print('x664: {}'.format(x664))
        elif isinstance(x664, torch.Tensor):
            print('x664: {}'.format(x664.shape))
        elif isinstance(x664, tuple):
            tuple_shapes = '('
            for item in x664:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x664: {}'.format(tuple_shapes))
        else:
            print('x664: {}'.format(x664))
        x665=self.batchnorm2d125(x664)
        if x665 is None:
            print('x665: {}'.format(x665))
        elif isinstance(x665, torch.Tensor):
            print('x665: {}'.format(x665.shape))
        elif isinstance(x665, tuple):
            tuple_shapes = '('
            for item in x665:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x665: {}'.format(tuple_shapes))
        else:
            print('x665: {}'.format(x665))
        x666=stochastic_depth(x665, 0.15272727272727274, 'row', False)
        if x666 is None:
            print('x666: {}'.format(x666))
        elif isinstance(x666, torch.Tensor):
            print('x666: {}'.format(x666.shape))
        elif isinstance(x666, tuple):
            tuple_shapes = '('
            for item in x666:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x666: {}'.format(tuple_shapes))
        else:
            print('x666: {}'.format(x666))
        x667=operator.add(x666, x651)
        if x667 is None:
            print('x667: {}'.format(x667))
        elif isinstance(x667, torch.Tensor):
            print('x667: {}'.format(x667.shape))
        elif isinstance(x667, tuple):
            tuple_shapes = '('
            for item in x667:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x667: {}'.format(tuple_shapes))
        else:
            print('x667: {}'.format(x667))
        x668=self.conv2d212(x667)
        if x668 is None:
            print('x668: {}'.format(x668))
        elif isinstance(x668, torch.Tensor):
            print('x668: {}'.format(x668.shape))
        elif isinstance(x668, tuple):
            tuple_shapes = '('
            for item in x668:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x668: {}'.format(tuple_shapes))
        else:
            print('x668: {}'.format(x668))
        x669=self.batchnorm2d126(x668)
        if x669 is None:
            print('x669: {}'.format(x669))
        elif isinstance(x669, torch.Tensor):
            print('x669: {}'.format(x669.shape))
        elif isinstance(x669, tuple):
            tuple_shapes = '('
            for item in x669:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x669: {}'.format(tuple_shapes))
        else:
            print('x669: {}'.format(x669))
        x670=self.silu126(x669)
        if x670 is None:
            print('x670: {}'.format(x670))
        elif isinstance(x670, torch.Tensor):
            print('x670: {}'.format(x670.shape))
        elif isinstance(x670, tuple):
            tuple_shapes = '('
            for item in x670:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x670: {}'.format(tuple_shapes))
        else:
            print('x670: {}'.format(x670))
        x671=self.conv2d213(x670)
        if x671 is None:
            print('x671: {}'.format(x671))
        elif isinstance(x671, torch.Tensor):
            print('x671: {}'.format(x671.shape))
        elif isinstance(x671, tuple):
            tuple_shapes = '('
            for item in x671:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x671: {}'.format(tuple_shapes))
        else:
            print('x671: {}'.format(x671))
        x672=self.batchnorm2d127(x671)
        if x672 is None:
            print('x672: {}'.format(x672))
        elif isinstance(x672, torch.Tensor):
            print('x672: {}'.format(x672.shape))
        elif isinstance(x672, tuple):
            tuple_shapes = '('
            for item in x672:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x672: {}'.format(tuple_shapes))
        else:
            print('x672: {}'.format(x672))
        x673=self.silu127(x672)
        if x673 is None:
            print('x673: {}'.format(x673))
        elif isinstance(x673, torch.Tensor):
            print('x673: {}'.format(x673.shape))
        elif isinstance(x673, tuple):
            tuple_shapes = '('
            for item in x673:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x673: {}'.format(tuple_shapes))
        else:
            print('x673: {}'.format(x673))
        x674=self.adaptiveavgpool2d43(x673)
        if x674 is None:
            print('x674: {}'.format(x674))
        elif isinstance(x674, torch.Tensor):
            print('x674: {}'.format(x674.shape))
        elif isinstance(x674, tuple):
            tuple_shapes = '('
            for item in x674:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x674: {}'.format(tuple_shapes))
        else:
            print('x674: {}'.format(x674))
        x675=self.conv2d214(x674)
        if x675 is None:
            print('x675: {}'.format(x675))
        elif isinstance(x675, torch.Tensor):
            print('x675: {}'.format(x675.shape))
        elif isinstance(x675, tuple):
            tuple_shapes = '('
            for item in x675:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x675: {}'.format(tuple_shapes))
        else:
            print('x675: {}'.format(x675))
        x676=self.silu128(x675)
        if x676 is None:
            print('x676: {}'.format(x676))
        elif isinstance(x676, torch.Tensor):
            print('x676: {}'.format(x676.shape))
        elif isinstance(x676, tuple):
            tuple_shapes = '('
            for item in x676:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x676: {}'.format(tuple_shapes))
        else:
            print('x676: {}'.format(x676))
        x677=self.conv2d215(x676)
        if x677 is None:
            print('x677: {}'.format(x677))
        elif isinstance(x677, torch.Tensor):
            print('x677: {}'.format(x677.shape))
        elif isinstance(x677, tuple):
            tuple_shapes = '('
            for item in x677:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x677: {}'.format(tuple_shapes))
        else:
            print('x677: {}'.format(x677))
        x678=self.sigmoid43(x677)
        if x678 is None:
            print('x678: {}'.format(x678))
        elif isinstance(x678, torch.Tensor):
            print('x678: {}'.format(x678.shape))
        elif isinstance(x678, tuple):
            tuple_shapes = '('
            for item in x678:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x678: {}'.format(tuple_shapes))
        else:
            print('x678: {}'.format(x678))
        x679=operator.mul(x678, x673)
        if x679 is None:
            print('x679: {}'.format(x679))
        elif isinstance(x679, torch.Tensor):
            print('x679: {}'.format(x679.shape))
        elif isinstance(x679, tuple):
            tuple_shapes = '('
            for item in x679:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x679: {}'.format(tuple_shapes))
        else:
            print('x679: {}'.format(x679))
        x680=self.conv2d216(x679)
        if x680 is None:
            print('x680: {}'.format(x680))
        elif isinstance(x680, torch.Tensor):
            print('x680: {}'.format(x680.shape))
        elif isinstance(x680, tuple):
            tuple_shapes = '('
            for item in x680:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x680: {}'.format(tuple_shapes))
        else:
            print('x680: {}'.format(x680))
        x681=self.batchnorm2d128(x680)
        if x681 is None:
            print('x681: {}'.format(x681))
        elif isinstance(x681, torch.Tensor):
            print('x681: {}'.format(x681.shape))
        elif isinstance(x681, tuple):
            tuple_shapes = '('
            for item in x681:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x681: {}'.format(tuple_shapes))
        else:
            print('x681: {}'.format(x681))
        x682=stochastic_depth(x681, 0.15636363636363634, 'row', False)
        if x682 is None:
            print('x682: {}'.format(x682))
        elif isinstance(x682, torch.Tensor):
            print('x682: {}'.format(x682.shape))
        elif isinstance(x682, tuple):
            tuple_shapes = '('
            for item in x682:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x682: {}'.format(tuple_shapes))
        else:
            print('x682: {}'.format(x682))
        x683=operator.add(x682, x667)
        if x683 is None:
            print('x683: {}'.format(x683))
        elif isinstance(x683, torch.Tensor):
            print('x683: {}'.format(x683.shape))
        elif isinstance(x683, tuple):
            tuple_shapes = '('
            for item in x683:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x683: {}'.format(tuple_shapes))
        else:
            print('x683: {}'.format(x683))
        x684=self.conv2d217(x683)
        if x684 is None:
            print('x684: {}'.format(x684))
        elif isinstance(x684, torch.Tensor):
            print('x684: {}'.format(x684.shape))
        elif isinstance(x684, tuple):
            tuple_shapes = '('
            for item in x684:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x684: {}'.format(tuple_shapes))
        else:
            print('x684: {}'.format(x684))
        x685=self.batchnorm2d129(x684)
        if x685 is None:
            print('x685: {}'.format(x685))
        elif isinstance(x685, torch.Tensor):
            print('x685: {}'.format(x685.shape))
        elif isinstance(x685, tuple):
            tuple_shapes = '('
            for item in x685:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x685: {}'.format(tuple_shapes))
        else:
            print('x685: {}'.format(x685))
        x686=self.silu129(x685)
        if x686 is None:
            print('x686: {}'.format(x686))
        elif isinstance(x686, torch.Tensor):
            print('x686: {}'.format(x686.shape))
        elif isinstance(x686, tuple):
            tuple_shapes = '('
            for item in x686:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x686: {}'.format(tuple_shapes))
        else:
            print('x686: {}'.format(x686))
        x687=self.conv2d218(x686)
        if x687 is None:
            print('x687: {}'.format(x687))
        elif isinstance(x687, torch.Tensor):
            print('x687: {}'.format(x687.shape))
        elif isinstance(x687, tuple):
            tuple_shapes = '('
            for item in x687:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x687: {}'.format(tuple_shapes))
        else:
            print('x687: {}'.format(x687))
        x688=self.batchnorm2d130(x687)
        if x688 is None:
            print('x688: {}'.format(x688))
        elif isinstance(x688, torch.Tensor):
            print('x688: {}'.format(x688.shape))
        elif isinstance(x688, tuple):
            tuple_shapes = '('
            for item in x688:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x688: {}'.format(tuple_shapes))
        else:
            print('x688: {}'.format(x688))
        x689=self.silu130(x688)
        if x689 is None:
            print('x689: {}'.format(x689))
        elif isinstance(x689, torch.Tensor):
            print('x689: {}'.format(x689.shape))
        elif isinstance(x689, tuple):
            tuple_shapes = '('
            for item in x689:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x689: {}'.format(tuple_shapes))
        else:
            print('x689: {}'.format(x689))
        x690=self.adaptiveavgpool2d44(x689)
        if x690 is None:
            print('x690: {}'.format(x690))
        elif isinstance(x690, torch.Tensor):
            print('x690: {}'.format(x690.shape))
        elif isinstance(x690, tuple):
            tuple_shapes = '('
            for item in x690:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x690: {}'.format(tuple_shapes))
        else:
            print('x690: {}'.format(x690))
        x691=self.conv2d219(x690)
        if x691 is None:
            print('x691: {}'.format(x691))
        elif isinstance(x691, torch.Tensor):
            print('x691: {}'.format(x691.shape))
        elif isinstance(x691, tuple):
            tuple_shapes = '('
            for item in x691:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x691: {}'.format(tuple_shapes))
        else:
            print('x691: {}'.format(x691))
        x692=self.silu131(x691)
        if x692 is None:
            print('x692: {}'.format(x692))
        elif isinstance(x692, torch.Tensor):
            print('x692: {}'.format(x692.shape))
        elif isinstance(x692, tuple):
            tuple_shapes = '('
            for item in x692:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x692: {}'.format(tuple_shapes))
        else:
            print('x692: {}'.format(x692))
        x693=self.conv2d220(x692)
        if x693 is None:
            print('x693: {}'.format(x693))
        elif isinstance(x693, torch.Tensor):
            print('x693: {}'.format(x693.shape))
        elif isinstance(x693, tuple):
            tuple_shapes = '('
            for item in x693:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x693: {}'.format(tuple_shapes))
        else:
            print('x693: {}'.format(x693))
        x694=self.sigmoid44(x693)
        if x694 is None:
            print('x694: {}'.format(x694))
        elif isinstance(x694, torch.Tensor):
            print('x694: {}'.format(x694.shape))
        elif isinstance(x694, tuple):
            tuple_shapes = '('
            for item in x694:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x694: {}'.format(tuple_shapes))
        else:
            print('x694: {}'.format(x694))
        x695=operator.mul(x694, x689)
        if x695 is None:
            print('x695: {}'.format(x695))
        elif isinstance(x695, torch.Tensor):
            print('x695: {}'.format(x695.shape))
        elif isinstance(x695, tuple):
            tuple_shapes = '('
            for item in x695:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x695: {}'.format(tuple_shapes))
        else:
            print('x695: {}'.format(x695))
        x696=self.conv2d221(x695)
        if x696 is None:
            print('x696: {}'.format(x696))
        elif isinstance(x696, torch.Tensor):
            print('x696: {}'.format(x696.shape))
        elif isinstance(x696, tuple):
            tuple_shapes = '('
            for item in x696:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x696: {}'.format(tuple_shapes))
        else:
            print('x696: {}'.format(x696))
        x697=self.batchnorm2d131(x696)
        if x697 is None:
            print('x697: {}'.format(x697))
        elif isinstance(x697, torch.Tensor):
            print('x697: {}'.format(x697.shape))
        elif isinstance(x697, tuple):
            tuple_shapes = '('
            for item in x697:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x697: {}'.format(tuple_shapes))
        else:
            print('x697: {}'.format(x697))
        x698=stochastic_depth(x697, 0.16, 'row', False)
        if x698 is None:
            print('x698: {}'.format(x698))
        elif isinstance(x698, torch.Tensor):
            print('x698: {}'.format(x698.shape))
        elif isinstance(x698, tuple):
            tuple_shapes = '('
            for item in x698:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x698: {}'.format(tuple_shapes))
        else:
            print('x698: {}'.format(x698))
        x699=operator.add(x698, x683)
        if x699 is None:
            print('x699: {}'.format(x699))
        elif isinstance(x699, torch.Tensor):
            print('x699: {}'.format(x699.shape))
        elif isinstance(x699, tuple):
            tuple_shapes = '('
            for item in x699:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x699: {}'.format(tuple_shapes))
        else:
            print('x699: {}'.format(x699))
        x700=self.conv2d222(x699)
        if x700 is None:
            print('x700: {}'.format(x700))
        elif isinstance(x700, torch.Tensor):
            print('x700: {}'.format(x700.shape))
        elif isinstance(x700, tuple):
            tuple_shapes = '('
            for item in x700:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x700: {}'.format(tuple_shapes))
        else:
            print('x700: {}'.format(x700))
        x701=self.batchnorm2d132(x700)
        if x701 is None:
            print('x701: {}'.format(x701))
        elif isinstance(x701, torch.Tensor):
            print('x701: {}'.format(x701.shape))
        elif isinstance(x701, tuple):
            tuple_shapes = '('
            for item in x701:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x701: {}'.format(tuple_shapes))
        else:
            print('x701: {}'.format(x701))
        x702=self.silu132(x701)
        if x702 is None:
            print('x702: {}'.format(x702))
        elif isinstance(x702, torch.Tensor):
            print('x702: {}'.format(x702.shape))
        elif isinstance(x702, tuple):
            tuple_shapes = '('
            for item in x702:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x702: {}'.format(tuple_shapes))
        else:
            print('x702: {}'.format(x702))
        x703=self.conv2d223(x702)
        if x703 is None:
            print('x703: {}'.format(x703))
        elif isinstance(x703, torch.Tensor):
            print('x703: {}'.format(x703.shape))
        elif isinstance(x703, tuple):
            tuple_shapes = '('
            for item in x703:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x703: {}'.format(tuple_shapes))
        else:
            print('x703: {}'.format(x703))
        x704=self.batchnorm2d133(x703)
        if x704 is None:
            print('x704: {}'.format(x704))
        elif isinstance(x704, torch.Tensor):
            print('x704: {}'.format(x704.shape))
        elif isinstance(x704, tuple):
            tuple_shapes = '('
            for item in x704:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x704: {}'.format(tuple_shapes))
        else:
            print('x704: {}'.format(x704))
        x705=self.silu133(x704)
        if x705 is None:
            print('x705: {}'.format(x705))
        elif isinstance(x705, torch.Tensor):
            print('x705: {}'.format(x705.shape))
        elif isinstance(x705, tuple):
            tuple_shapes = '('
            for item in x705:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x705: {}'.format(tuple_shapes))
        else:
            print('x705: {}'.format(x705))
        x706=self.adaptiveavgpool2d45(x705)
        if x706 is None:
            print('x706: {}'.format(x706))
        elif isinstance(x706, torch.Tensor):
            print('x706: {}'.format(x706.shape))
        elif isinstance(x706, tuple):
            tuple_shapes = '('
            for item in x706:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x706: {}'.format(tuple_shapes))
        else:
            print('x706: {}'.format(x706))
        x707=self.conv2d224(x706)
        if x707 is None:
            print('x707: {}'.format(x707))
        elif isinstance(x707, torch.Tensor):
            print('x707: {}'.format(x707.shape))
        elif isinstance(x707, tuple):
            tuple_shapes = '('
            for item in x707:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x707: {}'.format(tuple_shapes))
        else:
            print('x707: {}'.format(x707))
        x708=self.silu134(x707)
        if x708 is None:
            print('x708: {}'.format(x708))
        elif isinstance(x708, torch.Tensor):
            print('x708: {}'.format(x708.shape))
        elif isinstance(x708, tuple):
            tuple_shapes = '('
            for item in x708:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x708: {}'.format(tuple_shapes))
        else:
            print('x708: {}'.format(x708))
        x709=self.conv2d225(x708)
        if x709 is None:
            print('x709: {}'.format(x709))
        elif isinstance(x709, torch.Tensor):
            print('x709: {}'.format(x709.shape))
        elif isinstance(x709, tuple):
            tuple_shapes = '('
            for item in x709:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x709: {}'.format(tuple_shapes))
        else:
            print('x709: {}'.format(x709))
        x710=self.sigmoid45(x709)
        if x710 is None:
            print('x710: {}'.format(x710))
        elif isinstance(x710, torch.Tensor):
            print('x710: {}'.format(x710.shape))
        elif isinstance(x710, tuple):
            tuple_shapes = '('
            for item in x710:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x710: {}'.format(tuple_shapes))
        else:
            print('x710: {}'.format(x710))
        x711=operator.mul(x710, x705)
        if x711 is None:
            print('x711: {}'.format(x711))
        elif isinstance(x711, torch.Tensor):
            print('x711: {}'.format(x711.shape))
        elif isinstance(x711, tuple):
            tuple_shapes = '('
            for item in x711:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x711: {}'.format(tuple_shapes))
        else:
            print('x711: {}'.format(x711))
        x712=self.conv2d226(x711)
        if x712 is None:
            print('x712: {}'.format(x712))
        elif isinstance(x712, torch.Tensor):
            print('x712: {}'.format(x712.shape))
        elif isinstance(x712, tuple):
            tuple_shapes = '('
            for item in x712:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x712: {}'.format(tuple_shapes))
        else:
            print('x712: {}'.format(x712))
        x713=self.batchnorm2d134(x712)
        if x713 is None:
            print('x713: {}'.format(x713))
        elif isinstance(x713, torch.Tensor):
            print('x713: {}'.format(x713.shape))
        elif isinstance(x713, tuple):
            tuple_shapes = '('
            for item in x713:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x713: {}'.format(tuple_shapes))
        else:
            print('x713: {}'.format(x713))
        x714=stochastic_depth(x713, 0.16363636363636364, 'row', False)
        if x714 is None:
            print('x714: {}'.format(x714))
        elif isinstance(x714, torch.Tensor):
            print('x714: {}'.format(x714.shape))
        elif isinstance(x714, tuple):
            tuple_shapes = '('
            for item in x714:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x714: {}'.format(tuple_shapes))
        else:
            print('x714: {}'.format(x714))
        x715=operator.add(x714, x699)
        if x715 is None:
            print('x715: {}'.format(x715))
        elif isinstance(x715, torch.Tensor):
            print('x715: {}'.format(x715.shape))
        elif isinstance(x715, tuple):
            tuple_shapes = '('
            for item in x715:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x715: {}'.format(tuple_shapes))
        else:
            print('x715: {}'.format(x715))
        x716=self.conv2d227(x715)
        if x716 is None:
            print('x716: {}'.format(x716))
        elif isinstance(x716, torch.Tensor):
            print('x716: {}'.format(x716.shape))
        elif isinstance(x716, tuple):
            tuple_shapes = '('
            for item in x716:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x716: {}'.format(tuple_shapes))
        else:
            print('x716: {}'.format(x716))
        x717=self.batchnorm2d135(x716)
        if x717 is None:
            print('x717: {}'.format(x717))
        elif isinstance(x717, torch.Tensor):
            print('x717: {}'.format(x717.shape))
        elif isinstance(x717, tuple):
            tuple_shapes = '('
            for item in x717:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x717: {}'.format(tuple_shapes))
        else:
            print('x717: {}'.format(x717))
        x718=self.silu135(x717)
        if x718 is None:
            print('x718: {}'.format(x718))
        elif isinstance(x718, torch.Tensor):
            print('x718: {}'.format(x718.shape))
        elif isinstance(x718, tuple):
            tuple_shapes = '('
            for item in x718:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x718: {}'.format(tuple_shapes))
        else:
            print('x718: {}'.format(x718))
        x719=self.conv2d228(x718)
        if x719 is None:
            print('x719: {}'.format(x719))
        elif isinstance(x719, torch.Tensor):
            print('x719: {}'.format(x719.shape))
        elif isinstance(x719, tuple):
            tuple_shapes = '('
            for item in x719:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x719: {}'.format(tuple_shapes))
        else:
            print('x719: {}'.format(x719))
        x720=self.batchnorm2d136(x719)
        if x720 is None:
            print('x720: {}'.format(x720))
        elif isinstance(x720, torch.Tensor):
            print('x720: {}'.format(x720.shape))
        elif isinstance(x720, tuple):
            tuple_shapes = '('
            for item in x720:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x720: {}'.format(tuple_shapes))
        else:
            print('x720: {}'.format(x720))
        x721=self.silu136(x720)
        if x721 is None:
            print('x721: {}'.format(x721))
        elif isinstance(x721, torch.Tensor):
            print('x721: {}'.format(x721.shape))
        elif isinstance(x721, tuple):
            tuple_shapes = '('
            for item in x721:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x721: {}'.format(tuple_shapes))
        else:
            print('x721: {}'.format(x721))
        x722=self.adaptiveavgpool2d46(x721)
        if x722 is None:
            print('x722: {}'.format(x722))
        elif isinstance(x722, torch.Tensor):
            print('x722: {}'.format(x722.shape))
        elif isinstance(x722, tuple):
            tuple_shapes = '('
            for item in x722:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x722: {}'.format(tuple_shapes))
        else:
            print('x722: {}'.format(x722))
        x723=self.conv2d229(x722)
        if x723 is None:
            print('x723: {}'.format(x723))
        elif isinstance(x723, torch.Tensor):
            print('x723: {}'.format(x723.shape))
        elif isinstance(x723, tuple):
            tuple_shapes = '('
            for item in x723:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x723: {}'.format(tuple_shapes))
        else:
            print('x723: {}'.format(x723))
        x724=self.silu137(x723)
        if x724 is None:
            print('x724: {}'.format(x724))
        elif isinstance(x724, torch.Tensor):
            print('x724: {}'.format(x724.shape))
        elif isinstance(x724, tuple):
            tuple_shapes = '('
            for item in x724:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x724: {}'.format(tuple_shapes))
        else:
            print('x724: {}'.format(x724))
        x725=self.conv2d230(x724)
        if x725 is None:
            print('x725: {}'.format(x725))
        elif isinstance(x725, torch.Tensor):
            print('x725: {}'.format(x725.shape))
        elif isinstance(x725, tuple):
            tuple_shapes = '('
            for item in x725:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x725: {}'.format(tuple_shapes))
        else:
            print('x725: {}'.format(x725))
        x726=self.sigmoid46(x725)
        if x726 is None:
            print('x726: {}'.format(x726))
        elif isinstance(x726, torch.Tensor):
            print('x726: {}'.format(x726.shape))
        elif isinstance(x726, tuple):
            tuple_shapes = '('
            for item in x726:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x726: {}'.format(tuple_shapes))
        else:
            print('x726: {}'.format(x726))
        x727=operator.mul(x726, x721)
        if x727 is None:
            print('x727: {}'.format(x727))
        elif isinstance(x727, torch.Tensor):
            print('x727: {}'.format(x727.shape))
        elif isinstance(x727, tuple):
            tuple_shapes = '('
            for item in x727:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x727: {}'.format(tuple_shapes))
        else:
            print('x727: {}'.format(x727))
        x728=self.conv2d231(x727)
        if x728 is None:
            print('x728: {}'.format(x728))
        elif isinstance(x728, torch.Tensor):
            print('x728: {}'.format(x728.shape))
        elif isinstance(x728, tuple):
            tuple_shapes = '('
            for item in x728:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x728: {}'.format(tuple_shapes))
        else:
            print('x728: {}'.format(x728))
        x729=self.batchnorm2d137(x728)
        if x729 is None:
            print('x729: {}'.format(x729))
        elif isinstance(x729, torch.Tensor):
            print('x729: {}'.format(x729.shape))
        elif isinstance(x729, tuple):
            tuple_shapes = '('
            for item in x729:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x729: {}'.format(tuple_shapes))
        else:
            print('x729: {}'.format(x729))
        x730=stochastic_depth(x729, 0.1672727272727273, 'row', False)
        if x730 is None:
            print('x730: {}'.format(x730))
        elif isinstance(x730, torch.Tensor):
            print('x730: {}'.format(x730.shape))
        elif isinstance(x730, tuple):
            tuple_shapes = '('
            for item in x730:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x730: {}'.format(tuple_shapes))
        else:
            print('x730: {}'.format(x730))
        x731=operator.add(x730, x715)
        if x731 is None:
            print('x731: {}'.format(x731))
        elif isinstance(x731, torch.Tensor):
            print('x731: {}'.format(x731.shape))
        elif isinstance(x731, tuple):
            tuple_shapes = '('
            for item in x731:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x731: {}'.format(tuple_shapes))
        else:
            print('x731: {}'.format(x731))
        x732=self.conv2d232(x731)
        if x732 is None:
            print('x732: {}'.format(x732))
        elif isinstance(x732, torch.Tensor):
            print('x732: {}'.format(x732.shape))
        elif isinstance(x732, tuple):
            tuple_shapes = '('
            for item in x732:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x732: {}'.format(tuple_shapes))
        else:
            print('x732: {}'.format(x732))
        x733=self.batchnorm2d138(x732)
        if x733 is None:
            print('x733: {}'.format(x733))
        elif isinstance(x733, torch.Tensor):
            print('x733: {}'.format(x733.shape))
        elif isinstance(x733, tuple):
            tuple_shapes = '('
            for item in x733:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x733: {}'.format(tuple_shapes))
        else:
            print('x733: {}'.format(x733))
        x734=self.silu138(x733)
        if x734 is None:
            print('x734: {}'.format(x734))
        elif isinstance(x734, torch.Tensor):
            print('x734: {}'.format(x734.shape))
        elif isinstance(x734, tuple):
            tuple_shapes = '('
            for item in x734:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x734: {}'.format(tuple_shapes))
        else:
            print('x734: {}'.format(x734))
        x735=self.conv2d233(x734)
        if x735 is None:
            print('x735: {}'.format(x735))
        elif isinstance(x735, torch.Tensor):
            print('x735: {}'.format(x735.shape))
        elif isinstance(x735, tuple):
            tuple_shapes = '('
            for item in x735:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x735: {}'.format(tuple_shapes))
        else:
            print('x735: {}'.format(x735))
        x736=self.batchnorm2d139(x735)
        if x736 is None:
            print('x736: {}'.format(x736))
        elif isinstance(x736, torch.Tensor):
            print('x736: {}'.format(x736.shape))
        elif isinstance(x736, tuple):
            tuple_shapes = '('
            for item in x736:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x736: {}'.format(tuple_shapes))
        else:
            print('x736: {}'.format(x736))
        x737=self.silu139(x736)
        if x737 is None:
            print('x737: {}'.format(x737))
        elif isinstance(x737, torch.Tensor):
            print('x737: {}'.format(x737.shape))
        elif isinstance(x737, tuple):
            tuple_shapes = '('
            for item in x737:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x737: {}'.format(tuple_shapes))
        else:
            print('x737: {}'.format(x737))
        x738=self.adaptiveavgpool2d47(x737)
        if x738 is None:
            print('x738: {}'.format(x738))
        elif isinstance(x738, torch.Tensor):
            print('x738: {}'.format(x738.shape))
        elif isinstance(x738, tuple):
            tuple_shapes = '('
            for item in x738:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x738: {}'.format(tuple_shapes))
        else:
            print('x738: {}'.format(x738))
        x739=self.conv2d234(x738)
        if x739 is None:
            print('x739: {}'.format(x739))
        elif isinstance(x739, torch.Tensor):
            print('x739: {}'.format(x739.shape))
        elif isinstance(x739, tuple):
            tuple_shapes = '('
            for item in x739:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x739: {}'.format(tuple_shapes))
        else:
            print('x739: {}'.format(x739))
        x740=self.silu140(x739)
        if x740 is None:
            print('x740: {}'.format(x740))
        elif isinstance(x740, torch.Tensor):
            print('x740: {}'.format(x740.shape))
        elif isinstance(x740, tuple):
            tuple_shapes = '('
            for item in x740:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x740: {}'.format(tuple_shapes))
        else:
            print('x740: {}'.format(x740))
        x741=self.conv2d235(x740)
        if x741 is None:
            print('x741: {}'.format(x741))
        elif isinstance(x741, torch.Tensor):
            print('x741: {}'.format(x741.shape))
        elif isinstance(x741, tuple):
            tuple_shapes = '('
            for item in x741:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x741: {}'.format(tuple_shapes))
        else:
            print('x741: {}'.format(x741))
        x742=self.sigmoid47(x741)
        if x742 is None:
            print('x742: {}'.format(x742))
        elif isinstance(x742, torch.Tensor):
            print('x742: {}'.format(x742.shape))
        elif isinstance(x742, tuple):
            tuple_shapes = '('
            for item in x742:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x742: {}'.format(tuple_shapes))
        else:
            print('x742: {}'.format(x742))
        x743=operator.mul(x742, x737)
        if x743 is None:
            print('x743: {}'.format(x743))
        elif isinstance(x743, torch.Tensor):
            print('x743: {}'.format(x743.shape))
        elif isinstance(x743, tuple):
            tuple_shapes = '('
            for item in x743:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x743: {}'.format(tuple_shapes))
        else:
            print('x743: {}'.format(x743))
        x744=self.conv2d236(x743)
        if x744 is None:
            print('x744: {}'.format(x744))
        elif isinstance(x744, torch.Tensor):
            print('x744: {}'.format(x744.shape))
        elif isinstance(x744, tuple):
            tuple_shapes = '('
            for item in x744:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x744: {}'.format(tuple_shapes))
        else:
            print('x744: {}'.format(x744))
        x745=self.batchnorm2d140(x744)
        if x745 is None:
            print('x745: {}'.format(x745))
        elif isinstance(x745, torch.Tensor):
            print('x745: {}'.format(x745.shape))
        elif isinstance(x745, tuple):
            tuple_shapes = '('
            for item in x745:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x745: {}'.format(tuple_shapes))
        else:
            print('x745: {}'.format(x745))
        x746=stochastic_depth(x745, 0.17090909090909093, 'row', False)
        if x746 is None:
            print('x746: {}'.format(x746))
        elif isinstance(x746, torch.Tensor):
            print('x746: {}'.format(x746.shape))
        elif isinstance(x746, tuple):
            tuple_shapes = '('
            for item in x746:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x746: {}'.format(tuple_shapes))
        else:
            print('x746: {}'.format(x746))
        x747=operator.add(x746, x731)
        if x747 is None:
            print('x747: {}'.format(x747))
        elif isinstance(x747, torch.Tensor):
            print('x747: {}'.format(x747.shape))
        elif isinstance(x747, tuple):
            tuple_shapes = '('
            for item in x747:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x747: {}'.format(tuple_shapes))
        else:
            print('x747: {}'.format(x747))
        x748=self.conv2d237(x747)
        if x748 is None:
            print('x748: {}'.format(x748))
        elif isinstance(x748, torch.Tensor):
            print('x748: {}'.format(x748.shape))
        elif isinstance(x748, tuple):
            tuple_shapes = '('
            for item in x748:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x748: {}'.format(tuple_shapes))
        else:
            print('x748: {}'.format(x748))
        x749=self.batchnorm2d141(x748)
        if x749 is None:
            print('x749: {}'.format(x749))
        elif isinstance(x749, torch.Tensor):
            print('x749: {}'.format(x749.shape))
        elif isinstance(x749, tuple):
            tuple_shapes = '('
            for item in x749:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x749: {}'.format(tuple_shapes))
        else:
            print('x749: {}'.format(x749))
        x750=self.silu141(x749)
        if x750 is None:
            print('x750: {}'.format(x750))
        elif isinstance(x750, torch.Tensor):
            print('x750: {}'.format(x750.shape))
        elif isinstance(x750, tuple):
            tuple_shapes = '('
            for item in x750:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x750: {}'.format(tuple_shapes))
        else:
            print('x750: {}'.format(x750))
        x751=self.conv2d238(x750)
        if x751 is None:
            print('x751: {}'.format(x751))
        elif isinstance(x751, torch.Tensor):
            print('x751: {}'.format(x751.shape))
        elif isinstance(x751, tuple):
            tuple_shapes = '('
            for item in x751:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x751: {}'.format(tuple_shapes))
        else:
            print('x751: {}'.format(x751))
        x752=self.batchnorm2d142(x751)
        if x752 is None:
            print('x752: {}'.format(x752))
        elif isinstance(x752, torch.Tensor):
            print('x752: {}'.format(x752.shape))
        elif isinstance(x752, tuple):
            tuple_shapes = '('
            for item in x752:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x752: {}'.format(tuple_shapes))
        else:
            print('x752: {}'.format(x752))
        x753=self.silu142(x752)
        if x753 is None:
            print('x753: {}'.format(x753))
        elif isinstance(x753, torch.Tensor):
            print('x753: {}'.format(x753.shape))
        elif isinstance(x753, tuple):
            tuple_shapes = '('
            for item in x753:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x753: {}'.format(tuple_shapes))
        else:
            print('x753: {}'.format(x753))
        x754=self.adaptiveavgpool2d48(x753)
        if x754 is None:
            print('x754: {}'.format(x754))
        elif isinstance(x754, torch.Tensor):
            print('x754: {}'.format(x754.shape))
        elif isinstance(x754, tuple):
            tuple_shapes = '('
            for item in x754:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x754: {}'.format(tuple_shapes))
        else:
            print('x754: {}'.format(x754))
        x755=self.conv2d239(x754)
        if x755 is None:
            print('x755: {}'.format(x755))
        elif isinstance(x755, torch.Tensor):
            print('x755: {}'.format(x755.shape))
        elif isinstance(x755, tuple):
            tuple_shapes = '('
            for item in x755:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x755: {}'.format(tuple_shapes))
        else:
            print('x755: {}'.format(x755))
        x756=self.silu143(x755)
        if x756 is None:
            print('x756: {}'.format(x756))
        elif isinstance(x756, torch.Tensor):
            print('x756: {}'.format(x756.shape))
        elif isinstance(x756, tuple):
            tuple_shapes = '('
            for item in x756:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x756: {}'.format(tuple_shapes))
        else:
            print('x756: {}'.format(x756))
        x757=self.conv2d240(x756)
        if x757 is None:
            print('x757: {}'.format(x757))
        elif isinstance(x757, torch.Tensor):
            print('x757: {}'.format(x757.shape))
        elif isinstance(x757, tuple):
            tuple_shapes = '('
            for item in x757:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x757: {}'.format(tuple_shapes))
        else:
            print('x757: {}'.format(x757))
        x758=self.sigmoid48(x757)
        if x758 is None:
            print('x758: {}'.format(x758))
        elif isinstance(x758, torch.Tensor):
            print('x758: {}'.format(x758.shape))
        elif isinstance(x758, tuple):
            tuple_shapes = '('
            for item in x758:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x758: {}'.format(tuple_shapes))
        else:
            print('x758: {}'.format(x758))
        x759=operator.mul(x758, x753)
        if x759 is None:
            print('x759: {}'.format(x759))
        elif isinstance(x759, torch.Tensor):
            print('x759: {}'.format(x759.shape))
        elif isinstance(x759, tuple):
            tuple_shapes = '('
            for item in x759:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x759: {}'.format(tuple_shapes))
        else:
            print('x759: {}'.format(x759))
        x760=self.conv2d241(x759)
        if x760 is None:
            print('x760: {}'.format(x760))
        elif isinstance(x760, torch.Tensor):
            print('x760: {}'.format(x760.shape))
        elif isinstance(x760, tuple):
            tuple_shapes = '('
            for item in x760:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x760: {}'.format(tuple_shapes))
        else:
            print('x760: {}'.format(x760))
        x761=self.batchnorm2d143(x760)
        if x761 is None:
            print('x761: {}'.format(x761))
        elif isinstance(x761, torch.Tensor):
            print('x761: {}'.format(x761.shape))
        elif isinstance(x761, tuple):
            tuple_shapes = '('
            for item in x761:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x761: {}'.format(tuple_shapes))
        else:
            print('x761: {}'.format(x761))
        x762=stochastic_depth(x761, 0.17454545454545456, 'row', False)
        if x762 is None:
            print('x762: {}'.format(x762))
        elif isinstance(x762, torch.Tensor):
            print('x762: {}'.format(x762.shape))
        elif isinstance(x762, tuple):
            tuple_shapes = '('
            for item in x762:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x762: {}'.format(tuple_shapes))
        else:
            print('x762: {}'.format(x762))
        x763=operator.add(x762, x747)
        if x763 is None:
            print('x763: {}'.format(x763))
        elif isinstance(x763, torch.Tensor):
            print('x763: {}'.format(x763.shape))
        elif isinstance(x763, tuple):
            tuple_shapes = '('
            for item in x763:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x763: {}'.format(tuple_shapes))
        else:
            print('x763: {}'.format(x763))
        x764=self.conv2d242(x763)
        if x764 is None:
            print('x764: {}'.format(x764))
        elif isinstance(x764, torch.Tensor):
            print('x764: {}'.format(x764.shape))
        elif isinstance(x764, tuple):
            tuple_shapes = '('
            for item in x764:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x764: {}'.format(tuple_shapes))
        else:
            print('x764: {}'.format(x764))
        x765=self.batchnorm2d144(x764)
        if x765 is None:
            print('x765: {}'.format(x765))
        elif isinstance(x765, torch.Tensor):
            print('x765: {}'.format(x765.shape))
        elif isinstance(x765, tuple):
            tuple_shapes = '('
            for item in x765:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x765: {}'.format(tuple_shapes))
        else:
            print('x765: {}'.format(x765))
        x766=self.silu144(x765)
        if x766 is None:
            print('x766: {}'.format(x766))
        elif isinstance(x766, torch.Tensor):
            print('x766: {}'.format(x766.shape))
        elif isinstance(x766, tuple):
            tuple_shapes = '('
            for item in x766:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x766: {}'.format(tuple_shapes))
        else:
            print('x766: {}'.format(x766))
        x767=self.conv2d243(x766)
        if x767 is None:
            print('x767: {}'.format(x767))
        elif isinstance(x767, torch.Tensor):
            print('x767: {}'.format(x767.shape))
        elif isinstance(x767, tuple):
            tuple_shapes = '('
            for item in x767:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x767: {}'.format(tuple_shapes))
        else:
            print('x767: {}'.format(x767))
        x768=self.batchnorm2d145(x767)
        if x768 is None:
            print('x768: {}'.format(x768))
        elif isinstance(x768, torch.Tensor):
            print('x768: {}'.format(x768.shape))
        elif isinstance(x768, tuple):
            tuple_shapes = '('
            for item in x768:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x768: {}'.format(tuple_shapes))
        else:
            print('x768: {}'.format(x768))
        x769=self.silu145(x768)
        if x769 is None:
            print('x769: {}'.format(x769))
        elif isinstance(x769, torch.Tensor):
            print('x769: {}'.format(x769.shape))
        elif isinstance(x769, tuple):
            tuple_shapes = '('
            for item in x769:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x769: {}'.format(tuple_shapes))
        else:
            print('x769: {}'.format(x769))
        x770=self.adaptiveavgpool2d49(x769)
        if x770 is None:
            print('x770: {}'.format(x770))
        elif isinstance(x770, torch.Tensor):
            print('x770: {}'.format(x770.shape))
        elif isinstance(x770, tuple):
            tuple_shapes = '('
            for item in x770:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x770: {}'.format(tuple_shapes))
        else:
            print('x770: {}'.format(x770))
        x771=self.conv2d244(x770)
        if x771 is None:
            print('x771: {}'.format(x771))
        elif isinstance(x771, torch.Tensor):
            print('x771: {}'.format(x771.shape))
        elif isinstance(x771, tuple):
            tuple_shapes = '('
            for item in x771:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x771: {}'.format(tuple_shapes))
        else:
            print('x771: {}'.format(x771))
        x772=self.silu146(x771)
        if x772 is None:
            print('x772: {}'.format(x772))
        elif isinstance(x772, torch.Tensor):
            print('x772: {}'.format(x772.shape))
        elif isinstance(x772, tuple):
            tuple_shapes = '('
            for item in x772:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x772: {}'.format(tuple_shapes))
        else:
            print('x772: {}'.format(x772))
        x773=self.conv2d245(x772)
        if x773 is None:
            print('x773: {}'.format(x773))
        elif isinstance(x773, torch.Tensor):
            print('x773: {}'.format(x773.shape))
        elif isinstance(x773, tuple):
            tuple_shapes = '('
            for item in x773:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x773: {}'.format(tuple_shapes))
        else:
            print('x773: {}'.format(x773))
        x774=self.sigmoid49(x773)
        if x774 is None:
            print('x774: {}'.format(x774))
        elif isinstance(x774, torch.Tensor):
            print('x774: {}'.format(x774.shape))
        elif isinstance(x774, tuple):
            tuple_shapes = '('
            for item in x774:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x774: {}'.format(tuple_shapes))
        else:
            print('x774: {}'.format(x774))
        x775=operator.mul(x774, x769)
        if x775 is None:
            print('x775: {}'.format(x775))
        elif isinstance(x775, torch.Tensor):
            print('x775: {}'.format(x775.shape))
        elif isinstance(x775, tuple):
            tuple_shapes = '('
            for item in x775:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x775: {}'.format(tuple_shapes))
        else:
            print('x775: {}'.format(x775))
        x776=self.conv2d246(x775)
        if x776 is None:
            print('x776: {}'.format(x776))
        elif isinstance(x776, torch.Tensor):
            print('x776: {}'.format(x776.shape))
        elif isinstance(x776, tuple):
            tuple_shapes = '('
            for item in x776:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x776: {}'.format(tuple_shapes))
        else:
            print('x776: {}'.format(x776))
        x777=self.batchnorm2d146(x776)
        if x777 is None:
            print('x777: {}'.format(x777))
        elif isinstance(x777, torch.Tensor):
            print('x777: {}'.format(x777.shape))
        elif isinstance(x777, tuple):
            tuple_shapes = '('
            for item in x777:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x777: {}'.format(tuple_shapes))
        else:
            print('x777: {}'.format(x777))
        x778=stochastic_depth(x777, 0.1781818181818182, 'row', False)
        if x778 is None:
            print('x778: {}'.format(x778))
        elif isinstance(x778, torch.Tensor):
            print('x778: {}'.format(x778.shape))
        elif isinstance(x778, tuple):
            tuple_shapes = '('
            for item in x778:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x778: {}'.format(tuple_shapes))
        else:
            print('x778: {}'.format(x778))
        x779=operator.add(x778, x763)
        if x779 is None:
            print('x779: {}'.format(x779))
        elif isinstance(x779, torch.Tensor):
            print('x779: {}'.format(x779.shape))
        elif isinstance(x779, tuple):
            tuple_shapes = '('
            for item in x779:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x779: {}'.format(tuple_shapes))
        else:
            print('x779: {}'.format(x779))
        x780=self.conv2d247(x779)
        if x780 is None:
            print('x780: {}'.format(x780))
        elif isinstance(x780, torch.Tensor):
            print('x780: {}'.format(x780.shape))
        elif isinstance(x780, tuple):
            tuple_shapes = '('
            for item in x780:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x780: {}'.format(tuple_shapes))
        else:
            print('x780: {}'.format(x780))
        x781=self.batchnorm2d147(x780)
        if x781 is None:
            print('x781: {}'.format(x781))
        elif isinstance(x781, torch.Tensor):
            print('x781: {}'.format(x781.shape))
        elif isinstance(x781, tuple):
            tuple_shapes = '('
            for item in x781:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x781: {}'.format(tuple_shapes))
        else:
            print('x781: {}'.format(x781))
        x782=self.silu147(x781)
        if x782 is None:
            print('x782: {}'.format(x782))
        elif isinstance(x782, torch.Tensor):
            print('x782: {}'.format(x782.shape))
        elif isinstance(x782, tuple):
            tuple_shapes = '('
            for item in x782:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x782: {}'.format(tuple_shapes))
        else:
            print('x782: {}'.format(x782))
        x783=self.conv2d248(x782)
        if x783 is None:
            print('x783: {}'.format(x783))
        elif isinstance(x783, torch.Tensor):
            print('x783: {}'.format(x783.shape))
        elif isinstance(x783, tuple):
            tuple_shapes = '('
            for item in x783:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x783: {}'.format(tuple_shapes))
        else:
            print('x783: {}'.format(x783))
        x784=self.batchnorm2d148(x783)
        if x784 is None:
            print('x784: {}'.format(x784))
        elif isinstance(x784, torch.Tensor):
            print('x784: {}'.format(x784.shape))
        elif isinstance(x784, tuple):
            tuple_shapes = '('
            for item in x784:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x784: {}'.format(tuple_shapes))
        else:
            print('x784: {}'.format(x784))
        x785=self.silu148(x784)
        if x785 is None:
            print('x785: {}'.format(x785))
        elif isinstance(x785, torch.Tensor):
            print('x785: {}'.format(x785.shape))
        elif isinstance(x785, tuple):
            tuple_shapes = '('
            for item in x785:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x785: {}'.format(tuple_shapes))
        else:
            print('x785: {}'.format(x785))
        x786=self.adaptiveavgpool2d50(x785)
        if x786 is None:
            print('x786: {}'.format(x786))
        elif isinstance(x786, torch.Tensor):
            print('x786: {}'.format(x786.shape))
        elif isinstance(x786, tuple):
            tuple_shapes = '('
            for item in x786:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x786: {}'.format(tuple_shapes))
        else:
            print('x786: {}'.format(x786))
        x787=self.conv2d249(x786)
        if x787 is None:
            print('x787: {}'.format(x787))
        elif isinstance(x787, torch.Tensor):
            print('x787: {}'.format(x787.shape))
        elif isinstance(x787, tuple):
            tuple_shapes = '('
            for item in x787:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x787: {}'.format(tuple_shapes))
        else:
            print('x787: {}'.format(x787))
        x788=self.silu149(x787)
        if x788 is None:
            print('x788: {}'.format(x788))
        elif isinstance(x788, torch.Tensor):
            print('x788: {}'.format(x788.shape))
        elif isinstance(x788, tuple):
            tuple_shapes = '('
            for item in x788:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x788: {}'.format(tuple_shapes))
        else:
            print('x788: {}'.format(x788))
        x789=self.conv2d250(x788)
        if x789 is None:
            print('x789: {}'.format(x789))
        elif isinstance(x789, torch.Tensor):
            print('x789: {}'.format(x789.shape))
        elif isinstance(x789, tuple):
            tuple_shapes = '('
            for item in x789:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x789: {}'.format(tuple_shapes))
        else:
            print('x789: {}'.format(x789))
        x790=self.sigmoid50(x789)
        if x790 is None:
            print('x790: {}'.format(x790))
        elif isinstance(x790, torch.Tensor):
            print('x790: {}'.format(x790.shape))
        elif isinstance(x790, tuple):
            tuple_shapes = '('
            for item in x790:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x790: {}'.format(tuple_shapes))
        else:
            print('x790: {}'.format(x790))
        x791=operator.mul(x790, x785)
        if x791 is None:
            print('x791: {}'.format(x791))
        elif isinstance(x791, torch.Tensor):
            print('x791: {}'.format(x791.shape))
        elif isinstance(x791, tuple):
            tuple_shapes = '('
            for item in x791:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x791: {}'.format(tuple_shapes))
        else:
            print('x791: {}'.format(x791))
        x792=self.conv2d251(x791)
        if x792 is None:
            print('x792: {}'.format(x792))
        elif isinstance(x792, torch.Tensor):
            print('x792: {}'.format(x792.shape))
        elif isinstance(x792, tuple):
            tuple_shapes = '('
            for item in x792:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x792: {}'.format(tuple_shapes))
        else:
            print('x792: {}'.format(x792))
        x793=self.batchnorm2d149(x792)
        if x793 is None:
            print('x793: {}'.format(x793))
        elif isinstance(x793, torch.Tensor):
            print('x793: {}'.format(x793.shape))
        elif isinstance(x793, tuple):
            tuple_shapes = '('
            for item in x793:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x793: {}'.format(tuple_shapes))
        else:
            print('x793: {}'.format(x793))
        x794=stochastic_depth(x793, 0.18181818181818182, 'row', False)
        if x794 is None:
            print('x794: {}'.format(x794))
        elif isinstance(x794, torch.Tensor):
            print('x794: {}'.format(x794.shape))
        elif isinstance(x794, tuple):
            tuple_shapes = '('
            for item in x794:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x794: {}'.format(tuple_shapes))
        else:
            print('x794: {}'.format(x794))
        x795=operator.add(x794, x779)
        if x795 is None:
            print('x795: {}'.format(x795))
        elif isinstance(x795, torch.Tensor):
            print('x795: {}'.format(x795.shape))
        elif isinstance(x795, tuple):
            tuple_shapes = '('
            for item in x795:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x795: {}'.format(tuple_shapes))
        else:
            print('x795: {}'.format(x795))
        x796=self.conv2d252(x795)
        if x796 is None:
            print('x796: {}'.format(x796))
        elif isinstance(x796, torch.Tensor):
            print('x796: {}'.format(x796.shape))
        elif isinstance(x796, tuple):
            tuple_shapes = '('
            for item in x796:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x796: {}'.format(tuple_shapes))
        else:
            print('x796: {}'.format(x796))
        x797=self.batchnorm2d150(x796)
        if x797 is None:
            print('x797: {}'.format(x797))
        elif isinstance(x797, torch.Tensor):
            print('x797: {}'.format(x797.shape))
        elif isinstance(x797, tuple):
            tuple_shapes = '('
            for item in x797:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x797: {}'.format(tuple_shapes))
        else:
            print('x797: {}'.format(x797))
        x798=self.silu150(x797)
        if x798 is None:
            print('x798: {}'.format(x798))
        elif isinstance(x798, torch.Tensor):
            print('x798: {}'.format(x798.shape))
        elif isinstance(x798, tuple):
            tuple_shapes = '('
            for item in x798:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x798: {}'.format(tuple_shapes))
        else:
            print('x798: {}'.format(x798))
        x799=self.conv2d253(x798)
        if x799 is None:
            print('x799: {}'.format(x799))
        elif isinstance(x799, torch.Tensor):
            print('x799: {}'.format(x799.shape))
        elif isinstance(x799, tuple):
            tuple_shapes = '('
            for item in x799:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x799: {}'.format(tuple_shapes))
        else:
            print('x799: {}'.format(x799))
        x800=self.batchnorm2d151(x799)
        if x800 is None:
            print('x800: {}'.format(x800))
        elif isinstance(x800, torch.Tensor):
            print('x800: {}'.format(x800.shape))
        elif isinstance(x800, tuple):
            tuple_shapes = '('
            for item in x800:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x800: {}'.format(tuple_shapes))
        else:
            print('x800: {}'.format(x800))
        x801=self.silu151(x800)
        if x801 is None:
            print('x801: {}'.format(x801))
        elif isinstance(x801, torch.Tensor):
            print('x801: {}'.format(x801.shape))
        elif isinstance(x801, tuple):
            tuple_shapes = '('
            for item in x801:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x801: {}'.format(tuple_shapes))
        else:
            print('x801: {}'.format(x801))
        x802=self.adaptiveavgpool2d51(x801)
        if x802 is None:
            print('x802: {}'.format(x802))
        elif isinstance(x802, torch.Tensor):
            print('x802: {}'.format(x802.shape))
        elif isinstance(x802, tuple):
            tuple_shapes = '('
            for item in x802:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x802: {}'.format(tuple_shapes))
        else:
            print('x802: {}'.format(x802))
        x803=self.conv2d254(x802)
        if x803 is None:
            print('x803: {}'.format(x803))
        elif isinstance(x803, torch.Tensor):
            print('x803: {}'.format(x803.shape))
        elif isinstance(x803, tuple):
            tuple_shapes = '('
            for item in x803:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x803: {}'.format(tuple_shapes))
        else:
            print('x803: {}'.format(x803))
        x804=self.silu152(x803)
        if x804 is None:
            print('x804: {}'.format(x804))
        elif isinstance(x804, torch.Tensor):
            print('x804: {}'.format(x804.shape))
        elif isinstance(x804, tuple):
            tuple_shapes = '('
            for item in x804:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x804: {}'.format(tuple_shapes))
        else:
            print('x804: {}'.format(x804))
        x805=self.conv2d255(x804)
        if x805 is None:
            print('x805: {}'.format(x805))
        elif isinstance(x805, torch.Tensor):
            print('x805: {}'.format(x805.shape))
        elif isinstance(x805, tuple):
            tuple_shapes = '('
            for item in x805:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x805: {}'.format(tuple_shapes))
        else:
            print('x805: {}'.format(x805))
        x806=self.sigmoid51(x805)
        if x806 is None:
            print('x806: {}'.format(x806))
        elif isinstance(x806, torch.Tensor):
            print('x806: {}'.format(x806.shape))
        elif isinstance(x806, tuple):
            tuple_shapes = '('
            for item in x806:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x806: {}'.format(tuple_shapes))
        else:
            print('x806: {}'.format(x806))
        x807=operator.mul(x806, x801)
        if x807 is None:
            print('x807: {}'.format(x807))
        elif isinstance(x807, torch.Tensor):
            print('x807: {}'.format(x807.shape))
        elif isinstance(x807, tuple):
            tuple_shapes = '('
            for item in x807:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x807: {}'.format(tuple_shapes))
        else:
            print('x807: {}'.format(x807))
        x808=self.conv2d256(x807)
        if x808 is None:
            print('x808: {}'.format(x808))
        elif isinstance(x808, torch.Tensor):
            print('x808: {}'.format(x808.shape))
        elif isinstance(x808, tuple):
            tuple_shapes = '('
            for item in x808:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x808: {}'.format(tuple_shapes))
        else:
            print('x808: {}'.format(x808))
        x809=self.batchnorm2d152(x808)
        if x809 is None:
            print('x809: {}'.format(x809))
        elif isinstance(x809, torch.Tensor):
            print('x809: {}'.format(x809.shape))
        elif isinstance(x809, tuple):
            tuple_shapes = '('
            for item in x809:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x809: {}'.format(tuple_shapes))
        else:
            print('x809: {}'.format(x809))
        x810=self.conv2d257(x809)
        if x810 is None:
            print('x810: {}'.format(x810))
        elif isinstance(x810, torch.Tensor):
            print('x810: {}'.format(x810.shape))
        elif isinstance(x810, tuple):
            tuple_shapes = '('
            for item in x810:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x810: {}'.format(tuple_shapes))
        else:
            print('x810: {}'.format(x810))
        x811=self.batchnorm2d153(x810)
        if x811 is None:
            print('x811: {}'.format(x811))
        elif isinstance(x811, torch.Tensor):
            print('x811: {}'.format(x811.shape))
        elif isinstance(x811, tuple):
            tuple_shapes = '('
            for item in x811:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x811: {}'.format(tuple_shapes))
        else:
            print('x811: {}'.format(x811))
        x812=self.silu153(x811)
        if x812 is None:
            print('x812: {}'.format(x812))
        elif isinstance(x812, torch.Tensor):
            print('x812: {}'.format(x812.shape))
        elif isinstance(x812, tuple):
            tuple_shapes = '('
            for item in x812:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x812: {}'.format(tuple_shapes))
        else:
            print('x812: {}'.format(x812))
        x813=self.conv2d258(x812)
        if x813 is None:
            print('x813: {}'.format(x813))
        elif isinstance(x813, torch.Tensor):
            print('x813: {}'.format(x813.shape))
        elif isinstance(x813, tuple):
            tuple_shapes = '('
            for item in x813:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x813: {}'.format(tuple_shapes))
        else:
            print('x813: {}'.format(x813))
        x814=self.batchnorm2d154(x813)
        if x814 is None:
            print('x814: {}'.format(x814))
        elif isinstance(x814, torch.Tensor):
            print('x814: {}'.format(x814.shape))
        elif isinstance(x814, tuple):
            tuple_shapes = '('
            for item in x814:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x814: {}'.format(tuple_shapes))
        else:
            print('x814: {}'.format(x814))
        x815=self.silu154(x814)
        if x815 is None:
            print('x815: {}'.format(x815))
        elif isinstance(x815, torch.Tensor):
            print('x815: {}'.format(x815.shape))
        elif isinstance(x815, tuple):
            tuple_shapes = '('
            for item in x815:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x815: {}'.format(tuple_shapes))
        else:
            print('x815: {}'.format(x815))
        x816=self.adaptiveavgpool2d52(x815)
        if x816 is None:
            print('x816: {}'.format(x816))
        elif isinstance(x816, torch.Tensor):
            print('x816: {}'.format(x816.shape))
        elif isinstance(x816, tuple):
            tuple_shapes = '('
            for item in x816:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x816: {}'.format(tuple_shapes))
        else:
            print('x816: {}'.format(x816))
        x817=self.conv2d259(x816)
        if x817 is None:
            print('x817: {}'.format(x817))
        elif isinstance(x817, torch.Tensor):
            print('x817: {}'.format(x817.shape))
        elif isinstance(x817, tuple):
            tuple_shapes = '('
            for item in x817:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x817: {}'.format(tuple_shapes))
        else:
            print('x817: {}'.format(x817))
        x818=self.silu155(x817)
        if x818 is None:
            print('x818: {}'.format(x818))
        elif isinstance(x818, torch.Tensor):
            print('x818: {}'.format(x818.shape))
        elif isinstance(x818, tuple):
            tuple_shapes = '('
            for item in x818:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x818: {}'.format(tuple_shapes))
        else:
            print('x818: {}'.format(x818))
        x819=self.conv2d260(x818)
        if x819 is None:
            print('x819: {}'.format(x819))
        elif isinstance(x819, torch.Tensor):
            print('x819: {}'.format(x819.shape))
        elif isinstance(x819, tuple):
            tuple_shapes = '('
            for item in x819:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x819: {}'.format(tuple_shapes))
        else:
            print('x819: {}'.format(x819))
        x820=self.sigmoid52(x819)
        if x820 is None:
            print('x820: {}'.format(x820))
        elif isinstance(x820, torch.Tensor):
            print('x820: {}'.format(x820.shape))
        elif isinstance(x820, tuple):
            tuple_shapes = '('
            for item in x820:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x820: {}'.format(tuple_shapes))
        else:
            print('x820: {}'.format(x820))
        x821=operator.mul(x820, x815)
        if x821 is None:
            print('x821: {}'.format(x821))
        elif isinstance(x821, torch.Tensor):
            print('x821: {}'.format(x821.shape))
        elif isinstance(x821, tuple):
            tuple_shapes = '('
            for item in x821:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x821: {}'.format(tuple_shapes))
        else:
            print('x821: {}'.format(x821))
        x822=self.conv2d261(x821)
        if x822 is None:
            print('x822: {}'.format(x822))
        elif isinstance(x822, torch.Tensor):
            print('x822: {}'.format(x822.shape))
        elif isinstance(x822, tuple):
            tuple_shapes = '('
            for item in x822:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x822: {}'.format(tuple_shapes))
        else:
            print('x822: {}'.format(x822))
        x823=self.batchnorm2d155(x822)
        if x823 is None:
            print('x823: {}'.format(x823))
        elif isinstance(x823, torch.Tensor):
            print('x823: {}'.format(x823.shape))
        elif isinstance(x823, tuple):
            tuple_shapes = '('
            for item in x823:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x823: {}'.format(tuple_shapes))
        else:
            print('x823: {}'.format(x823))
        x824=stochastic_depth(x823, 0.1890909090909091, 'row', False)
        if x824 is None:
            print('x824: {}'.format(x824))
        elif isinstance(x824, torch.Tensor):
            print('x824: {}'.format(x824.shape))
        elif isinstance(x824, tuple):
            tuple_shapes = '('
            for item in x824:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x824: {}'.format(tuple_shapes))
        else:
            print('x824: {}'.format(x824))
        x825=operator.add(x824, x809)
        if x825 is None:
            print('x825: {}'.format(x825))
        elif isinstance(x825, torch.Tensor):
            print('x825: {}'.format(x825.shape))
        elif isinstance(x825, tuple):
            tuple_shapes = '('
            for item in x825:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x825: {}'.format(tuple_shapes))
        else:
            print('x825: {}'.format(x825))
        x826=self.conv2d262(x825)
        if x826 is None:
            print('x826: {}'.format(x826))
        elif isinstance(x826, torch.Tensor):
            print('x826: {}'.format(x826.shape))
        elif isinstance(x826, tuple):
            tuple_shapes = '('
            for item in x826:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x826: {}'.format(tuple_shapes))
        else:
            print('x826: {}'.format(x826))
        x827=self.batchnorm2d156(x826)
        if x827 is None:
            print('x827: {}'.format(x827))
        elif isinstance(x827, torch.Tensor):
            print('x827: {}'.format(x827.shape))
        elif isinstance(x827, tuple):
            tuple_shapes = '('
            for item in x827:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x827: {}'.format(tuple_shapes))
        else:
            print('x827: {}'.format(x827))
        x828=self.silu156(x827)
        if x828 is None:
            print('x828: {}'.format(x828))
        elif isinstance(x828, torch.Tensor):
            print('x828: {}'.format(x828.shape))
        elif isinstance(x828, tuple):
            tuple_shapes = '('
            for item in x828:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x828: {}'.format(tuple_shapes))
        else:
            print('x828: {}'.format(x828))
        x829=self.conv2d263(x828)
        if x829 is None:
            print('x829: {}'.format(x829))
        elif isinstance(x829, torch.Tensor):
            print('x829: {}'.format(x829.shape))
        elif isinstance(x829, tuple):
            tuple_shapes = '('
            for item in x829:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x829: {}'.format(tuple_shapes))
        else:
            print('x829: {}'.format(x829))
        x830=self.batchnorm2d157(x829)
        if x830 is None:
            print('x830: {}'.format(x830))
        elif isinstance(x830, torch.Tensor):
            print('x830: {}'.format(x830.shape))
        elif isinstance(x830, tuple):
            tuple_shapes = '('
            for item in x830:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x830: {}'.format(tuple_shapes))
        else:
            print('x830: {}'.format(x830))
        x831=self.silu157(x830)
        if x831 is None:
            print('x831: {}'.format(x831))
        elif isinstance(x831, torch.Tensor):
            print('x831: {}'.format(x831.shape))
        elif isinstance(x831, tuple):
            tuple_shapes = '('
            for item in x831:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x831: {}'.format(tuple_shapes))
        else:
            print('x831: {}'.format(x831))
        x832=self.adaptiveavgpool2d53(x831)
        if x832 is None:
            print('x832: {}'.format(x832))
        elif isinstance(x832, torch.Tensor):
            print('x832: {}'.format(x832.shape))
        elif isinstance(x832, tuple):
            tuple_shapes = '('
            for item in x832:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x832: {}'.format(tuple_shapes))
        else:
            print('x832: {}'.format(x832))
        x833=self.conv2d264(x832)
        if x833 is None:
            print('x833: {}'.format(x833))
        elif isinstance(x833, torch.Tensor):
            print('x833: {}'.format(x833.shape))
        elif isinstance(x833, tuple):
            tuple_shapes = '('
            for item in x833:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x833: {}'.format(tuple_shapes))
        else:
            print('x833: {}'.format(x833))
        x834=self.silu158(x833)
        if x834 is None:
            print('x834: {}'.format(x834))
        elif isinstance(x834, torch.Tensor):
            print('x834: {}'.format(x834.shape))
        elif isinstance(x834, tuple):
            tuple_shapes = '('
            for item in x834:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x834: {}'.format(tuple_shapes))
        else:
            print('x834: {}'.format(x834))
        x835=self.conv2d265(x834)
        if x835 is None:
            print('x835: {}'.format(x835))
        elif isinstance(x835, torch.Tensor):
            print('x835: {}'.format(x835.shape))
        elif isinstance(x835, tuple):
            tuple_shapes = '('
            for item in x835:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x835: {}'.format(tuple_shapes))
        else:
            print('x835: {}'.format(x835))
        x836=self.sigmoid53(x835)
        if x836 is None:
            print('x836: {}'.format(x836))
        elif isinstance(x836, torch.Tensor):
            print('x836: {}'.format(x836.shape))
        elif isinstance(x836, tuple):
            tuple_shapes = '('
            for item in x836:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x836: {}'.format(tuple_shapes))
        else:
            print('x836: {}'.format(x836))
        x837=operator.mul(x836, x831)
        if x837 is None:
            print('x837: {}'.format(x837))
        elif isinstance(x837, torch.Tensor):
            print('x837: {}'.format(x837.shape))
        elif isinstance(x837, tuple):
            tuple_shapes = '('
            for item in x837:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x837: {}'.format(tuple_shapes))
        else:
            print('x837: {}'.format(x837))
        x838=self.conv2d266(x837)
        if x838 is None:
            print('x838: {}'.format(x838))
        elif isinstance(x838, torch.Tensor):
            print('x838: {}'.format(x838.shape))
        elif isinstance(x838, tuple):
            tuple_shapes = '('
            for item in x838:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x838: {}'.format(tuple_shapes))
        else:
            print('x838: {}'.format(x838))
        x839=self.batchnorm2d158(x838)
        if x839 is None:
            print('x839: {}'.format(x839))
        elif isinstance(x839, torch.Tensor):
            print('x839: {}'.format(x839.shape))
        elif isinstance(x839, tuple):
            tuple_shapes = '('
            for item in x839:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x839: {}'.format(tuple_shapes))
        else:
            print('x839: {}'.format(x839))
        x840=stochastic_depth(x839, 0.19272727272727275, 'row', False)
        if x840 is None:
            print('x840: {}'.format(x840))
        elif isinstance(x840, torch.Tensor):
            print('x840: {}'.format(x840.shape))
        elif isinstance(x840, tuple):
            tuple_shapes = '('
            for item in x840:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x840: {}'.format(tuple_shapes))
        else:
            print('x840: {}'.format(x840))
        x841=operator.add(x840, x825)
        if x841 is None:
            print('x841: {}'.format(x841))
        elif isinstance(x841, torch.Tensor):
            print('x841: {}'.format(x841.shape))
        elif isinstance(x841, tuple):
            tuple_shapes = '('
            for item in x841:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x841: {}'.format(tuple_shapes))
        else:
            print('x841: {}'.format(x841))
        x842=self.conv2d267(x841)
        if x842 is None:
            print('x842: {}'.format(x842))
        elif isinstance(x842, torch.Tensor):
            print('x842: {}'.format(x842.shape))
        elif isinstance(x842, tuple):
            tuple_shapes = '('
            for item in x842:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x842: {}'.format(tuple_shapes))
        else:
            print('x842: {}'.format(x842))
        x843=self.batchnorm2d159(x842)
        if x843 is None:
            print('x843: {}'.format(x843))
        elif isinstance(x843, torch.Tensor):
            print('x843: {}'.format(x843.shape))
        elif isinstance(x843, tuple):
            tuple_shapes = '('
            for item in x843:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x843: {}'.format(tuple_shapes))
        else:
            print('x843: {}'.format(x843))
        x844=self.silu159(x843)
        if x844 is None:
            print('x844: {}'.format(x844))
        elif isinstance(x844, torch.Tensor):
            print('x844: {}'.format(x844.shape))
        elif isinstance(x844, tuple):
            tuple_shapes = '('
            for item in x844:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x844: {}'.format(tuple_shapes))
        else:
            print('x844: {}'.format(x844))
        x845=self.conv2d268(x844)
        if x845 is None:
            print('x845: {}'.format(x845))
        elif isinstance(x845, torch.Tensor):
            print('x845: {}'.format(x845.shape))
        elif isinstance(x845, tuple):
            tuple_shapes = '('
            for item in x845:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x845: {}'.format(tuple_shapes))
        else:
            print('x845: {}'.format(x845))
        x846=self.batchnorm2d160(x845)
        if x846 is None:
            print('x846: {}'.format(x846))
        elif isinstance(x846, torch.Tensor):
            print('x846: {}'.format(x846.shape))
        elif isinstance(x846, tuple):
            tuple_shapes = '('
            for item in x846:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x846: {}'.format(tuple_shapes))
        else:
            print('x846: {}'.format(x846))
        x847=self.silu160(x846)
        if x847 is None:
            print('x847: {}'.format(x847))
        elif isinstance(x847, torch.Tensor):
            print('x847: {}'.format(x847.shape))
        elif isinstance(x847, tuple):
            tuple_shapes = '('
            for item in x847:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x847: {}'.format(tuple_shapes))
        else:
            print('x847: {}'.format(x847))
        x848=self.adaptiveavgpool2d54(x847)
        if x848 is None:
            print('x848: {}'.format(x848))
        elif isinstance(x848, torch.Tensor):
            print('x848: {}'.format(x848.shape))
        elif isinstance(x848, tuple):
            tuple_shapes = '('
            for item in x848:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x848: {}'.format(tuple_shapes))
        else:
            print('x848: {}'.format(x848))
        x849=self.conv2d269(x848)
        if x849 is None:
            print('x849: {}'.format(x849))
        elif isinstance(x849, torch.Tensor):
            print('x849: {}'.format(x849.shape))
        elif isinstance(x849, tuple):
            tuple_shapes = '('
            for item in x849:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x849: {}'.format(tuple_shapes))
        else:
            print('x849: {}'.format(x849))
        x850=self.silu161(x849)
        if x850 is None:
            print('x850: {}'.format(x850))
        elif isinstance(x850, torch.Tensor):
            print('x850: {}'.format(x850.shape))
        elif isinstance(x850, tuple):
            tuple_shapes = '('
            for item in x850:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x850: {}'.format(tuple_shapes))
        else:
            print('x850: {}'.format(x850))
        x851=self.conv2d270(x850)
        if x851 is None:
            print('x851: {}'.format(x851))
        elif isinstance(x851, torch.Tensor):
            print('x851: {}'.format(x851.shape))
        elif isinstance(x851, tuple):
            tuple_shapes = '('
            for item in x851:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x851: {}'.format(tuple_shapes))
        else:
            print('x851: {}'.format(x851))
        x852=self.sigmoid54(x851)
        if x852 is None:
            print('x852: {}'.format(x852))
        elif isinstance(x852, torch.Tensor):
            print('x852: {}'.format(x852.shape))
        elif isinstance(x852, tuple):
            tuple_shapes = '('
            for item in x852:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x852: {}'.format(tuple_shapes))
        else:
            print('x852: {}'.format(x852))
        x853=operator.mul(x852, x847)
        if x853 is None:
            print('x853: {}'.format(x853))
        elif isinstance(x853, torch.Tensor):
            print('x853: {}'.format(x853.shape))
        elif isinstance(x853, tuple):
            tuple_shapes = '('
            for item in x853:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x853: {}'.format(tuple_shapes))
        else:
            print('x853: {}'.format(x853))
        x854=self.conv2d271(x853)
        if x854 is None:
            print('x854: {}'.format(x854))
        elif isinstance(x854, torch.Tensor):
            print('x854: {}'.format(x854.shape))
        elif isinstance(x854, tuple):
            tuple_shapes = '('
            for item in x854:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x854: {}'.format(tuple_shapes))
        else:
            print('x854: {}'.format(x854))
        x855=self.batchnorm2d161(x854)
        if x855 is None:
            print('x855: {}'.format(x855))
        elif isinstance(x855, torch.Tensor):
            print('x855: {}'.format(x855.shape))
        elif isinstance(x855, tuple):
            tuple_shapes = '('
            for item in x855:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x855: {}'.format(tuple_shapes))
        else:
            print('x855: {}'.format(x855))
        x856=stochastic_depth(x855, 0.19636363636363638, 'row', False)
        if x856 is None:
            print('x856: {}'.format(x856))
        elif isinstance(x856, torch.Tensor):
            print('x856: {}'.format(x856.shape))
        elif isinstance(x856, tuple):
            tuple_shapes = '('
            for item in x856:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x856: {}'.format(tuple_shapes))
        else:
            print('x856: {}'.format(x856))
        x857=operator.add(x856, x841)
        if x857 is None:
            print('x857: {}'.format(x857))
        elif isinstance(x857, torch.Tensor):
            print('x857: {}'.format(x857.shape))
        elif isinstance(x857, tuple):
            tuple_shapes = '('
            for item in x857:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x857: {}'.format(tuple_shapes))
        else:
            print('x857: {}'.format(x857))
        x858=self.conv2d272(x857)
        if x858 is None:
            print('x858: {}'.format(x858))
        elif isinstance(x858, torch.Tensor):
            print('x858: {}'.format(x858.shape))
        elif isinstance(x858, tuple):
            tuple_shapes = '('
            for item in x858:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x858: {}'.format(tuple_shapes))
        else:
            print('x858: {}'.format(x858))
        x859=self.batchnorm2d162(x858)
        if x859 is None:
            print('x859: {}'.format(x859))
        elif isinstance(x859, torch.Tensor):
            print('x859: {}'.format(x859.shape))
        elif isinstance(x859, tuple):
            tuple_shapes = '('
            for item in x859:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x859: {}'.format(tuple_shapes))
        else:
            print('x859: {}'.format(x859))
        x860=self.silu162(x859)
        if x860 is None:
            print('x860: {}'.format(x860))
        elif isinstance(x860, torch.Tensor):
            print('x860: {}'.format(x860.shape))
        elif isinstance(x860, tuple):
            tuple_shapes = '('
            for item in x860:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x860: {}'.format(tuple_shapes))
        else:
            print('x860: {}'.format(x860))
        x861=self.adaptiveavgpool2d55(x860)
        if x861 is None:
            print('x861: {}'.format(x861))
        elif isinstance(x861, torch.Tensor):
            print('x861: {}'.format(x861.shape))
        elif isinstance(x861, tuple):
            tuple_shapes = '('
            for item in x861:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x861: {}'.format(tuple_shapes))
        else:
            print('x861: {}'.format(x861))
        x862=torch.flatten(x861, 1)
        if x862 is None:
            print('x862: {}'.format(x862))
        elif isinstance(x862, torch.Tensor):
            print('x862: {}'.format(x862.shape))
        elif isinstance(x862, tuple):
            tuple_shapes = '('
            for item in x862:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x862: {}'.format(tuple_shapes))
        else:
            print('x862: {}'.format(x862))
        x863=self.dropout0(x862)
        if x863 is None:
            print('x863: {}'.format(x863))
        elif isinstance(x863, torch.Tensor):
            print('x863: {}'.format(x863.shape))
        elif isinstance(x863, tuple):
            tuple_shapes = '('
            for item in x863:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x863: {}'.format(tuple_shapes))
        else:
            print('x863: {}'.format(x863))
        x864=self.linear0(x863)
        if x864 is None:
            print('x864: {}'.format(x864))
        elif isinstance(x864, torch.Tensor):
            print('x864: {}'.format(x864.shape))
        elif isinstance(x864, tuple):
            tuple_shapes = '('
            for item in x864:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x864: {}'.format(tuple_shapes))
        else:
            print('x864: {}'.format(x864))

m = M().eval()
x = torch.rand(1, 3, 224, 224)
output = m(x)
