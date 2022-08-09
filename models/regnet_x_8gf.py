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
        self.conv2d1 = Conv2d(32, 80, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d1 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(32, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(80, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(80, 240, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(80, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d10 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d13 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d16 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d19 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.batchnorm2d22 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(240, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(240, 720, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d24 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(240, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(720, 720, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d26 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d29 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d32 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d35 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d38 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d41 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d44 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d47 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d50 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d53 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu51 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d56 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d57 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d59 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d62 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu59 = ReLU(inplace=True)
        self.conv2d63 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d65 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d68 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(720, 1920, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d70 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d71 = Conv2d(720, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d71 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d72 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1920, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear0 = Linear(in_features=1920, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.conv2d2(x3)
        x7=self.batchnorm2d2(x6)
        x8=self.relu1(x7)
        x9=self.conv2d3(x8)
        x10=self.batchnorm2d3(x9)
        x11=self.relu2(x10)
        x12=self.conv2d4(x11)
        x13=self.batchnorm2d4(x12)
        x14=operator.add(x5, x13)
        x15=self.relu3(x14)
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        x18=self.relu4(x17)
        x19=self.conv2d6(x18)
        x20=self.batchnorm2d6(x19)
        x21=self.relu5(x20)
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d7(x22)
        x24=operator.add(x15, x23)
        x25=self.relu6(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        x28=self.conv2d9(x25)
        x29=self.batchnorm2d9(x28)
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        x32=self.batchnorm2d10(x31)
        x33=self.relu8(x32)
        x34=self.conv2d11(x33)
        x35=self.batchnorm2d11(x34)
        x36=operator.add(x27, x35)
        x37=self.relu9(x36)
        x38=self.conv2d12(x37)
        x39=self.batchnorm2d12(x38)
        x40=self.relu10(x39)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d13(x41)
        x43=self.relu11(x42)
        x44=self.conv2d14(x43)
        x45=self.batchnorm2d14(x44)
        x46=operator.add(x37, x45)
        x47=self.relu12(x46)
        x48=self.conv2d15(x47)
        x49=self.batchnorm2d15(x48)
        x50=self.relu13(x49)
        x51=self.conv2d16(x50)
        x52=self.batchnorm2d16(x51)
        x53=self.relu14(x52)
        x54=self.conv2d17(x53)
        x55=self.batchnorm2d17(x54)
        x56=operator.add(x47, x55)
        x57=self.relu15(x56)
        x58=self.conv2d18(x57)
        x59=self.batchnorm2d18(x58)
        x60=self.relu16(x59)
        x61=self.conv2d19(x60)
        x62=self.batchnorm2d19(x61)
        x63=self.relu17(x62)
        x64=self.conv2d20(x63)
        x65=self.batchnorm2d20(x64)
        x66=operator.add(x57, x65)
        x67=self.relu18(x66)
        x68=self.conv2d21(x67)
        x69=self.batchnorm2d21(x68)
        x70=self.relu19(x69)
        x71=self.conv2d22(x70)
        x72=self.batchnorm2d22(x71)
        x73=self.relu20(x72)
        x74=self.conv2d23(x73)
        x75=self.batchnorm2d23(x74)
        x76=operator.add(x67, x75)
        x77=self.relu21(x76)
        x78=self.conv2d24(x77)
        x79=self.batchnorm2d24(x78)
        x80=self.conv2d25(x77)
        x81=self.batchnorm2d25(x80)
        x82=self.relu22(x81)
        x83=self.conv2d26(x82)
        x84=self.batchnorm2d26(x83)
        x85=self.relu23(x84)
        x86=self.conv2d27(x85)
        x87=self.batchnorm2d27(x86)
        x88=operator.add(x79, x87)
        x89=self.relu24(x88)
        x90=self.conv2d28(x89)
        x91=self.batchnorm2d28(x90)
        x92=self.relu25(x91)
        x93=self.conv2d29(x92)
        x94=self.batchnorm2d29(x93)
        x95=self.relu26(x94)
        x96=self.conv2d30(x95)
        x97=self.batchnorm2d30(x96)
        x98=operator.add(x89, x97)
        x99=self.relu27(x98)
        x100=self.conv2d31(x99)
        x101=self.batchnorm2d31(x100)
        x102=self.relu28(x101)
        x103=self.conv2d32(x102)
        x104=self.batchnorm2d32(x103)
        x105=self.relu29(x104)
        x106=self.conv2d33(x105)
        x107=self.batchnorm2d33(x106)
        x108=operator.add(x99, x107)
        x109=self.relu30(x108)
        x110=self.conv2d34(x109)
        x111=self.batchnorm2d34(x110)
        x112=self.relu31(x111)
        x113=self.conv2d35(x112)
        x114=self.batchnorm2d35(x113)
        x115=self.relu32(x114)
        x116=self.conv2d36(x115)
        x117=self.batchnorm2d36(x116)
        x118=operator.add(x109, x117)
        x119=self.relu33(x118)
        x120=self.conv2d37(x119)
        x121=self.batchnorm2d37(x120)
        x122=self.relu34(x121)
        x123=self.conv2d38(x122)
        x124=self.batchnorm2d38(x123)
        x125=self.relu35(x124)
        x126=self.conv2d39(x125)
        x127=self.batchnorm2d39(x126)
        x128=operator.add(x119, x127)
        x129=self.relu36(x128)
        x130=self.conv2d40(x129)
        x131=self.batchnorm2d40(x130)
        x132=self.relu37(x131)
        x133=self.conv2d41(x132)
        x134=self.batchnorm2d41(x133)
        x135=self.relu38(x134)
        x136=self.conv2d42(x135)
        x137=self.batchnorm2d42(x136)
        x138=operator.add(x129, x137)
        x139=self.relu39(x138)
        x140=self.conv2d43(x139)
        x141=self.batchnorm2d43(x140)
        x142=self.relu40(x141)
        x143=self.conv2d44(x142)
        x144=self.batchnorm2d44(x143)
        x145=self.relu41(x144)
        x146=self.conv2d45(x145)
        x147=self.batchnorm2d45(x146)
        x148=operator.add(x139, x147)
        x149=self.relu42(x148)
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        x152=self.relu43(x151)
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu44(x154)
        x156=self.conv2d48(x155)
        x157=self.batchnorm2d48(x156)
        x158=operator.add(x149, x157)
        x159=self.relu45(x158)
        x160=self.conv2d49(x159)
        x161=self.batchnorm2d49(x160)
        x162=self.relu46(x161)
        x163=self.conv2d50(x162)
        x164=self.batchnorm2d50(x163)
        x165=self.relu47(x164)
        x166=self.conv2d51(x165)
        x167=self.batchnorm2d51(x166)
        x168=operator.add(x159, x167)
        x169=self.relu48(x168)
        x170=self.conv2d52(x169)
        x171=self.batchnorm2d52(x170)
        x172=self.relu49(x171)
        x173=self.conv2d53(x172)
        x174=self.batchnorm2d53(x173)
        x175=self.relu50(x174)
        x176=self.conv2d54(x175)
        x177=self.batchnorm2d54(x176)
        x178=operator.add(x169, x177)
        x179=self.relu51(x178)
        x180=self.conv2d55(x179)
        x181=self.batchnorm2d55(x180)
        x182=self.relu52(x181)
        x183=self.conv2d56(x182)
        x184=self.batchnorm2d56(x183)
        x185=self.relu53(x184)
        x186=self.conv2d57(x185)
        x187=self.batchnorm2d57(x186)
        x188=operator.add(x179, x187)
        x189=self.relu54(x188)
        x190=self.conv2d58(x189)
        x191=self.batchnorm2d58(x190)
        x192=self.relu55(x191)
        x193=self.conv2d59(x192)
        x194=self.batchnorm2d59(x193)
        x195=self.relu56(x194)
        x196=self.conv2d60(x195)
        x197=self.batchnorm2d60(x196)
        x198=operator.add(x189, x197)
        x199=self.relu57(x198)
        x200=self.conv2d61(x199)
        x201=self.batchnorm2d61(x200)
        x202=self.relu58(x201)
        x203=self.conv2d62(x202)
        x204=self.batchnorm2d62(x203)
        x205=self.relu59(x204)
        x206=self.conv2d63(x205)
        x207=self.batchnorm2d63(x206)
        x208=operator.add(x199, x207)
        x209=self.relu60(x208)
        x210=self.conv2d64(x209)
        x211=self.batchnorm2d64(x210)
        x212=self.relu61(x211)
        x213=self.conv2d65(x212)
        x214=self.batchnorm2d65(x213)
        x215=self.relu62(x214)
        x216=self.conv2d66(x215)
        x217=self.batchnorm2d66(x216)
        x218=operator.add(x209, x217)
        x219=self.relu63(x218)
        x220=self.conv2d67(x219)
        x221=self.batchnorm2d67(x220)
        x222=self.relu64(x221)
        x223=self.conv2d68(x222)
        x224=self.batchnorm2d68(x223)
        x225=self.relu65(x224)
        x226=self.conv2d69(x225)
        x227=self.batchnorm2d69(x226)
        x228=operator.add(x219, x227)
        x229=self.relu66(x228)
        x230=self.conv2d70(x229)
        x231=self.batchnorm2d70(x230)
        x232=self.conv2d71(x229)
        x233=self.batchnorm2d71(x232)
        x234=self.relu67(x233)
        x235=self.conv2d72(x234)
        x236=self.batchnorm2d72(x235)
        x237=self.relu68(x236)
        x238=self.conv2d73(x237)
        x239=self.batchnorm2d73(x238)
        x240=operator.add(x231, x239)
        x241=self.relu69(x240)
        x242=self.adaptiveavgpool2d0(x241)
        x243=x242.flatten(start_dim=1)
        x244=self.linear0(x243)

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
