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
        self.conv2d0 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm2d0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)
        self.conv2d6 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d29 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d32 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d35 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d38 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d41 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d44 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu42 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d48 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d50 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d53 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu51 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d56 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d57 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu54 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d59 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d62 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu59 = ReLU(inplace=True)
        self.conv2d63 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d65 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d68 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d71 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d74 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d74 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu71 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu72 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)
        self.conv2d77 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d77 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)
        self.conv2d78 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu75 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d80 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu77 = ReLU(inplace=True)
        self.conv2d81 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu78 = ReLU(inplace=True)
        self.conv2d82 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)
        self.conv2d83 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d83 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu80 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu81 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d86 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d86 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu83 = ReLU(inplace=True)
        self.conv2d87 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu84 = ReLU(inplace=True)
        self.conv2d88 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d88 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d89 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu86 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu87 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d92 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu89 = ReLU(inplace=True)
        self.conv2d93 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu90 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu91 = ReLU(inplace=True)
        self.conv2d95 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d95 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu92 = ReLU(inplace=True)
        self.conv2d96 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d97 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d97 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu93 = ReLU(inplace=True)
        self.conv2d98 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d99 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu95 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d100 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu96 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d102 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu98 = ReLU(inplace=True)
        self.conv2d103 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu99 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu100 = ReLU()
        self.dropout0 = Dropout(p=0.1, inplace=False)
        self.conv2d105 = Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d106 = Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.conv2d107 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x0=x
        x1=builtins.getattr(x0, 'shape')
        x2=operator.getitem(x1, slice(-2, None, None))
        x3=self.conv2d0(x0)
        x4=self.batchnorm2d0(x3)
        x5=self.relu0(x4)
        x6=self.maxpool2d0(x5)
        x7=self.conv2d1(x6)
        x8=self.batchnorm2d1(x7)
        x9=self.relu1(x8)
        x10=self.conv2d2(x9)
        x11=self.batchnorm2d2(x10)
        x12=self.relu1(x11)
        x13=self.conv2d3(x12)
        x14=self.batchnorm2d3(x13)
        x15=self.conv2d4(x6)
        x16=self.batchnorm2d4(x15)
        x17=operator.add(x14, x16)
        x18=self.relu1(x17)
        x19=self.conv2d5(x18)
        x20=self.batchnorm2d5(x19)
        x21=self.relu4(x20)
        x22=self.conv2d6(x21)
        x23=self.batchnorm2d6(x22)
        x24=self.relu4(x23)
        x25=self.conv2d7(x24)
        x26=self.batchnorm2d7(x25)
        x27=operator.add(x26, x18)
        x28=self.relu4(x27)
        x29=self.conv2d8(x28)
        x30=self.batchnorm2d8(x29)
        x31=self.relu7(x30)
        x32=self.conv2d9(x31)
        x33=self.batchnorm2d9(x32)
        x34=self.relu7(x33)
        x35=self.conv2d10(x34)
        x36=self.batchnorm2d10(x35)
        x37=operator.add(x36, x28)
        x38=self.relu7(x37)
        x39=self.conv2d11(x38)
        x40=self.batchnorm2d11(x39)
        x41=self.relu10(x40)
        x42=self.conv2d12(x41)
        x43=self.batchnorm2d12(x42)
        x44=self.relu10(x43)
        x45=self.conv2d13(x44)
        x46=self.batchnorm2d13(x45)
        x47=self.conv2d14(x38)
        x48=self.batchnorm2d14(x47)
        x49=operator.add(x46, x48)
        x50=self.relu10(x49)
        x51=self.conv2d15(x50)
        x52=self.batchnorm2d15(x51)
        x53=self.relu13(x52)
        x54=self.conv2d16(x53)
        x55=self.batchnorm2d16(x54)
        x56=self.relu13(x55)
        x57=self.conv2d17(x56)
        x58=self.batchnorm2d17(x57)
        x59=operator.add(x58, x50)
        x60=self.relu13(x59)
        x61=self.conv2d18(x60)
        x62=self.batchnorm2d18(x61)
        x63=self.relu16(x62)
        x64=self.conv2d19(x63)
        x65=self.batchnorm2d19(x64)
        x66=self.relu16(x65)
        x67=self.conv2d20(x66)
        x68=self.batchnorm2d20(x67)
        x69=operator.add(x68, x60)
        x70=self.relu16(x69)
        x71=self.conv2d21(x70)
        x72=self.batchnorm2d21(x71)
        x73=self.relu19(x72)
        x74=self.conv2d22(x73)
        x75=self.batchnorm2d22(x74)
        x76=self.relu19(x75)
        x77=self.conv2d23(x76)
        x78=self.batchnorm2d23(x77)
        x79=operator.add(x78, x70)
        x80=self.relu19(x79)
        x81=self.conv2d24(x80)
        x82=self.batchnorm2d24(x81)
        x83=self.relu22(x82)
        x84=self.conv2d25(x83)
        x85=self.batchnorm2d25(x84)
        x86=self.relu22(x85)
        x87=self.conv2d26(x86)
        x88=self.batchnorm2d26(x87)
        x89=self.conv2d27(x80)
        x90=self.batchnorm2d27(x89)
        x91=operator.add(x88, x90)
        x92=self.relu22(x91)
        x93=self.conv2d28(x92)
        x94=self.batchnorm2d28(x93)
        x95=self.relu25(x94)
        x96=self.conv2d29(x95)
        x97=self.batchnorm2d29(x96)
        x98=self.relu25(x97)
        x99=self.conv2d30(x98)
        x100=self.batchnorm2d30(x99)
        x101=operator.add(x100, x92)
        x102=self.relu25(x101)
        x103=self.conv2d31(x102)
        x104=self.batchnorm2d31(x103)
        x105=self.relu28(x104)
        x106=self.conv2d32(x105)
        x107=self.batchnorm2d32(x106)
        x108=self.relu28(x107)
        x109=self.conv2d33(x108)
        x110=self.batchnorm2d33(x109)
        x111=operator.add(x110, x102)
        x112=self.relu28(x111)
        x113=self.conv2d34(x112)
        x114=self.batchnorm2d34(x113)
        x115=self.relu31(x114)
        x116=self.conv2d35(x115)
        x117=self.batchnorm2d35(x116)
        x118=self.relu31(x117)
        x119=self.conv2d36(x118)
        x120=self.batchnorm2d36(x119)
        x121=operator.add(x120, x112)
        x122=self.relu31(x121)
        x123=self.conv2d37(x122)
        x124=self.batchnorm2d37(x123)
        x125=self.relu34(x124)
        x126=self.conv2d38(x125)
        x127=self.batchnorm2d38(x126)
        x128=self.relu34(x127)
        x129=self.conv2d39(x128)
        x130=self.batchnorm2d39(x129)
        x131=operator.add(x130, x122)
        x132=self.relu34(x131)
        x133=self.conv2d40(x132)
        x134=self.batchnorm2d40(x133)
        x135=self.relu37(x134)
        x136=self.conv2d41(x135)
        x137=self.batchnorm2d41(x136)
        x138=self.relu37(x137)
        x139=self.conv2d42(x138)
        x140=self.batchnorm2d42(x139)
        x141=operator.add(x140, x132)
        x142=self.relu37(x141)
        x143=self.conv2d43(x142)
        x144=self.batchnorm2d43(x143)
        x145=self.relu40(x144)
        x146=self.conv2d44(x145)
        x147=self.batchnorm2d44(x146)
        x148=self.relu40(x147)
        x149=self.conv2d45(x148)
        x150=self.batchnorm2d45(x149)
        x151=operator.add(x150, x142)
        x152=self.relu40(x151)
        x153=self.conv2d46(x152)
        x154=self.batchnorm2d46(x153)
        x155=self.relu43(x154)
        x156=self.conv2d47(x155)
        x157=self.batchnorm2d47(x156)
        x158=self.relu43(x157)
        x159=self.conv2d48(x158)
        x160=self.batchnorm2d48(x159)
        x161=operator.add(x160, x152)
        x162=self.relu43(x161)
        x163=self.conv2d49(x162)
        x164=self.batchnorm2d49(x163)
        x165=self.relu46(x164)
        x166=self.conv2d50(x165)
        x167=self.batchnorm2d50(x166)
        x168=self.relu46(x167)
        x169=self.conv2d51(x168)
        x170=self.batchnorm2d51(x169)
        x171=operator.add(x170, x162)
        x172=self.relu46(x171)
        x173=self.conv2d52(x172)
        x174=self.batchnorm2d52(x173)
        x175=self.relu49(x174)
        x176=self.conv2d53(x175)
        x177=self.batchnorm2d53(x176)
        x178=self.relu49(x177)
        x179=self.conv2d54(x178)
        x180=self.batchnorm2d54(x179)
        x181=operator.add(x180, x172)
        x182=self.relu49(x181)
        x183=self.conv2d55(x182)
        x184=self.batchnorm2d55(x183)
        x185=self.relu52(x184)
        x186=self.conv2d56(x185)
        x187=self.batchnorm2d56(x186)
        x188=self.relu52(x187)
        x189=self.conv2d57(x188)
        x190=self.batchnorm2d57(x189)
        x191=operator.add(x190, x182)
        x192=self.relu52(x191)
        x193=self.conv2d58(x192)
        x194=self.batchnorm2d58(x193)
        x195=self.relu55(x194)
        x196=self.conv2d59(x195)
        x197=self.batchnorm2d59(x196)
        x198=self.relu55(x197)
        x199=self.conv2d60(x198)
        x200=self.batchnorm2d60(x199)
        x201=operator.add(x200, x192)
        x202=self.relu55(x201)
        x203=self.conv2d61(x202)
        x204=self.batchnorm2d61(x203)
        x205=self.relu58(x204)
        x206=self.conv2d62(x205)
        x207=self.batchnorm2d62(x206)
        x208=self.relu58(x207)
        x209=self.conv2d63(x208)
        x210=self.batchnorm2d63(x209)
        x211=operator.add(x210, x202)
        x212=self.relu58(x211)
        x213=self.conv2d64(x212)
        x214=self.batchnorm2d64(x213)
        x215=self.relu61(x214)
        x216=self.conv2d65(x215)
        x217=self.batchnorm2d65(x216)
        x218=self.relu61(x217)
        x219=self.conv2d66(x218)
        x220=self.batchnorm2d66(x219)
        x221=operator.add(x220, x212)
        x222=self.relu61(x221)
        x223=self.conv2d67(x222)
        x224=self.batchnorm2d67(x223)
        x225=self.relu64(x224)
        x226=self.conv2d68(x225)
        x227=self.batchnorm2d68(x226)
        x228=self.relu64(x227)
        x229=self.conv2d69(x228)
        x230=self.batchnorm2d69(x229)
        x231=operator.add(x230, x222)
        x232=self.relu64(x231)
        x233=self.conv2d70(x232)
        x234=self.batchnorm2d70(x233)
        x235=self.relu67(x234)
        x236=self.conv2d71(x235)
        x237=self.batchnorm2d71(x236)
        x238=self.relu67(x237)
        x239=self.conv2d72(x238)
        x240=self.batchnorm2d72(x239)
        x241=operator.add(x240, x232)
        x242=self.relu67(x241)
        x243=self.conv2d73(x242)
        x244=self.batchnorm2d73(x243)
        x245=self.relu70(x244)
        x246=self.conv2d74(x245)
        x247=self.batchnorm2d74(x246)
        x248=self.relu70(x247)
        x249=self.conv2d75(x248)
        x250=self.batchnorm2d75(x249)
        x251=operator.add(x250, x242)
        x252=self.relu70(x251)
        x253=self.conv2d76(x252)
        x254=self.batchnorm2d76(x253)
        x255=self.relu73(x254)
        x256=self.conv2d77(x255)
        x257=self.batchnorm2d77(x256)
        x258=self.relu73(x257)
        x259=self.conv2d78(x258)
        x260=self.batchnorm2d78(x259)
        x261=operator.add(x260, x252)
        x262=self.relu73(x261)
        x263=self.conv2d79(x262)
        x264=self.batchnorm2d79(x263)
        x265=self.relu76(x264)
        x266=self.conv2d80(x265)
        x267=self.batchnorm2d80(x266)
        x268=self.relu76(x267)
        x269=self.conv2d81(x268)
        x270=self.batchnorm2d81(x269)
        x271=operator.add(x270, x262)
        x272=self.relu76(x271)
        x273=self.conv2d82(x272)
        x274=self.batchnorm2d82(x273)
        x275=self.relu79(x274)
        x276=self.conv2d83(x275)
        x277=self.batchnorm2d83(x276)
        x278=self.relu79(x277)
        x279=self.conv2d84(x278)
        x280=self.batchnorm2d84(x279)
        x281=operator.add(x280, x272)
        x282=self.relu79(x281)
        x283=self.conv2d85(x282)
        x284=self.batchnorm2d85(x283)
        x285=self.relu82(x284)
        x286=self.conv2d86(x285)
        x287=self.batchnorm2d86(x286)
        x288=self.relu82(x287)
        x289=self.conv2d87(x288)
        x290=self.batchnorm2d87(x289)
        x291=operator.add(x290, x282)
        x292=self.relu82(x291)
        x293=self.conv2d88(x292)
        x294=self.batchnorm2d88(x293)
        x295=self.relu85(x294)
        x296=self.conv2d89(x295)
        x297=self.batchnorm2d89(x296)
        x298=self.relu85(x297)
        x299=self.conv2d90(x298)
        x300=self.batchnorm2d90(x299)
        x301=operator.add(x300, x292)
        x302=self.relu85(x301)
        x303=self.conv2d91(x302)
        x304=self.batchnorm2d91(x303)
        x305=self.relu88(x304)
        x306=self.conv2d92(x305)
        x307=self.batchnorm2d92(x306)
        x308=self.relu88(x307)
        x309=self.conv2d93(x308)
        x310=self.batchnorm2d93(x309)
        x311=operator.add(x310, x302)
        x312=self.relu88(x311)
        x313=self.conv2d94(x312)
        x314=self.batchnorm2d94(x313)
        x315=self.relu91(x314)
        x316=self.conv2d95(x315)
        x317=self.batchnorm2d95(x316)
        x318=self.relu91(x317)
        x319=self.conv2d96(x318)
        x320=self.batchnorm2d96(x319)
        x321=self.conv2d97(x312)
        x322=self.batchnorm2d97(x321)
        x323=operator.add(x320, x322)
        x324=self.relu91(x323)
        x325=self.conv2d98(x324)
        x326=self.batchnorm2d98(x325)
        x327=self.relu94(x326)
        x328=self.conv2d99(x327)
        x329=self.batchnorm2d99(x328)
        x330=self.relu94(x329)
        x331=self.conv2d100(x330)
        x332=self.batchnorm2d100(x331)
        x333=operator.add(x332, x324)
        x334=self.relu94(x333)
        x335=self.conv2d101(x334)
        x336=self.batchnorm2d101(x335)
        x337=self.relu97(x336)
        x338=self.conv2d102(x337)
        x339=self.batchnorm2d102(x338)
        x340=self.relu97(x339)
        x341=self.conv2d103(x340)
        x342=self.batchnorm2d103(x341)
        x343=operator.add(x342, x334)
        x344=self.relu97(x343)
        x345=self.conv2d104(x344)
        x346=self.batchnorm2d104(x345)
        x347=self.relu100(x346)
        x348=self.dropout0(x347)
        x349=self.conv2d105(x348)
        x350=torch.nn.functional.interpolate(x349,size=x2, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)
        x351=self.conv2d106(x312)
        x352=self.batchnorm2d105(x351)
        x353=self.relu101(x352)
        x354=self.dropout1(x353)
        x355=self.conv2d107(x354)
        x356=torch.nn.functional.interpolate(x355,size=x2, scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None, antialias=False)

m = M().eval()
x = torch.rand(1, 3, 224, 224)
start = time.time()
output = m(x)
end = time.time()
print(end-start)
