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
        self.relu67 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d74 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d75 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x239):
        x240=self.relu67(x239)
        x241=self.conv2d73(x240)
        x242=self.batchnorm2d73(x241)
        x243=self.relu70(x242)
        x244=self.conv2d74(x243)
        x245=self.batchnorm2d74(x244)
        x246=self.relu70(x245)
        x247=self.conv2d75(x246)
        x248=self.batchnorm2d75(x247)
        return x248

m = M().eval()
x239 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
