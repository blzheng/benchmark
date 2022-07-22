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
        self.conv2d1 = Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)
        self.conv2d2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x4):
        x5=self.conv2d1(x4)
        x6=self.batchnorm2d1(x5)
        x7=self.relu1(x6)
        x8=self.conv2d2(x7)
        x9=self.batchnorm2d2(x8)
        x10=self.relu1(x9)
        return x10

m = M().eval()
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x4)
end = time.time()
print(end-start)
