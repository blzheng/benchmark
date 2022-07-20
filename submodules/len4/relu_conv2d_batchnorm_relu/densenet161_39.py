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
        self.relu81 = ReLU(inplace=True)
        self.conv2d81 = Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)

    def forward(self, x289):
        x290=self.relu81(x289)
        x291=self.conv2d81(x290)
        x292=self.batchnorm2d82(x291)
        x293=self.relu82(x292)
        return x293

m = M().eval()
x289 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x289)
end = time.time()
print(end-start)
