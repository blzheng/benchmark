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
        self.batchnorm2d65 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(1008, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)
        self.conv2d66 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x232):
        x233=self.batchnorm2d65(x232)
        x234=self.relu65(x233)
        x235=self.conv2d65(x234)
        x236=self.batchnorm2d66(x235)
        x237=self.relu66(x236)
        x238=self.conv2d66(x237)
        return x238

m = M().eval()
x232 = torch.randn(torch.Size([1, 1008, 14, 14]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)
