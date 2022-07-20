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
        self.batchnorm2d72 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(2520, 2520, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x235):
        x236=self.batchnorm2d72(x235)
        x237=self.relu68(x236)
        x238=self.conv2d73(x237)
        x239=self.batchnorm2d73(x238)
        return x239

m = M().eval()
x235 = torch.randn(torch.Size([1, 2520, 7, 7]))
start = time.time()
output = m(x235)
end = time.time()
print(end-start)
