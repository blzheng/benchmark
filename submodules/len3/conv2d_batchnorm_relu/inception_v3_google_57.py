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
        self.conv2d57 = Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d57 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x198):
        x199=self.conv2d57(x198)
        x200=self.batchnorm2d57(x199)
        x201=torch.nn.functional.relu(x200,inplace=True)
        return x201

m = M().eval()
x198 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
