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
        self.relu42 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x139, x147):
        x148=operator.add(x139, x147)
        x149=self.relu42(x148)
        x150=self.conv2d46(x149)
        x151=self.batchnorm2d46(x150)
        return x151

m = M().eval()
x139 = torch.randn(torch.Size([1, 432, 14, 14]))
x147 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x139, x147)
end = time.time()
print(end-start)
