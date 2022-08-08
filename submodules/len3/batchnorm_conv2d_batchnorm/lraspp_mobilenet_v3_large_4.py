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
        self.batchnorm2d38 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x147):
        x148=self.batchnorm2d38(x147)
        x149=self.conv2d51(x148)
        x150=self.batchnorm2d39(x149)
        return x150

m = M().eval()
x147 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
