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
        self.conv2d60 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x198, x192):
        x199=self.conv2d60(x198)
        x200=self.batchnorm2d60(x199)
        x201=operator.add(x200, x192)
        return x201

m = M().eval()
x198 = torch.randn(torch.Size([1, 256, 28, 28]))
x192 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x198, x192)
end = time.time()
print(end-start)
