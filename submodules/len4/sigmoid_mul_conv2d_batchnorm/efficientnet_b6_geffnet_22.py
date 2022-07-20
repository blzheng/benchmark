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
        self.conv2d112 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x333, x329):
        x334=x333.sigmoid()
        x335=operator.mul(x329, x334)
        x336=self.conv2d112(x335)
        x337=self.batchnorm2d66(x336)
        return x337

m = M().eval()
x333 = torch.randn(torch.Size([1, 864, 1, 1]))
x329 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x333, x329)
end = time.time()
print(end-start)
