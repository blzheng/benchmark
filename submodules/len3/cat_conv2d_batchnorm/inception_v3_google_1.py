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
        self.conv2d12 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x33, x39, x48, x52):
        x53=torch.cat([x33, x39, x48, x52], 1)
        x54=self.conv2d12(x53)
        x55=self.batchnorm2d12(x54)
        return x55

m = M().eval()
x33 = torch.randn(torch.Size([1, 64, 25, 25]))
x39 = torch.randn(torch.Size([1, 64, 25, 25]))
x48 = torch.randn(torch.Size([1, 96, 25, 25]))
x52 = torch.randn(torch.Size([1, 32, 25, 25]))
start = time.time()
output = m(x33, x39, x48, x52)
end = time.time()
print(end-start)
