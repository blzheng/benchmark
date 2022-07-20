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
        self.conv2d3 = Conv2d(8, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x8):
        x9=self.conv2d3(x8)
        x10=self.batchnorm2d3(x9)
        return x10

m = M().eval()
x8 = torch.randn(torch.Size([1, 8, 112, 112]))
start = time.time()
output = m(x8)
end = time.time()
print(end-start)
