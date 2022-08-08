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
        self.conv2d2 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x6):
        x7=self.conv2d2(x6)
        x8=self.batchnorm2d2(x7)
        return x8

m = M().eval()
x6 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
