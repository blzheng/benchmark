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
        self.conv2d52 = Conv2d(408, 912, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d52 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169):
        x170=self.conv2d52(x169)
        x171=self.batchnorm2d52(x170)
        return x171

m = M().eval()
x169 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
