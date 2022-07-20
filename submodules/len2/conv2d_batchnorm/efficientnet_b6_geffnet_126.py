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
        self.conv2d212 = Conv2d(2064, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x633):
        x634=self.conv2d212(x633)
        x635=self.batchnorm2d126(x634)
        return x635

m = M().eval()
x633 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x633)
end = time.time()
print(end-start)
