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
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d124 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x359, x364):
        x365=operator.mul(x359, x364)
        x366=self.conv2d123(x365)
        x367=self.batchnorm2d73(x366)
        x368=self.conv2d124(x367)
        return x368

m = M().eval()
x359 = torch.randn(torch.Size([1, 1392, 7, 7]))
x364 = torch.randn(torch.Size([1, 1392, 1, 1]))
start = time.time()
output = m(x359, x364)
end = time.time()
print(end-start)
