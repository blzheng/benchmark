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
        self.batchnorm2d74 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x381):
        x382=self.conv2d123(x381)
        x383=self.batchnorm2d73(x382)
        x384=self.conv2d124(x383)
        x385=self.batchnorm2d74(x384)
        return x385

m = M().eval()
x381 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x381)
end = time.time()
print(end-start)
