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
        self.conv2d133 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x420, x415):
        x421=operator.mul(x420, x415)
        x422=self.conv2d133(x421)
        x423=self.batchnorm2d81(x422)
        return x423

m = M().eval()
x420 = torch.randn(torch.Size([1, 2904, 1, 1]))
x415 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x420, x415)
end = time.time()
print(end-start)