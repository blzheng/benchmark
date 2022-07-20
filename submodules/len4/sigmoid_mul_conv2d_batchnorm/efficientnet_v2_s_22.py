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
        self.sigmoid22 = Sigmoid()
        self.conv2d133 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x421, x417):
        x422=self.sigmoid22(x421)
        x423=operator.mul(x422, x417)
        x424=self.conv2d133(x423)
        x425=self.batchnorm2d87(x424)
        return x425

m = M().eval()
x421 = torch.randn(torch.Size([1, 1536, 1, 1]))
x417 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x421, x417)
end = time.time()
print(end-start)
