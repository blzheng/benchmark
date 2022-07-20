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
        self.conv2d160 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid32 = Sigmoid()
        self.conv2d161 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x502, x499):
        x503=self.conv2d160(x502)
        x504=self.sigmoid32(x503)
        x505=operator.mul(x504, x499)
        x506=self.conv2d161(x505)
        x507=self.batchnorm2d95(x506)
        return x507

m = M().eval()
x502 = torch.randn(torch.Size([1, 56, 1, 1]))
x499 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x502, x499)
end = time.time()
print(end-start)
