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
        self.conv2d77 = Conv2d(30, 720, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d78 = Conv2d(720, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x236, x233):
        x237=self.conv2d77(x236)
        x238=self.sigmoid15(x237)
        x239=operator.mul(x238, x233)
        x240=self.conv2d78(x239)
        x241=self.batchnorm2d46(x240)
        return x241

m = M().eval()
x236 = torch.randn(torch.Size([1, 30, 1, 1]))
x233 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x236, x233)
end = time.time()
print(end-start)
