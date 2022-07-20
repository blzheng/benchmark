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
        self.conv2d9 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d15 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x26, x43):
        x27=self.conv2d9(x26)
        x28=self.batchnorm2d5(x27)
        x44=operator.add(x43, x28)
        x45=self.conv2d15(x44)
        x46=self.batchnorm2d9(x45)
        return x46

m = M().eval()
x26 = torch.randn(torch.Size([1, 96, 56, 56]))
x43 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x26, x43)
end = time.time()
print(end-start)
