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
        self.conv2d34 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x99, x94):
        x100=operator.mul(x99, x94)
        x101=self.conv2d34(x100)
        x102=self.batchnorm2d20(x101)
        return x102

m = M().eval()
x99 = torch.randn(torch.Size([1, 480, 1, 1]))
x94 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x99, x94)
end = time.time()
print(end-start)
