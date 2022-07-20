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
        self.sigmoid34 = Sigmoid()
        self.conv2d198 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d128 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x632, x628):
        x633=self.sigmoid34(x632)
        x634=operator.mul(x633, x628)
        x635=self.conv2d198(x634)
        x636=self.batchnorm2d128(x635)
        return x636

m = M().eval()
x632 = torch.randn(torch.Size([1, 1824, 1, 1]))
x628 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x632, x628)
end = time.time()
print(end-start)
