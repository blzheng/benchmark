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
        self.conv2d132 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d133 = Conv2d(1056, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d134 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x425, x422):
        x426=self.conv2d132(x425)
        x427=self.sigmoid21(x426)
        x428=operator.mul(x427, x422)
        x429=self.conv2d133(x428)
        x430=self.batchnorm2d89(x429)
        x431=self.conv2d134(x430)
        return x431

m = M().eval()
x425 = torch.randn(torch.Size([1, 44, 1, 1]))
x422 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x425, x422)
end = time.time()
print(end-start)
