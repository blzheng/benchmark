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
        self.conv2d157 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid26 = Sigmoid()
        self.conv2d158 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x503, x500):
        x504=self.conv2d157(x503)
        x505=self.sigmoid26(x504)
        x506=operator.mul(x505, x500)
        x507=self.conv2d158(x506)
        x508=self.batchnorm2d104(x507)
        return x508

m = M().eval()
x503 = torch.randn(torch.Size([1, 76, 1, 1]))
x500 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x503, x500)
end = time.time()
print(end-start)
