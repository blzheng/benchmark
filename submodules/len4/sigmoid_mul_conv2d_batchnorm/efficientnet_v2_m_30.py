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
        self.sigmoid30 = Sigmoid()
        self.conv2d178 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x568, x564):
        x569=self.sigmoid30(x568)
        x570=operator.mul(x569, x564)
        x571=self.conv2d178(x570)
        x572=self.batchnorm2d116(x571)
        return x572

m = M().eval()
x568 = torch.randn(torch.Size([1, 1824, 1, 1]))
x564 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x568, x564)
end = time.time()
print(end-start)
