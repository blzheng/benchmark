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
        self.conv2d182 = Conv2d(1824, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(512, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x569, x564):
        x570=operator.mul(x569, x564)
        x571=self.conv2d182(x570)
        x572=self.batchnorm2d108(x571)
        x573=self.conv2d183(x572)
        return x573

m = M().eval()
x569 = torch.randn(torch.Size([1, 1824, 1, 1]))
x564 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x569, x564)
end = time.time()
print(end-start)
