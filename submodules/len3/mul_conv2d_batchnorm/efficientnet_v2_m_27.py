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
        self.conv2d163 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x521, x516):
        x522=operator.mul(x521, x516)
        x523=self.conv2d163(x522)
        x524=self.batchnorm2d107(x523)
        return x524

m = M().eval()
x521 = torch.randn(torch.Size([1, 1824, 1, 1]))
x516 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x521, x516)
end = time.time()
print(end-start)
