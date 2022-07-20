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
        self.conv2d149 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d99 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x477, x462):
        x478=operator.add(x477, x462)
        x479=self.conv2d149(x478)
        x480=self.batchnorm2d99(x479)
        return x480

m = M().eval()
x477 = torch.randn(torch.Size([1, 304, 7, 7]))
x462 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x477, x462)
end = time.time()
print(end-start)
