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
        self.batchnorm2d105 = BatchNorm2d(304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d178 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x529, x516):
        x530=self.batchnorm2d105(x529)
        x531=operator.add(x530, x516)
        x532=self.conv2d178(x531)
        return x532

m = M().eval()
x529 = torch.randn(torch.Size([1, 304, 7, 7]))
x516 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x529, x516)
end = time.time()
print(end-start)
