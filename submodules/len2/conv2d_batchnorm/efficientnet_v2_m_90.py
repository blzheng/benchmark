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
        self.conv2d134 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x430):
        x431=self.conv2d134(x430)
        x432=self.batchnorm2d90(x431)
        return x432

m = M().eval()
x430 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x430)
end = time.time()
print(end-start)
