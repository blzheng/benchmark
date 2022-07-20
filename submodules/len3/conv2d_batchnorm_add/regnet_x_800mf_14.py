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
        self.conv2d37 = Conv2d(288, 672, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d37 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x119, x129):
        x120=self.conv2d37(x119)
        x121=self.batchnorm2d37(x120)
        x130=operator.add(x121, x129)
        return x130

m = M().eval()
x119 = torch.randn(torch.Size([1, 288, 14, 14]))
x129 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x119, x129)
end = time.time()
print(end-start)
