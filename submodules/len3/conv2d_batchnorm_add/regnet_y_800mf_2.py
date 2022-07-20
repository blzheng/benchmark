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
        self.conv2d7 = Conv2d(64, 144, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d5 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21, x37):
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d5(x22)
        x38=operator.add(x23, x37)
        return x38

m = M().eval()
x21 = torch.randn(torch.Size([1, 64, 56, 56]))
x37 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x21, x37)
end = time.time()
print(end-start)
