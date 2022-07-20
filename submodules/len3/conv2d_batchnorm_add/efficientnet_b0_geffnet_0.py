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
        self.conv2d9 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27, x43):
        x28=self.conv2d9(x27)
        x29=self.batchnorm2d5(x28)
        x44=operator.add(x43, x29)
        return x44

m = M().eval()
x27 = torch.randn(torch.Size([1, 96, 56, 56]))
x43 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x27, x43)
end = time.time()
print(end-start)
