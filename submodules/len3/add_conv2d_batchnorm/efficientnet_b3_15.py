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
        self.conv2d109 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x336, x321):
        x337=operator.add(x336, x321)
        x338=self.conv2d109(x337)
        x339=self.batchnorm2d65(x338)
        return x339

m = M().eval()
x336 = torch.randn(torch.Size([1, 232, 7, 7]))
x321 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x336, x321)
end = time.time()
print(end-start)
