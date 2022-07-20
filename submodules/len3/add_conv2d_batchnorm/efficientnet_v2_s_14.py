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
        self.conv2d64 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x204, x189):
        x205=operator.add(x204, x189)
        x206=self.conv2d64(x205)
        x207=self.batchnorm2d46(x206)
        return x207

m = M().eval()
x204 = torch.randn(torch.Size([1, 160, 14, 14]))
x189 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x204, x189)
end = time.time()
print(end-start)
