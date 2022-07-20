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
        self.conv2d63 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x205, x199):
        x206=self.conv2d63(x205)
        x207=self.batchnorm2d63(x206)
        x208=operator.add(x199, x207)
        return x208

m = M().eval()
x205 = torch.randn(torch.Size([1, 896, 14, 14]))
x199 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x205, x199)
end = time.time()
print(end-start)
