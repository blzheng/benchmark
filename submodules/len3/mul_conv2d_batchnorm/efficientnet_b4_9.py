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
        self.conv2d48 = Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x146, x141):
        x147=operator.mul(x146, x141)
        x148=self.conv2d48(x147)
        x149=self.batchnorm2d28(x148)
        return x149

m = M().eval()
x146 = torch.randn(torch.Size([1, 336, 1, 1]))
x141 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x146, x141)
end = time.time()
print(end-start)
