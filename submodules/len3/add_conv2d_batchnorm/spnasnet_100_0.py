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
        self.conv2d9 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x28, x19):
        x29=operator.add(x28, x19)
        x30=self.conv2d9(x29)
        x31=self.batchnorm2d9(x30)
        return x31

m = M().eval()
x28 = torch.randn(torch.Size([1, 24, 56, 56]))
x19 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x28, x19)
end = time.time()
print(end-start)
