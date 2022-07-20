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
        self.conv2d42 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x125, x121):
        x126=x125.sigmoid()
        x127=operator.mul(x121, x126)
        x128=self.conv2d42(x127)
        x129=self.batchnorm2d24(x128)
        return x129

m = M().eval()
x125 = torch.randn(torch.Size([1, 240, 1, 1]))
x121 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x125, x121)
end = time.time()
print(end-start)
