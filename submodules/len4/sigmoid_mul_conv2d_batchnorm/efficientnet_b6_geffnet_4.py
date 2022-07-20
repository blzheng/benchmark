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
        self.conv2d22 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x65, x61):
        x66=x65.sigmoid()
        x67=operator.mul(x61, x66)
        x68=self.conv2d22(x67)
        x69=self.batchnorm2d12(x68)
        return x69

m = M().eval()
x65 = torch.randn(torch.Size([1, 240, 1, 1]))
x61 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x65, x61)
end = time.time()
print(end-start)
