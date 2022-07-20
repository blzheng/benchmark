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
        self.conv2d13 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x38, x34):
        x39=x38.sigmoid()
        x40=operator.mul(x34, x39)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d7(x41)
        return x42

m = M().eval()
x38 = torch.randn(torch.Size([1, 144, 1, 1]))
x34 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x38, x34)
end = time.time()
print(end-start)
