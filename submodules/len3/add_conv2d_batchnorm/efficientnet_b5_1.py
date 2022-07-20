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
        self.conv2d13 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x39, x27):
        x40=operator.add(x39, x27)
        x41=self.conv2d13(x40)
        x42=self.batchnorm2d7(x41)
        return x42

m = M().eval()
x39 = torch.randn(torch.Size([1, 24, 112, 112]))
x27 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x39, x27)
end = time.time()
print(end-start)
