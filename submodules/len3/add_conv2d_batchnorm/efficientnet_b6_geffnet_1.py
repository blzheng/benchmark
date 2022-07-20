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
        self.conv2d13 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x40, x28):
        x41=operator.add(x40, x28)
        x42=self.conv2d13(x41)
        x43=self.batchnorm2d7(x42)
        return x43

m = M().eval()
x40 = torch.randn(torch.Size([1, 32, 112, 112]))
x28 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x40, x28)
end = time.time()
print(end-start)
