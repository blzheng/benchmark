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
        self.conv2d20 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x63, x57):
        x64=self.conv2d20(x63)
        x65=self.batchnorm2d20(x64)
        x66=operator.add(x57, x65)
        x67=self.relu18(x66)
        x68=self.conv2d21(x67)
        return x68

m = M().eval()
x63 = torch.randn(torch.Size([1, 192, 28, 28]))
x57 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x63, x57)
end = time.time()
print(end-start)
