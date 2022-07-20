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
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x66):
        x67=self.batchnorm2d16(x66)
        x68=torch.nn.functional.relu(x67,inplace=True)
        x69=self.conv2d17(x68)
        x70=self.batchnorm2d17(x69)
        return x70

m = M().eval()
x66 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
