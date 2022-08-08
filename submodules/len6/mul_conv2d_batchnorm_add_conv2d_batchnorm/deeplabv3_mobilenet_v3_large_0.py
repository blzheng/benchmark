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
        self.conv2d23 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x68, x63, x57):
        x69=operator.mul(x68, x63)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d17(x70)
        x72=operator.add(x71, x57)
        x73=self.conv2d24(x72)
        x74=self.batchnorm2d18(x73)
        return x74

m = M().eval()
x68 = torch.randn(torch.Size([1, 120, 1, 1]))
x63 = torch.randn(torch.Size([1, 120, 28, 28]))
x57 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x68, x63, x57)
end = time.time()
print(end-start)
