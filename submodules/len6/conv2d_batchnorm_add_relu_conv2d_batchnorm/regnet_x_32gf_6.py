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
        self.conv2d17 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x53, x47):
        x54=self.conv2d17(x53)
        x55=self.batchnorm2d17(x54)
        x56=operator.add(x47, x55)
        x57=self.relu15(x56)
        x58=self.conv2d18(x57)
        x59=self.batchnorm2d18(x58)
        return x59

m = M().eval()
x53 = torch.randn(torch.Size([1, 672, 28, 28]))
x47 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x53, x47)
end = time.time()
print(end-start)
