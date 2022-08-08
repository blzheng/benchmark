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
        self.conv2d16 = Conv2d(32, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=96, bias=False)
        self.batchnorm2d17 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x51):
        x52=self.conv2d16(x51)
        x53=self.batchnorm2d16(x52)
        x54=self.relu11(x53)
        x55=self.conv2d17(x54)
        x56=self.batchnorm2d17(x55)
        x57=self.relu12(x56)
        return x57

m = M().eval()
x51 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)