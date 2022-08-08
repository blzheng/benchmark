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
        self.conv2d4 = Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d4 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu2 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d6 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x12):
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d4(x13)
        x15=self.relu2(x14)
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        x18=self.conv2d6(x17)
        return x18

m = M().eval()
x12 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
