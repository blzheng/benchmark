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
        self.conv2d45 = Conv2d(1056, 264, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d46 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d47 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x142, x141, x135):
        x143=self.conv2d45(x142)
        x144=self.relu35(x143)
        x145=self.conv2d46(x144)
        x146=self.sigmoid8(x145)
        x147=operator.mul(x146, x141)
        x148=self.conv2d47(x147)
        x149=self.batchnorm2d29(x148)
        x150=operator.add(x135, x149)
        return x150

m = M().eval()
x142 = torch.randn(torch.Size([1, 1056, 1, 1]))
x141 = torch.randn(torch.Size([1, 1056, 28, 28]))
x135 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x142, x141, x135)
end = time.time()
print(end-start)
