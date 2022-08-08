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
        self.relu37 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x141):
        x142=self.relu37(x141)
        x143=self.conv2d43(x142)
        x144=self.batchnorm2d43(x143)
        x145=self.relu40(x144)
        return x145

m = M().eval()
x141 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x141)
end = time.time()
print(end-start)
