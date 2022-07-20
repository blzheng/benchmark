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
        self.relu28 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x108, x100):
        x109=operator.add(x108, x100)
        x110=self.relu28(x109)
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        return x112

m = M().eval()
x108 = torch.randn(torch.Size([1, 1024, 14, 14]))
x100 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x108, x100)
end = time.time()
print(end-start)
