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
        self.conv2d26 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x101):
        x102=self.conv2d26(x101)
        x103=self.batchnorm2d26(x102)
        x104=torch.nn.functional.relu(x103,inplace=True)
        return x104

m = M().eval()
x101 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
