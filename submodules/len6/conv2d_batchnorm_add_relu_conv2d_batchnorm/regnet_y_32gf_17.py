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
        self.conv2d78 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x245, x233):
        x246=self.conv2d78(x245)
        x247=self.batchnorm2d48(x246)
        x248=operator.add(x233, x247)
        x249=self.relu60(x248)
        x250=self.conv2d79(x249)
        x251=self.batchnorm2d49(x250)
        return x251

m = M().eval()
x245 = torch.randn(torch.Size([1, 1392, 14, 14]))
x233 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x245, x233)
end = time.time()
print(end-start)
