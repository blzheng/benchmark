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
        self.batchnorm2d28 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(352, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x101):
        x102=self.batchnorm2d28(x101)
        x103=self.relu28(x102)
        x104=self.conv2d28(x103)
        return x104

m = M().eval()
x101 = torch.randn(torch.Size([1, 352, 28, 28]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
