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
        self.batchnorm2d108 = BatchNorm2d(512, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x571):
        x572=self.batchnorm2d108(x571)
        x573=self.conv2d183(x572)
        return x573

m = M().eval()
x571 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x571)
end = time.time()
print(end-start)
