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
        self.sigmoid30 = Sigmoid()
        self.conv2d151 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x471, x467):
        x472=self.sigmoid30(x471)
        x473=operator.mul(x472, x467)
        x474=self.conv2d151(x473)
        x475=self.batchnorm2d89(x474)
        return x475

m = M().eval()
x471 = torch.randn(torch.Size([1, 1344, 1, 1]))
x467 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x471, x467)
end = time.time()
print(end-start)
