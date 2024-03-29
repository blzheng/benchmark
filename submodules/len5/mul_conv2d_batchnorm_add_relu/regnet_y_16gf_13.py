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
        self.conv2d73 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)

    def forward(self, x228, x223, x217):
        x229=operator.mul(x228, x223)
        x230=self.conv2d73(x229)
        x231=self.batchnorm2d45(x230)
        x232=operator.add(x217, x231)
        x233=self.relu56(x232)
        return x233

m = M().eval()
x228 = torch.randn(torch.Size([1, 1232, 1, 1]))
x223 = torch.randn(torch.Size([1, 1232, 14, 14]))
x217 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x228, x223, x217)
end = time.time()
print(end-start)
