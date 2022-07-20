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
        self.conv2d60 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x195, x189):
        x196=self.conv2d60(x195)
        x197=self.batchnorm2d60(x196)
        x198=operator.add(x189, x197)
        x199=self.relu57(x198)
        x200=self.conv2d61(x199)
        return x200

m = M().eval()
x195 = torch.randn(torch.Size([1, 432, 14, 14]))
x189 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x195, x189)
end = time.time()
print(end-start)
