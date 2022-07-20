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
        self.batchnorm2d44 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d61 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x176, x163):
        x177=self.batchnorm2d44(x176)
        x178=operator.add(x177, x163)
        x179=self.conv2d61(x178)
        x180=self.batchnorm2d45(x179)
        return x180

m = M().eval()
x176 = torch.randn(torch.Size([1, 160, 7, 7]))
x163 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x176, x163)
end = time.time()
print(end-start)
