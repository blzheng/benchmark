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
        self.conv2d55 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x177, x168):
        x178=operator.add(x177, x168)
        x179=self.conv2d55(x178)
        x180=self.batchnorm2d55(x179)
        return x180

m = M().eval()
x177 = torch.randn(torch.Size([1, 184, 7, 7]))
x168 = torch.randn(torch.Size([1, 184, 7, 7]))
start = time.time()
output = m(x177, x168)
end = time.time()
print(end-start)
