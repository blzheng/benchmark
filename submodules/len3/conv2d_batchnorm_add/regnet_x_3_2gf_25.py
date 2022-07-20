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
        self.conv2d72 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x235, x229):
        x236=self.conv2d72(x235)
        x237=self.batchnorm2d72(x236)
        x238=operator.add(x229, x237)
        return x238

m = M().eval()
x235 = torch.randn(torch.Size([1, 432, 14, 14]))
x229 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x235, x229)
end = time.time()
print(end-start)
