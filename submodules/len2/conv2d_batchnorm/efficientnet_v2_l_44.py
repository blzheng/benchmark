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
        self.conv2d52 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x175):
        x176=self.conv2d52(x175)
        x177=self.batchnorm2d44(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
