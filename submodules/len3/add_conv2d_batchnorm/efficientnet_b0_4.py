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
        self.conv2d50 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x149, x134):
        x150=operator.add(x149, x134)
        x151=self.conv2d50(x150)
        x152=self.batchnorm2d30(x151)
        return x152

m = M().eval()
x149 = torch.randn(torch.Size([1, 112, 14, 14]))
x134 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x149, x134)
end = time.time()
print(end-start)
