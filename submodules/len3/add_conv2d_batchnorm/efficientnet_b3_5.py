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
        self.conv2d49 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148, x133):
        x149=operator.add(x148, x133)
        x150=self.conv2d49(x149)
        x151=self.batchnorm2d29(x150)
        return x151

m = M().eval()
x148 = torch.randn(torch.Size([1, 96, 14, 14]))
x133 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x148, x133)
end = time.time()
print(end-start)
