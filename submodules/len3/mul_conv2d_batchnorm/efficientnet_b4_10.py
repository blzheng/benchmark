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
        self.conv2d53 = Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x162, x157):
        x163=operator.mul(x162, x157)
        x164=self.conv2d53(x163)
        x165=self.batchnorm2d31(x164)
        return x165

m = M().eval()
x162 = torch.randn(torch.Size([1, 336, 1, 1]))
x157 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x162, x157)
end = time.time()
print(end-start)
