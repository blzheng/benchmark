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
        self.conv2d52 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x162, x147):
        x163=operator.add(x162, x147)
        x164=self.conv2d52(x163)
        x165=self.batchnorm2d30(x164)
        return x165

m = M().eval()
x162 = torch.randn(torch.Size([1, 48, 56, 56]))
x147 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x162, x147)
end = time.time()
print(end-start)
