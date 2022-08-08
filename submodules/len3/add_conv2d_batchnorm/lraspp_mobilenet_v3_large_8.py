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
        self.conv2d56 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x162, x148):
        x163=operator.add(x162, x148)
        x164=self.conv2d56(x163)
        x165=self.batchnorm2d42(x164)
        return x165

m = M().eval()
x162 = torch.randn(torch.Size([1, 160, 14, 14]))
x148 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x162, x148)
end = time.time()
print(end-start)
