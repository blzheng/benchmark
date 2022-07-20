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
        self.conv2d74 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x236, x221):
        x237=operator.add(x236, x221)
        x238=self.conv2d74(x237)
        x239=self.batchnorm2d52(x238)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 160, 14, 14]))
x221 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x236, x221)
end = time.time()
print(end-start)
