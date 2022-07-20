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
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x88, x103):
        x89=self.batchnorm2d20(x88)
        x104=operator.add(x103, x89)
        x105=self.conv2d36(x104)
        x106=self.batchnorm2d24(x105)
        return x106

m = M().eval()
x88 = torch.randn(torch.Size([1, 48, 14, 14]))
x103 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x88, x103)
end = time.time()
print(end-start)
