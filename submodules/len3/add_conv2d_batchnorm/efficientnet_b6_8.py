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
        self.conv2d58 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(432, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x179, x164):
        x180=operator.add(x179, x164)
        x181=self.conv2d58(x180)
        x182=self.batchnorm2d34(x181)
        return x182

m = M().eval()
x179 = torch.randn(torch.Size([1, 72, 28, 28]))
x164 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x179, x164)
end = time.time()
print(end-start)
