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
        self.conv2d29 = Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x102, x96):
        x103=operator.add(x102, x96)
        x104=self.conv2d29(x103)
        x105=self.batchnorm2d29(x104)
        return x105

m = M().eval()
x102 = torch.randn(torch.Size([1, 96, 28, 28]))
x96 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x102, x96)
end = time.time()
print(end-start)
