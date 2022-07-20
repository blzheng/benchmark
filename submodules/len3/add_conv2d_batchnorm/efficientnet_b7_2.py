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
        self.conv2d17 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(192, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x52, x40):
        x53=operator.add(x52, x40)
        x54=self.conv2d17(x53)
        x55=self.batchnorm2d9(x54)
        return x55

m = M().eval()
x52 = torch.randn(torch.Size([1, 32, 112, 112]))
x40 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x52, x40)
end = time.time()
print(end-start)
