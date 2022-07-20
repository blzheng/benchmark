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
        self.conv2d19 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x60, x51):
        x61=operator.add(x60, x51)
        x62=self.conv2d19(x61)
        x63=self.batchnorm2d19(x62)
        return x63

m = M().eval()
x60 = torch.randn(torch.Size([1, 32, 28, 28]))
x51 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x60, x51)
end = time.time()
print(end-start)
